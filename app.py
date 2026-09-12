import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading

from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
from run_logging import log as log_event

load_dotenv(Path(__file__).with_name(".env"))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024**3
transcribing = threading.Lock()


def run_worker(command, *, stdout, stderr, env, timeout):
    """Stream worker output to the terminal and retain it for error responses."""
    token = env.get('HF_TOKEN', '').strip()
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr,
                          env={**env, 'PYTHONUNBUFFERED': '1'}, text=True,
                          errors='replace') as process:
        def forward():
            for line in process.stdout:
                if token:
                    line = line.replace(token, '[redacted]')
                stdout.write(line)
                stdout.flush()
                print(line, end='', flush=True)

        reader = threading.Thread(target=forward, daemon=True)
        reader.start()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise
        finally:
            reader.join()
        return subprocess.CompletedProcess(command, process.returncode)


def speaker_blocks(segments):
    """Preserve sentence boundaries and speaker changes within ASR segments."""
    blocks = []
    for segment in segments:
        text = segment.get('text', '').strip()
        if not text:
            continue
        words = [word for word in segment.get('words', []) if word.get('word', '').strip()]
        # Some alignment failures leave incomplete word lists. Keep the full
        # transcript in that case instead of silently dropping unaligned text.
        if not words or ''.join(text.split()) != ''.join(
                ''.join(word['word'].split()) for word in words):
            blocks.append(dict(speaker=segment.get('speaker'), start=segment['start'],
                               end=segment['end'], text=text))
            continue

        block = None
        previous_end = segment['start']
        for word in words:
            # Missing labels remain unknown; the segment's majority speaker
            # is not reliable evidence for an individual word.
            speaker = word.get('speaker')
            start = word.get('start', previous_end)
            end = word.get('end', start)
            if (block is None or speaker != block['speaker']
                    or start - previous_end > 1.5 or end - block['start'] > 30):
                block = dict(speaker=speaker, start=start, end=end, text=word['word'].strip())
                blocks.append(block)
            else:
                block['text'] += ' ' + word['word'].strip()
                block['end'] = max(block['end'], end)
            previous_end = end
    return blocks


@app.get('/')
def index():
    return render_template('index.html')


@app.errorhandler(413)
def too_large(_):
    return jsonify(error='File is too large. Maximum size is 2 GB.'), 413


@app.post('/transcribe')
def transcribe():
    upload = request.files.get('file')
    if not upload or not upload.filename:
        return jsonify(error='Choose an audio or video file.'), 400
    speakers = {}
    for name in ('min_speakers', 'max_speakers'):
        value = request.form.get(name, '').strip()
        if value:
            try:
                speakers[name] = int(value)
                if speakers[name] < 1:
                    raise ValueError
            except ValueError:
                return jsonify(error=f'{name} must be a positive integer.'), 400
    if speakers.get('min_speakers', 1) > speakers.get('max_speakers', float('inf')):
        return jsonify(error='Minimum speakers must not exceed maximum speakers.'), 400
    token = os.environ.get('HF_TOKEN', '').strip()
    if not token:
        return jsonify(error='Set HF_TOKEN in .env or the server environment and restart the app.'), 503
    if not transcribing.acquire(blocking=False):
        return jsonify(error='A transcription is already running. Try again when it finishes.'), 409
    try:
        with tempfile.TemporaryDirectory(prefix='whisperx-') as directory:
            work = Path(directory)
            source = work / ('upload' + Path(upload.filename).suffix)
            upload.save(source)
            env = os.environ.copy()
            # WhisperX invokes ffmpeg by name. Use the bundled binary if needed.
            if not shutil.which('ffmpeg'):
                import imageio_ffmpeg
                (work / 'ffmpeg').symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
                env['PATH'] = str(work) + os.pathsep + env.get('PATH', '')
            # The worker calls the Python API; JSON carries job options, while
            # the token stays in the environment and out of process arguments.
            job = work / 'job.json'
            output = work / 'result.json'
            job.write_text(json.dumps({'audio_path': str(source),
                                       'output_path': str(output),
                                       **speakers}), encoding='utf-8')
            command = [sys.executable, str(Path(__file__).with_name('transcribe_worker.py')),
                       str(job)]
            with (work / 'run.log').open('w+') as log:
                result = run_worker(command, stdout=log, stderr=subprocess.STDOUT,
                                        env=env, timeout=6 * 60 * 60)
                if result.returncode:
                    log.seek(0)
                    failed_log = log.read()
                    # A Hub metadata request can fail even when models are cached.
                    # Retry in a fresh process so every HF library sees offline mode.
                    if 'huggingface.co' in failed_log and any(marker in failed_log for marker in (
                            'SSLError', 'ConnectionError', 'ConnectTimeout', 'ReadTimeout')):
                        env['HF_HUB_OFFLINE'] = '1'
                        log_event('RETRY using cached models in offline mode')
                        log.seek(0)
                        log.truncate()
                        result = run_worker(command, stdout=log, stderr=subprocess.STDOUT,
                                                env=env, timeout=6 * 60 * 60)
                if result.returncode:
                    log_event(f'ERROR worker exited with code {result.returncode}')
                    log.seek(0, 2)
                    log.seek(max(0, log.tell() - 12000))
                    detail = log.read().replace(token, '[redacted]')
                    message = 'WhisperX failed.'
                    if 'OfflineModeIsEnabled' in detail or 'LocalEntryNotFoundError' in detail:
                        message = ('Cannot reach Hugging Face and required model files are not cached. '
                                   'Restore access to huggingface.co and retry to download them.')
                    return jsonify(error=message, detail=detail), 500
            result = json.loads(output.read_text(encoding='utf-8'))
            result['blocks'] = speaker_blocks(result.get('segments', []))
            return jsonify(result)
    except subprocess.TimeoutExpired:
        log_event('ERROR transcription exceeded the six-hour time limit; worker stopped')
        return jsonify(error='Transcription exceeded the six-hour time limit.'), 504
    except Exception as exc:
        log_event(f'ERROR transcription: {str(exc).replace(token, "[redacted]")}')
        return jsonify(error=str(exc).replace(token, '[redacted]')), 500
    finally:
        transcribing.release()


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, threaded=True)
