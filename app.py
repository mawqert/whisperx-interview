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

load_dotenv(Path(__file__).with_name(".env"))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024**3
transcribing = threading.Lock()


def speaker_blocks(segments):
    blocks = []
    for segment in segments:
        text = segment.get('text', '').strip()
        if not text:
            continue
        speaker = segment.get('speaker')
        if blocks and speaker and blocks[-1]['speaker'] == speaker:
            blocks[-1]['text'] += ' ' + text
            blocks[-1]['end'] = max(blocks[-1]['end'], segment['end'])
        else:
            blocks.append(dict(speaker=speaker, start=segment['start'],
                               end=segment['end'], text=text))
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
            # Read the token in the child so it is not exposed in process arguments.
            command = [sys.executable, str(Path(__file__).with_name('transcribe_worker.py')),
                       str(source), '--device', 'cuda',
                       '--task', 'transcribe', '--language', 'ru',
                       '--compute_type', 'int8', '--batch_size', '8', '--diarize',
                       '--output_format', 'json', '--output_dir', str(work)]
            for name, value in speakers.items():
                command.extend(['--' + name, str(value)])
            with (work / 'run.log').open('w+') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                        env=env, timeout=6 * 60 * 60)
                if result.returncode:
                    log.seek(0)
                    failed_log = log.read()
                    # A Hub metadata request can fail even when models are cached.
                    # Retry in a fresh process so every HF library sees offline mode.
                    if 'huggingface.co' in failed_log and any(marker in failed_log for marker in (
                            'SSLError', 'ConnectionError', 'ConnectTimeout', 'ReadTimeout')):
                        env['HF_HUB_OFFLINE'] = '1'
                        log.seek(0)
                        log.truncate()
                        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                                env=env, timeout=6 * 60 * 60)
                if result.returncode:
                    log.seek(0, 2)
                    log.seek(max(0, log.tell() - 12000))
                    detail = log.read().replace(token, '[redacted]')
                    message = 'WhisperX failed.'
                    if 'OfflineModeIsEnabled' in detail or 'LocalEntryNotFoundError' in detail:
                        message = ('Cannot reach Hugging Face and required model files are not cached. '
                                   'Restore access to huggingface.co and retry to download them.')
                    return jsonify(error=message, detail=detail), 500
            result = json.loads(source.with_suffix('.json').read_text())
            result['blocks'] = speaker_blocks(result.get('segments', []))
            return jsonify(result)
    except subprocess.TimeoutExpired:
        return jsonify(error='Transcription exceeded the six-hour time limit.'), 504
    except Exception as exc:
        return jsonify(error=str(exc).replace(token, '[redacted]')), 500
    finally:
        transcribing.release()


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, threaded=True)
