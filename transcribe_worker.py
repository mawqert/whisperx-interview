"""Run WhisperX's Python pipeline in an isolated, time-limited worker."""

import gc
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import threading
import time

from model_config import runtime_settings
from runtime_setup import load_runtime, prepare_diarizer
from run_logging import log, log_load, stage

LANGUAGE = 'ru'


def diarization_progress():
    """WhisperX maps segmentation to 0–50% and embeddings to 50–99%."""
    last_bucket = -1
    started = time.monotonic()

    def report(percent):
        nonlocal last_bucket
        bucket = int(percent // 5)
        if bucket <= last_bucket:
            return
        last_bucket = bucket
        step = 'segmentation' if percent < 50 else 'embeddings/clustering'
        if percent >= 100:
            step = 'complete'
        log(f'PROGRESS diarization={percent:.0f}% step={step} '
            f'elapsed={time.monotonic() - started:.1f}s')

    return report


def transcribe_audio(audio_path, *, token, min_speakers=None, max_speakers=None):
    device, compute_type, model_name, batch_size = runtime_settings()
    log(f'CONFIG device={device} model={model_name} compute={compute_type} '
        f'batch={batch_size} language={LANGUAGE}')
    torch, whisperx, DiarizationPipeline = load_runtime()

    def release_memory():
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

    with stage('decode audio'):
        audio = whisperx.load_audio(str(audio_path))
    model = None
    try:
        with stage('load transcription model'):
            model = whisperx.load_model(
                model_name,
                device=device,
                compute_type=compute_type,
                language=LANGUAGE,
                task='transcribe',
                use_auth_token=token,
            )
        with stage('transcribe'):
            result = model.transcribe(audio, batch_size=batch_size)
    finally:
        del model
        release_memory()

    # Preserve decoding output before alignment and speaker assignment mutate it.
    extras = {'asr_segments': deepcopy(result['segments']),
              'run_info': {'model': model_name, 'compute_type': compute_type, 'device': device,
                           'batch_size': batch_size, 'language': LANGUAGE}}
    # Silence needs neither alignment nor speaker detection.
    if not result['segments']:
        return {**result, 'language': LANGUAGE, 'word_segments': [], **extras}

    align_model = None
    try:
        with stage('load alignment model'):
            align_model, metadata = whisperx.load_align_model(
                language_code=LANGUAGE, device=device,
            )
        with stage('align words'):
            result = whisperx.align(
                result['segments'], align_model, metadata, audio, device,
                return_char_alignments=False,
            )
    finally:
        del align_model
        release_memory()

    diarizer = None
    try:
        with stage('load diarization model'):
            diarizer = DiarizationPipeline(token=token, device=device)
            prepare_diarizer(diarizer)
        with stage('diarize'):
            progress = diarization_progress()
            progress(0)
            diarization = diarizer(
                audio, min_speakers=min_speakers, max_speakers=max_speakers,
                progress_callback=progress,
            )
        result = whisperx.assign_word_speakers(diarization, result)
    finally:
        del diarizer
        release_memory()

    result['language'] = LANGUAGE
    result.update(extras)
    return result


def run_job():
    job = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    result = transcribe_audio(
        job['audio_path'],
        token=os.environ['HF_TOKEN'].strip(),
        min_speakers=job.get('min_speakers'),
        max_speakers=job.get('max_speakers'),
    )
    Path(job['output_path']).write_text(
        json.dumps(result, ensure_ascii=False), encoding='utf-8',
    )


def main():
    started = time.monotonic()
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            log_load()
            stop.wait(15)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    try:
        with stage('job'):
            run_job()
        log(f'SUMMARY total={time.monotonic() - started:.1f}s')
    finally:
        stop.set()
        monitor_thread.join(timeout=4)


if __name__ == '__main__':
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
