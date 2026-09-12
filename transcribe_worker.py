"""Run WhisperX's Python pipeline in an isolated, time-limited worker."""

import gc
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import threading
import time

from model_config import MODEL_NAME, COMPUTE_TYPE, BATCH_SIZE, DIARIZATION_BATCH_SIZE
from runtime_setup import load_runtime, prepare_diarizer
from run_logging import log, log_load, stage

DEVICE = 'cuda'
LANGUAGE = 'ru'


def transcribe_audio(audio_path, *, token, min_speakers=None, max_speakers=None):
    log(f'CONFIG model={MODEL_NAME} compute={COMPUTE_TYPE} '
        f'batch={BATCH_SIZE} language={LANGUAGE}')
    torch, whisperx, DiarizationPipeline = load_runtime()

    def release_memory():
        gc.collect()
        torch.cuda.empty_cache()

    with stage('decode audio'):
        audio = whisperx.load_audio(str(audio_path))
    model = None
    try:
        with stage('load transcription model'):
            model = whisperx.load_model(
                MODEL_NAME,
                device=DEVICE,
                compute_type=COMPUTE_TYPE,
                language=LANGUAGE,
                task='transcribe',
                use_auth_token=token,
            )
        with stage('transcribe'):
            result = model.transcribe(audio, batch_size=BATCH_SIZE)
    finally:
        del model
        release_memory()

    # Preserve decoding output before alignment and speaker assignment mutate it.
    extras = {'asr_segments': deepcopy(result['segments']),
              'run_info': {'model': MODEL_NAME, 'compute_type': COMPUTE_TYPE,
                           'batch_size': BATCH_SIZE, 'language': LANGUAGE}}
    # Silence needs neither alignment nor speaker detection.
    if not result['segments']:
        return {**result, 'language': LANGUAGE, 'word_segments': [], **extras}

    align_model = None
    try:
        with stage('load alignment model'):
            align_model, metadata = whisperx.load_align_model(
                language_code=LANGUAGE, device=DEVICE,
            )
        with stage('align words'):
            result = whisperx.align(
                result['segments'], align_model, metadata, audio, DEVICE,
                return_char_alignments=False,
            )
    finally:
        del align_model
        release_memory()

    diarizer = None
    try:
        with stage('load diarization model'):
            diarizer = DiarizationPipeline(token=token, device=DEVICE)
            prepare_diarizer(diarizer, DIARIZATION_BATCH_SIZE)
        with stage('diarize'):
            diarization = diarizer(
                audio, min_speakers=min_speakers, max_speakers=max_speakers,
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
