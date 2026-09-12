"""Run WhisperX's Python pipeline in an isolated, time-limited worker."""

import gc
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import threading
import time

from huggingface_hub import snapshot_download
from model_config import DEFAULT_MODEL, MODEL_PROFILES
from run_logging import log, log_load, stage

MODEL_REPO = 'coriollon/whisper-large-v3-turbo-russian-codeswitch'
MODEL_SUBDIR = 'ct2_int8_float16'
DEVICE = 'cuda'
LANGUAGE = 'ru'


def transcribe_audio(audio_path, *, token, min_speakers=None, max_speakers=None,
                     model_name=DEFAULT_MODEL):
    if model_name not in MODEL_PROFILES:
        raise ValueError('Unsupported transcription model')
    profile = MODEL_PROFILES[model_name]
    log(f'CONFIG model={model_name} compute={profile["compute_type"]} '
        f'batch={profile["batch_size"]} language={LANGUAGE}')
    import torch
    import whisperx
    from whisperx.diarize import DiarizationPipeline

    def release_memory():
        gc.collect()
        torch.cuda.empty_cache()

    # Use the Hub cache so subsequent runs and HF_HUB_OFFLINE reuse these files.
    model_path = model_name
    if model_name == 'codeswitch':
        with stage('download code-switch model'):
            snapshot = snapshot_download(
                repo_id=MODEL_REPO,
                allow_patterns=f'{MODEL_SUBDIR}/*',
            )
        model_path = str(Path(snapshot) / MODEL_SUBDIR)
    with stage('decode audio'):
        audio = whisperx.load_audio(str(audio_path))
    model = None
    try:
        with stage('load transcription model'):
            model = whisperx.load_model(
                model_path,
                device=DEVICE,
                compute_type=profile['compute_type'],
                language=LANGUAGE,
                task='transcribe',
                use_auth_token=token,
            )
        with stage('transcribe'):
            result = model.transcribe(audio, batch_size=profile['batch_size'])
    finally:
        del model
        release_memory()

    # Preserve decoding output before alignment and speaker assignment mutate it.
    extras = {'asr_segments': deepcopy(result['segments']),
              'run_info': {'model': model_name, 'compute_type': profile['compute_type'],
                           'batch_size': profile['batch_size'], 'language': LANGUAGE}}
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
        model_name=job.get('model_name', DEFAULT_MODEL),
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
