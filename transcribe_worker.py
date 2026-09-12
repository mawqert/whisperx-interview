"""Run WhisperX's Python pipeline in an isolated, time-limited worker."""

import gc
import json
import os
from pathlib import Path
import sys

from huggingface_hub import snapshot_download

MODEL_REPO = 'coriollon/whisper-large-v3-turbo-russian-codeswitch'
MODEL_SUBDIR = 'ct2_int8_float16'
DEVICE = 'cuda'
LANGUAGE = 'ru'
BATCH_SIZE = 8


def transcribe_audio(audio_path, *, token, min_speakers=None, max_speakers=None):
    import torch
    import whisperx
    from whisperx.diarize import DiarizationPipeline

    def release_memory():
        gc.collect()
        torch.cuda.empty_cache()

    # Use the Hub cache so subsequent runs and HF_HUB_OFFLINE reuse these files.
    snapshot = snapshot_download(
        repo_id=MODEL_REPO,
        allow_patterns=f'{MODEL_SUBDIR}/*',
    )
    audio = whisperx.load_audio(str(audio_path))
    model = None
    try:
        model = whisperx.load_model(
            str(Path(snapshot) / MODEL_SUBDIR),
            device=DEVICE,
            compute_type='int8',
            language=LANGUAGE,
            task='transcribe',
            use_auth_token=token,
        )
        result = model.transcribe(audio, batch_size=BATCH_SIZE)
    finally:
        del model
        release_memory()

    # Silence needs neither alignment nor speaker detection.
    if not result['segments']:
        return {**result, 'language': LANGUAGE, 'word_segments': []}

    align_model = None
    try:
        align_model, metadata = whisperx.load_align_model(
            language_code=LANGUAGE, device=DEVICE,
        )
        result = whisperx.align(
            result['segments'], align_model, metadata, audio, DEVICE,
            return_char_alignments=False,
        )
    finally:
        del align_model
        release_memory()

    diarizer = None
    try:
        diarizer = DiarizationPipeline(token=token, device=DEVICE)
        diarization = diarizer(
            audio, min_speakers=min_speakers, max_speakers=max_speakers,
        )
        result = whisperx.assign_word_speakers(diarization, result)
    finally:
        del diarizer
        release_memory()

    result['language'] = LANGUAGE
    return result


def main():
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


if __name__ == '__main__':
    main()
