# WhisperX interview transcribe

Upload audio/video, transcribe in Russian, and copy the transcript grouped by speaker. Optional minimum/maximum speaker counts.

Requires an NVIDIA GPU with CUDA and access to [pyannote speaker diarization](https://huggingface.co/pyannote/speaker-diarization-community-1). Models download on first use.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
# Set HF_TOKEN in .env, then start:
./start.sh
```

Open http://127.0.0.1:8000. For subsequent runs, use `./start.sh`.

Uses [coriollon/whisper-large-v3-turbo-russian-codeswitch](https://huggingface.co/coriollon/whisper-large-v3-turbo-russian-codeswitch) for Russian speech containing English IT terms.
The Python worker downloads only the `ct2_int8_float16` build and runs on CUDA with INT8, batch size **8**, and explicit Russian language.

Transcription, word alignment, and speaker diarization use the [WhisperX Python API](https://github.com/m-bain/whisperX#python-usage-). Models are released between stages to limit GPU memory use. Each job runs in a separate process to enforce the six-hour timeout and release GPU resources on exit. Audio is decoded once and reused across stages. English terms may have less reliable word timestamps with the Russian aligner.

Restart after editing `.env`; existing environment variables take precedence.

Hugging Face connection failures retry using cached models. After the first successful run, set `HF_HUB_OFFLINE=1` in `.env` to skip online checks entirely.
