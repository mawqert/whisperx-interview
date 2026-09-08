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

Uses `large-v3-turbo`, CUDA float16, batch size 8. Restart after editing `.env`; existing environment variables take precedence.

Hugging Face connection failures retry using cached models. After the first successful run, set `HF_HUB_OFFLINE=1` in `.env` to skip online checks entirely.
