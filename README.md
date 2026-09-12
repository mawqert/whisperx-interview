# WhisperX interview transcription

Transcribe Russian audio or video with `large-v3`, timestamps, and speaker labels. Copy the transcript or download a text file named after your upload. Consecutive blocks from the same speaker are merged.

## Setup

Requires Python, an NVIDIA GPU with working CUDA support, and a Hugging Face account with access to [speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1). Accept the model’s access conditions before starting.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Set `HF_TOKEN` in `.env` to your Hugging Face read token, then run:

```bash
./start.sh
```

Open http://127.0.0.1:8000. Models download on first use.

## Use

1. Choose an audio or video file (maximum 2 GB).
2. Leave speaker counts blank for automatic detection. If you know the exact count, set both minimum and maximum to that number.
3. Click **Transcribe** and keep the page open until it finishes.
4. Copy or download the transcript.

One transcription runs at a time. The terminal shows progress, GPU usage, and errors. Review technical terms and speaker labels before relying on the transcript.

## Connection problems

After a successful transcription, add `HF_HUB_OFFLINE=1` to `.env` to use cached models without contacting Hugging Face. Remove it if required models need downloading.

Restart the app after changing `.env`.
