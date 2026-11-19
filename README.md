# WhisperX interview transcription

Transcribe Russian audio or video with `large-v3`, timestamps, and speaker labels. Copy the transcript or download a text file named after your upload. Consecutive blocks from the same speaker are merged.

## Docker Compose

Use Docker with Linux containers and Compose 2.30 or newer. Both variants need a Hugging Face read token with access to [speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1). Accept the model’s access conditions, copy `.env.example` to `.env`, and set `HF_TOKEN` there.

Build and start the CPU service:

```bash
docker compose up --build cpu
```

CPU defaults to `large-v3` with a batch size of 4. For a lighter CPU configuration, add these settings to `.env`:

```dotenv
WHISPERX_CPU_MODEL=small
WHISPERX_CPU_BATCH_SIZE=1
```

Apply changes with `docker compose up -d cpu`. The same CPU image supports both configurations. The first transcription with a new model downloads it to the shared cache.

For an NVIDIA GPU, enable GPU access in Docker and use a driver compatible with CUDA 12.8. To use the GPU service:

```bash
docker compose up --build gpu
```

The GPU service defaults to `large-v3`. To select `small`, set `WHISPERX_GPU_MODEL=small` in `.env` and run `docker compose up -d gpu`.

Open http://127.0.0.1:8000 for either service. Before switching, stop the running service with `docker compose stop cpu` or `docker compose stop gpu`. Both services use the same named volume for downloaded models.

### Published images

GitHub Actions builds `linux/amd64` images on pull requests targeting `master`. A push to `master` or a manual run of **Build and publish Docker images** from `master` publishes `ghcr.io/mawqert/whisperx:cpu` and `ghcr.io/mawqert/whisperx:gpu` using the repository’s `GITHUB_TOKEN`.

After publication, start a service from GHCR with:

```bash
docker compose up --pull always --no-build cpu
```

Use `gpu` in the same command to start the GPU service. For a private package, log in to `ghcr.io` with a token that has package read access first.

## Local setup

The local launcher uses CUDA by default and requires Python and the same Hugging Face model access. For CPU inference, add `WHISPERX_DEVICE=cpu` to `.env`.

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

Each service runs one transcription at a time. The terminal shows progress, errors, and GPU usage when available. Review technical terms and speaker labels before relying on the transcript.

## Connection problems

After a successful transcription, add `HF_HUB_OFFLINE=1` to `.env` to use cached models without contacting Hugging Face. Remove it if required models need downloading.

Restart the app after changing `.env`.
