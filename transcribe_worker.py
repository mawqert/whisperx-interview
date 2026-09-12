"""Resolve the quantized code-switch model before running the WhisperX CLI."""

import os
from pathlib import Path
import sys

from huggingface_hub import snapshot_download

MODEL_REPO = 'coriollon/whisper-large-v3-turbo-russian-codeswitch'
MODEL_SUBDIR = 'ct2_int8_float16'


def main():
    # Use the Hub cache so subsequent runs and HF_HUB_OFFLINE reuse these files.
    snapshot = snapshot_download(
        repo_id=MODEL_REPO,
        allow_patterns=f'{MODEL_SUBDIR}/*',
    )
    from whisperx.__main__ import cli

    # Keep the token out of the operating system's process arguments.
    sys.argv.extend([
        '--model', str(Path(snapshot) / MODEL_SUBDIR),
        '--hf_token', os.environ['HF_TOKEN'],
    ])
    cli()


if __name__ == '__main__':
    main()
