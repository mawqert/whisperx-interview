"""Small stdout logger and best-effort NVIDIA load snapshots."""

from contextlib import contextmanager
from datetime import datetime
import os
import subprocess
import time


def log(message):
    token = os.environ.get('HF_TOKEN', '').strip()
    if token:
        message = message.replace(token, '[redacted]')
    print(f'{datetime.now():%H:%M:%S} {message}', flush=True)


def log_load():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            for row in result.stdout.strip().splitlines():
                index, utilization, used, total = [value.strip() for value in row.split(',')]
                log(f'LOAD gpu={index} utilization={utilization}% VRAM={used}/{total} MiB')
        else:
            log('LOAD GPU statistics unavailable')
    except (OSError, subprocess.TimeoutExpired, ValueError):
        log('LOAD GPU statistics unavailable')


@contextmanager
def stage(name):
    started = time.monotonic()
    log(f'START {name}')
    try:
        yield
    except Exception as exc:
        log(f'ERROR {name} elapsed={time.monotonic() - started:.1f}s {type(exc).__name__}: {exc}')
        raise
    else:
        log(f'DONE {name} elapsed={time.monotonic() - started:.1f}s')
