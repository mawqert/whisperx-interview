"""Inference settings for local and container runtimes."""

import os

COMPUTE_TYPE = 'int8_float16'


def runtime_settings():
    device = os.environ.get('WHISPERX_DEVICE', 'cuda').strip().lower()
    if device not in ('cpu', 'cuda'):
        raise ValueError('WHISPERX_DEVICE must be cpu or cuda')
    default_compute = 'int8' if device == 'cpu' else COMPUTE_TYPE
    compute_type = os.environ.get('WHISPERX_COMPUTE_TYPE', default_compute).strip()
    if not compute_type:
        raise ValueError('WHISPERX_COMPUTE_TYPE must not be empty')
    model_var = 'WHISPERX_CPU_MODEL' if device == 'cpu' else 'WHISPERX_GPU_MODEL'
    model_name = os.getenv(model_var, 'large-v3').strip()
    if not model_name:
        raise ValueError(f'{model_var} must not be empty')
    batch_size = 4
    if device == 'cpu':
        raw_batch = os.getenv('WHISPERX_CPU_BATCH_SIZE', str(batch_size)).strip()
        if not raw_batch.isascii() or not raw_batch.isdecimal() or int(raw_batch) < 1:
            raise ValueError('WHISPERX_CPU_BATCH_SIZE must be a positive integer')
        batch_size = int(raw_batch)
    return device, compute_type, model_name, batch_size
