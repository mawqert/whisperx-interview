"""Runtime setup for WhisperX 3.8.6 and its pyannote backend."""

import warnings

from run_logging import log


def load_runtime():
    import torch

    # This pipeline decodes with ffmpeg and passes waveforms to pyannote.
    # Replace only its unused-decoder import warning with a concise explanation.
    with warnings.catch_warnings(record=True) as caught:
        import whisperx
        from whisperx.diarize import DiarizationPipeline
    for warning in caught:
        if str(warning.message).lstrip().startswith('torchcodec is not installed correctly'):
            log('INFO TorchCodec unavailable; using ffmpeg-decoded waveforms for pyannote')
        else:
            warnings.warn_explicit(warning.message, warning.category,
                                   warning.filename, warning.lineno)
    return torch, whisperx, DiarizationPipeline


def prepare_diarizer(diarizer):
    pipeline = diarizer.model
    # Use the model's own batch settings and lazy initialization, matching the
    # successful run. Do not run synthetic embedding probes ahead of inference.
    log(f'CONFIG diarization segmentation_batch={pipeline.segmentation_batch_size} '
        f'embedding_batch={pipeline.embedding_batch_size}')
