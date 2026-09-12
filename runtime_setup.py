"""Runtime setup for WhisperX 3.8.6 and its pyannote backend."""

import warnings

from run_logging import log


def load_runtime():
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
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


def prepare_diarizer(diarizer, batch_size):
    pipeline = diarizer.model
    pipeline.segmentation_batch_size = batch_size
    pipeline.embedding_batch_size = batch_size
    # pyannote probes progressively shorter random inputs to determine the
    # embedding model's minimum length. A one-frame probe has undefined sample
    # std; this is not an embedding computed from the uploaded recording.
    # Prime the cached property here, limiting suppression to this probe only.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', message=r'std\(\): degrees of freedom is <= 0\.',
            category=UserWarning, module=r'pyannote\.audio\.models\.blocks\.pooling',
        )
        minimum = pipeline._embedding.min_num_samples
    log(f'CONFIG diarization segmentation_batch={batch_size} embedding_batch={batch_size} '
        f'minimum_embedding_samples={minimum}')
