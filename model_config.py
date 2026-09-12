"""Supported CTranslate2 inference profiles for a 6 GB GPU."""

DEFAULT_MODEL = 'large-v3-turbo'
MODEL_PROFILES = {
    'large-v3-turbo': {'label': 'Standard turbo (default)', 'compute_type': 'int8', 'batch_size': 8},
    'large-v3': {'label': 'Full large-v3 (slower comparison)', 'compute_type': 'int8_float16', 'batch_size': 2},
    'codeswitch': {'label': 'Russian code-switch', 'compute_type': 'int8', 'batch_size': 8},
}
