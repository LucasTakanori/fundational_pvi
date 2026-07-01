"""Architecture tag normalization for foundation entry points."""

_ARCH_ALIASES = {
    "core_s": "crt",
    "cores": "crt",
    "core_u": "mae",
    "coreu": "mae",
    "transformer": "crt",
    "cnn_transformer": "crt",
    "masked_transformer": "mae",
}


def normalize_arch(tag: str) -> str:
    key = str(tag).lower().strip()
    return _ARCH_ALIASES.get(key, key)
