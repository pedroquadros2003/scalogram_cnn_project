import logging
logger = logging.getLogger(__name__)

def dict_to_str(d):
    parts = []
    for k, v in d.items():
        if isinstance(v, tuple):
            v = "x".join(map(str, v))  # (3,3) → 3x3
        parts.append(f"{k}{v}")
    return "_".join(parts)