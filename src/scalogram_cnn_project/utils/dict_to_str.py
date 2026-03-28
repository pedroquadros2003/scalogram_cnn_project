import logging
logger = logging.getLogger(__name__)

ABBREV = {
    "learning_rate": "lr",
    "batch_size": "bs",
    "kernel_size": "k",
    "loso_subject": "loso",
}

def dict_to_str(d):
    parts = []
    
    for k in sorted(d.keys()):
        v = d[k]
        key = ABBREV.get(k, k)  # use abbreviation if it exists

        if isinstance(v,  (list, tuple)):
            v = "x".join(map(str, v))

        parts.append(f"{key}_{v}")
    
    return "_".join(parts)