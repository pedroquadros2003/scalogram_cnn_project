import hashlib
import json


import logging
logger = logging.getLogger(__name__)


def make_hash_id(data: dict, prefix="img", size=10):
    s = json.dumps(data, sort_keys=True)
    h = hashlib.md5(s.encode()).hexdigest()[:size]
    return f"{prefix}_{h}"