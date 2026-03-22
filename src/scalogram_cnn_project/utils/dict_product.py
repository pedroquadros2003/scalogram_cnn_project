import itertools

import logging
logger = logging.getLogger(__name__)

def dict_product(d):
    if not d:
        yield {}
        return

    keys = d.keys()
    for values in itertools.product(*d.values()):
        yield dict(zip(keys, values))