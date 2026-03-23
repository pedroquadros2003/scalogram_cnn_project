import logging
logger = logging.getLogger(__name__)


def validate_dict_params(params, required_keys):
    """
    Example of use:\n
    validate_params(params, ["learning_rate", "batch_size", "optimizer"])
    """

    missing = []
    none_values = []

    for k in required_keys:
        if k not in params:
            missing.append(k)
        elif params[k] is None:
            none_values.append(k)

    if missing:
        raise ValueError(f"Missing keys: {missing}")
    
    if none_values:
        raise ValueError(f"None values: {none_values}")
