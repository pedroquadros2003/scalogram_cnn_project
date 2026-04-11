
import logging
logger = logging.getLogger(__name__)


def simplify_config_space(config_space):
    """
    Convert a YAML-style hyperparameter specification into a simple
    grid-search parameter dictionary.

    Parameters
    ----------
    config_space : dict
        Dictionary where each parameter has the structure:
        {
            "param_name": {
                "mode": "...",
                "values": [...]
            }
        }

    Returns
    -------
    dict
        Simplified dictionary where each parameter maps to a list of values.
    """

    simple = {}

    for name, spec in config_space.items():

        mode = spec["mode"]
        values = spec["values"]

        # fixed parameters
        if mode == "fixed":
            simple[name] = values

        # categorical choice
        elif mode == "choice":
            simple[name] = values
            

        else:
            raise ValueError(f"Unknown mode: {mode}")

    return simple