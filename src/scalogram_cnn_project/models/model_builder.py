import logging
logger = logging.getLogger(__name__)


def build_model(
    hp,
    create_model,
    MODEL_HYPER_PARAMS,
    MODEL_TRAIN_PARAMS
):

    params = {}

    ####################################################################
    # Resolve parameter according to search mode
    ####################################################################

    def resolve_param(name, spec):

        mode = spec["mode"]
        values = spec["values"]

        if mode == "fixed":

            return values[0]

        elif mode == "choice":

            return hp.Choice(name, values)

        elif mode == "float_interval":

            return hp.Float(
                name,
                min_value=values[0],
                max_value=values[1],
            )

        elif mode == "log_interval":

            return hp.Float(
                name,
                min_value=values[0],
                max_value=values[1],
                sampling="log",
            )

        elif mode == "int_interval":

            return hp.Int(
                name,
                min_value=values[0],
                max_value=values[1],
            )

        else:

            raise ValueError(f"Unknown mode '{mode}' for parameter '{name}'")

    ####################################################################
    # MODEL HYPERPARAMETERS
    ####################################################################

    for name, spec in MODEL_HYPER_PARAMS.items():

        params[name] = resolve_param(name, spec)

    ####################################################################
    # TRAINING PARAMETERS
    ####################################################################

    for name, spec in MODEL_TRAIN_PARAMS.items():

        params[name] = resolve_param(name, spec)


    ####################################################################
    # BUILD MODEL
    ####################################################################

    model, _ = create_model(params)

    return model