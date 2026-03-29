############################################################################
## Tensorflow and Supporting Modules
############################################################################

import tensorflow as tf 
from keras.optimizers import Adam, SGD, RMSprop
import numpy as np
import itertools
import json
import gc
import scalogram_cnn_project.settings.config as config
from scalogram_cnn_project.utils.dict_product import dict_product
from scalogram_cnn_project.utils.dict_to_str import dict_to_str

############################################################################
## Model Runners, Model Creators and Optimizers
############################################################################

from scalogram_cnn_project.models import model_v0, model_v1, model_v2
MODEL_CREATORS = {
    "v0": model_v0.create_model,
    "v1": model_v1.create_model,
    "v2": model_v2.create_model,
}

from scalogram_cnn_project.model_runners import model_runner_v0, model_runner_v1, model_runner_v2
MODEL_RUNNERS = {
    "v0": model_runner_v0.run_model,
    "v1": model_runner_v1.run_model,
    "v2": model_runner_v2.run_model,
}

OPTIMIZERS = {
    "adam"   : Adam,
    "sgd"    : SGD,
    "rmsprop": RMSprop,
}

############################################################################
## Logging module
############################################################################

import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("scalogram_cnn_project").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


############################################################################
## PARAMETERS
############################################################################

MODEL_RUNNER = "v1"
MODEL = "v2"

OVERLAP = 0.733
CMAP = "gray"
CHANNELS = ["C3", "C4"]
SUBJECTS = [1]
LOSO_SUBJECT = 2 ## If it does not apply, the model_runner will simply not use it.


INPUT_FOLDER = f"generated_scalograms_ALL_{CMAP}_overlap{OVERLAP}_subject1"
OUTPUT_FOLDER = "useless"
PROGRESS_FILE = config.OUTPUT_DIR / OUTPUT_FOLDER / "progress.json"
PARAM_REGISTRY_FILE = config.OUTPUT_DIR / OUTPUT_FOLDER / "param_registry.json"

############################################################################
## GRID PARAMETERS
############################################################################

MODEL_HYPER_PARAMS = {
    "epsilon" : [1e-3],
    "momentum": [0.99],
    "cmap": [CMAP],
    "channels": [CHANNELS],
    "mode": ["separate"],
}

MODEL_TRAIN_PARAMS = {
    "learning_rate": [1e-3, 1e-4, 1e-5],
    "batch_size": [16, 32, 64],
    "seed": [42],
    "subjects": [SUBJECTS],
    "loso_subject": [LOSO_SUBJECT],
    "overlap": [OVERLAP],
    "optimizer_name": ["adam", "sgd", "rmsprop"],
}

MODEL_HYPER_PARAMS.update(
{
    "extra_layer" : [False],
    "extra_layer_num_filters": [16],
    "first_layer_num_filters": [64],
    "second_layer_num_filters": [32],
    "kernel_size": [2],
    "num_neurons_dense": [128],
    "n_additional_features": [3]
}
)

############################################################################
## Main Function
############################################################################


if __name__ == "__main__":

    # =====================================
    # LOAD MODEL CREATOR AND MODEL RUNNER
    # =====================================

    run_model = MODEL_RUNNERS[MODEL_RUNNER]
    create_model = MODEL_CREATORS[MODEL]

    # ==============================
    # LOAD PREVIOUS PROGRESS
    # ==============================

    results={}
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)

    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r") as f:
            results = json.load(f)
        logger.info("Resuming experiment. %s models already done.", len(results))
    else:
        results = {}


    # ====================================
    # CREATING A GRID OF PARAMS
    # ====================================


    train_configs = list(dict_product(MODEL_TRAIN_PARAMS))
    model_configs = list(dict_product(MODEL_HYPER_PARAMS))

    model_id_counter = 0
    param_registry = {}   # id -> params
    grid_params = []

    for model_hp, train_hp in itertools.product(
        model_configs,
        train_configs
    ):
        params = {}

        # Model hyperparameters
        params.update(model_hp)

        # Training parameters
        params.update(train_hp)

        # Optimizer
        opt_name = train_hp["optimizer_name"]
        opt_class = OPTIMIZERS[opt_name]

        params["optimizer"] = opt_class(learning_rate=train_hp["learning_rate"])


        # Create Model ID
        model_id = f"model_{model_id_counter:05d}"
        model_id_counter += 1
        params["model_id"] = model_id


        # Save params
        serializable_params = {
            **model_hp,
            **train_hp
        }
        param_registry[model_id] = serializable_params

        grid_params.append(params)


    # ==============================
    # SAVE PARAMS REGISTRY
    # ==============================

    param_registry["MODEL"] = MODEL
    param_registry["MODEL_RUNNER"] = MODEL_RUNNER

    with open(PARAM_REGISTRY_FILE, "w") as f:
        json.dump(param_registry, f, indent=2)


    # ==============================
    # GRID SEARCH LOOP
    # ==============================


    for params in grid_params:
        
        model_id = params["model_id"]

        if model_id in results:
            logger.info("Skipping %s (already completed)", model_id)
            continue

        logger.info("Running %s", model_id)

        try:
            model, callback = create_model(params)

            acc = run_model(
                model=model, 
                callback=callback,
                parameters=params,
                input_folder=config.DATA_DIR / INPUT_FOLDER,
                output_folder=config.OUTPUT_DIR / OUTPUT_FOLDER
            )
        except Exception as e:
            logger.error("Error in %s: %s", model_id, e)
            acc = None


        # SAVE PROGRESS IMMEDIATELY
        results[model_id] = acc

        with open(PROGRESS_FILE, "w") as f:
            json.dump(results, f, indent=2)

        # CLEAN MEMORY
        tf.keras.backend.clear_session()
        gc.collect()



    logger.info("Final results: %s", results)
