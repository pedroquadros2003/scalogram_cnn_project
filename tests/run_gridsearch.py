############################################################################
## Tensorflow and Supporting Modules
############################################################################

import tensorflow as tf 
from tensorflow import keras
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

from scalogram_cnn_project.models import model_v0
MODEL_CREATORS = {
    "v0": model_v0.create_model,
}

from scalogram_cnn_project.model_runners import model_runner_v0, model_runner_v1, model_runner_v2
MODEL_RUNNERS = {
    "v0": model_runner_v0.run_model,
    "v1": model_runner_v1.run_model,
    "v2": model_runner_v2.run_model,
}

OPTIMIZERS = [
    ("adam", keras.optimizers.Adam),
    ("sgd" , keras.optimizers.SGD),
    ("rmsprop", keras.optimizers.RMSprop),
]

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

MODEL_RUNNER = "v2"
MODEL = "v0"
MODE = "separate"

OVERLAP = 0.85
CMAP = "gray"
CHANNELS = ["C3", "C4"]
LOSO_SUBJECT = 2 ## If it does not apply, the model_runner will simply not use it.


INPUT_FOLDER = f"generated_scalograms_C3C4_{CMAP}_overlap{OVERLAP}"
OUTPUT_FOLDER = "useless"
PROGRESS_FILE = config.OUTPUT_DIR / OUTPUT_FOLDER / "progress.json"

############################################################################
## GRID PARAMETERS
############################################################################

MODEL_HYPER_PARAMS = {
    # empty for the time being (but ready to be used)
}

MODEL_TRAIN_PARAMS = {
    "learning_rate": [1e-3, 1e-4, 1e-5],
    "batch_size": [16, 32, 64],
}

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
    # CREATING A GRID OF TRAINING PARAMS
    # ====================================

    fixed_params = {
        "loso_subject": LOSO_SUBJECT,
        "cmap": CMAP,
        "channels": CHANNELS,
        "seed": 42,
        "epsilon": 1e-3,
        "momentum": 0.99,
        "mode": MODE,
        "overlap": OVERLAP,
    }

    train_configs = list(dict_product(MODEL_TRAIN_PARAMS))
    model_configs = list(dict_product(MODEL_HYPER_PARAMS))

    grid_params = []

    for model_hp, train_hp, (opt_name, opt_class) in itertools.product(
        model_configs,
        train_configs,
        OPTIMIZERS
    ):
        params = fixed_params.copy()

        # Model hyperparameters
        params.update(model_hp)

        # Training parameters
        params.update(train_hp)

        # Optimizer
        params.update({
            "optimizer_name": opt_name,
            "optimizer": opt_class(learning_rate=train_hp["learning_rate"])
        })

        model_str = dict_to_str(model_hp)
        train_str = dict_to_str(train_hp)
        parts = [opt_name, train_str, model_str]
        params["model_name"] = "_".join(p for p in parts if p)

        grid_params.append(params)


    # ==============================
    # GRID SEARCH LOOP
    # ==============================


    for params in grid_params:
        
        model_name = params["model_name"]

        if model_name in results:
            logger.info("Skipping %s (already completed)", model_name)
            continue

        logger.info("Running %s", model_name)

        try:
            model, callback = create_model(params)

            acc = run_model(
                model=model, 
                callback=callback,
                training_parameters=params,
                input_folder=config.DATA_DIR / INPUT_FOLDER,
                output_folder=config.OUTPUT_DIR / OUTPUT_FOLDER
            )
        except Exception as e:
            logger.error("Error in %s: %s", model_name, e)
            acc = None


        # SAVE PROGRESS IMMEDIATELY
        results[model_name] = acc

        with open(PROGRESS_FILE, "w") as f:
            json.dump(results, f, indent=2)

        # CLEAN MEMORY
        tf.keras.backend.clear_session()
        gc.collect()



    logger.info("Final results: %s", results)
