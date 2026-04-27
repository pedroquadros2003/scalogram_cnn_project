############################################################################
## Tensorflow and Supporting Modules
############################################################################

import tensorflow as tf 
import numpy as np
import itertools
import json
import gc
import scalogram_cnn_project.settings.config as config
import yaml

from scalogram_cnn_project.utils.dict_product import dict_product
from scalogram_cnn_project.utils.simplify_config_space import simplify_config_space

############################################################################
## Configure GPU memory growth BEFORE anything touches the GPU
############################################################################

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

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

############################################################################
## Logging module
############################################################################

import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("scalogram_cnn_project").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


############################################################################
## FILE PARAMETERS
############################################################################

INPUT_FOLDER  = "generated_scalograms_ALL_gray_overlap0.733_extra_input_example"
OUTPUT_FOLDER = "useless"
PARAMS_FILE   =  config.PARAM_SEARCH_DIR / "gridsearch_example.yaml"


PROGRESS_FILE = config.OUTPUT_DIR / OUTPUT_FOLDER / "progress.json"
PARAM_REGISTRY_FILE = config.OUTPUT_DIR / OUTPUT_FOLDER / "param_registry.json"

############################################################################
## GRID PARAMETERS
############################################################################


MODEL =  "v1"  #  "v2" #
MODEL_RUNNER = "v1"


with open(PARAMS_FILE) as f:
    config_params = yaml.safe_load(f)

MODEL_HYPER_PARAMS = simplify_config_space(config_params["MODEL_HYPER_PARAMS"])
MODEL_TRAIN_PARAMS = simplify_config_space(config_params["MODEL_TRAIN_PARAMS"])


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


    with open(PARAM_REGISTRY_FILE, "w") as f:
        
        param_registry["MODEL"] = MODEL
        param_registry["MODEL_RUNNER"] = MODEL_RUNNER
        param_registry["INPUT_FOLDER"] = INPUT_FOLDER

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

            acc, _ = run_model(
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
