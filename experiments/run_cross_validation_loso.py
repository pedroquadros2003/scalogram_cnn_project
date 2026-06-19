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
import argparse

from scalogram_cnn_project.utils.dict_product import dict_product
from scalogram_cnn_project.utils.simplify_config_space import simplify_config_space


############################################################################
## Model Creators and Model Runners
############################################################################

from scalogram_cnn_project.models import model_v0, model_v1, model_v2
MODEL_CREATORS = {
    "v0": model_v0.create_model,
    "v1": model_v1.create_model,
    "v2": model_v2.create_model,
}

from scalogram_cnn_project.model_runners import model_runner_v2
MODEL_RUNNERS = {
    "v2": model_runner_v2.run_model,
}

############################################################################
## Logging
############################################################################

import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("scalogram_cnn_project").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

############################################################################
## PARSER CONFIGURATION
############################################################################

parser = argparse.ArgumentParser(description="Run Leave-One-Subject-Out Cross Validation Experiment")
parser.add_argument("--input_folder", type=str, default="generated_scalograms_ALL_gray_overlap0.733_extra_input_example", help="Input folder name")
parser.add_argument("--output_folder", type=str, default="generic_loso_example", help="Output folder name inside OUTPUT_DIR")
parser.add_argument("--params_file", type=str, default="parameter_searches/cross_validation_loso_example.yaml", help="YAML parameters file name")
parser.add_argument("--model", type=str, default="v0", choices=["v0", "v1", "v2"], help="Model version to use")

args = parser.parse_args()

############################################################################
## GRID PARAMETERS
############################################################################

INPUT_FOLDER  = args.input_folder
OUTPUT_FOLDER = args.output_folder
PARAMS_FILE   = args.params_file
MODEL         = args.model
MODEL_RUNNER  = "v2"


with open(PARAMS_FILE) as f:
    config_params = yaml.safe_load(f)

MODEL_HYPER_PARAMS = simplify_config_space(config_params["MODEL_HYPER_PARAMS"])
MODEL_TRAIN_PARAMS = simplify_config_space(config_params["MODEL_TRAIN_PARAMS"])



############################################################################
## MAIN
############################################################################

if __name__ == "__main__":

    # ====================================
    # FIXED FILENAMES
    # ====================================

    PROGRESS_FILE       = config.OUTPUT_DIR / OUTPUT_FOLDER / "progress.json"
    PARAM_REGISTRY_FILE = config.OUTPUT_DIR / OUTPUT_FOLDER / "param_registry.json"


    run_model = MODEL_RUNNERS[MODEL_RUNNER]
    create_model = MODEL_CREATORS[MODEL]

    results = {}

    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)

    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r") as f:
            results = json.load(f)

        logger.info("Resuming experiment. %d configs already done.", len(results))

    # ====================================
    # CREATE PARAMETER GRID
    # ====================================

    train_configs = list(dict_product(MODEL_TRAIN_PARAMS))
    model_configs = list(dict_product(MODEL_HYPER_PARAMS))

    grid_params = []
    param_registry = {}

    model_id_counter = 0

    for model_hp, train_hp in itertools.product(
        model_configs,
        train_configs
    ):

        params = {}

        params.update(model_hp)
        params.update(train_hp)

        # Generate model ID
        model_id = f"model_{model_id_counter:05d}"
        model_id_counter += 1

        params["model_id"] = model_id

        # Save parameters to registry
        serializable_params = {
            **model_hp,
            **train_hp
        }

        param_registry[model_id] = serializable_params

        grid_params.append(params)

    # ====================================
    # SAVE PARAM REGISTRY
    # ====================================

    with open(PARAM_REGISTRY_FILE, "w") as f:

        param_registry["MODEL"] = MODEL
        param_registry["MODEL_RUNNER"] = MODEL_RUNNER
        param_registry["INPUT_FOLDER"] = INPUT_FOLDER

        json.dump(param_registry, f, indent=2)

    # ====================================
    # GRID SEARCH LOOP
    # ====================================

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
                input_folder=INPUT_FOLDER,
                output_folder=config.OUTPUT_DIR / OUTPUT_FOLDER
            )

        except Exception as e:

            logger.error("Error in %s: %s", model_id, e)
            acc = None

        # Save progress
        results[model_id] = acc

        with open(PROGRESS_FILE, "w") as f:
            json.dump(results, f, indent=2)

        # Clean memory
        tf.keras.backend.clear_session()
        gc.collect()

    # ====================================
    # FINAL STATS
    # ====================================

    valid_results = [v for v in results.values() if v is not None]
    mean = sum(valid_results) / len(valid_results) if valid_results else None

    logger.info("Final results: %s", results)
    logger.info("Mean accuracy: %s", mean)