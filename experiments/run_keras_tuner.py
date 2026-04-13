############################################################################
## Tensorflow and Supporting Modules
############################################################################

import tensorflow as tf 
import numpy as np
import json
import scalogram_cnn_project.settings.config as config
import keras_tuner as kt
import yaml
import shutil

from scalogram_cnn_project.models.model_builder import build_model
from scalogram_cnn_project.utils.custom_tuner import CustomTuner

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

INPUT_FOLDER = "generated_scalograms_ALL_gray_overlap0.733_extra_input_example"
OUTPUT_FOLDER = "keras_search"

PARAMS_FILE = config.PARAM_SEARCH_DIR / "keras_search_example.yaml"

############################################################################
## SEARCH PARAMETERS
############################################################################

MODEL =  "v2"  
MODEL_RUNNER = "v1"

## MAX_TRIALS is the number of different models tested
MAX_TRIALS = 200

with open(PARAMS_FILE) as f:
    config_params = yaml.safe_load(f)

MODEL_HYPER_PARAMS = config_params["MODEL_HYPER_PARAMS"]
MODEL_TRAIN_PARAMS = config_params["MODEL_TRAIN_PARAMS"]


############################################################################
## PREPARE OUTPUT FOLDER
############################################################################

output_dir = config.OUTPUT_DIR / OUTPUT_FOLDER
output_dir.mkdir(parents=True, exist_ok=True)

# Copy YAML config for reproducibility
shutil.copy(PARAMS_FILE, config.OUTPUT_DIR / OUTPUT_FOLDER / "search_params.yaml")


############################################################################
## MAIN FUNCTION
############################################################################


if __name__ == "__main__":

    # =====================================
    # LOAD MODEL CREATOR AND MODEL RUNNER
    # =====================================

    run_model = MODEL_RUNNERS[MODEL_RUNNER]
    create_model = MODEL_CREATORS[MODEL]

    # =====================================
    # CREATE TUNER
    # =====================================

    tuner_hypermodel = lambda hp: build_model(hp, create_model, MODEL_HYPER_PARAMS, MODEL_TRAIN_PARAMS)

    tuner = CustomTuner(
        input_folder=config.DATA_DIR / INPUT_FOLDER,
        output_folder=config.OUTPUT_DIR / OUTPUT_FOLDER,
        model_hyper_params=MODEL_HYPER_PARAMS,
        model_train_params=MODEL_TRAIN_PARAMS,
        create_model=create_model,
        run_model=run_model,
        hypermodel = tuner_hypermodel,
        objective=kt.Objective("val_loss", direction="min"),
        max_trials=MAX_TRIALS,
        directory=config.OUTPUT_DIR / OUTPUT_FOLDER,
        project_name="keras_tuner_optimization",
        overwrite=False,   # allows resume
    )

    logger.info("Search space summary:")

    tuner.search_space_summary()

    # =====================================
    # RUN SEARCH
    # =====================================

    logger.info("Starting Keras Tuner search")

    logger.info("Existing trials: %d", len(tuner.oracle.trials))

    tuner.search()

    logger.info("Search finished")

    # =====================================
    # SHOW BEST RESULTS (TOP10)
    # =====================================

    best_trials = tuner.oracle.get_best_trials(num_trials=10)

    with open(config.OUTPUT_DIR / OUTPUT_FOLDER / "best_trials.txt", "w") as f:

        for i, trial in enumerate(best_trials):

            line = (
                f"Rank {i} | val_loss={trial.score:.5f} | "
                f"params={trial.hyperparameters.values}"
            )

            logger.info(line)

            f.write(line + "\n")