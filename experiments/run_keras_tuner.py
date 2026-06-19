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
import argparse

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
## PARSER CONFIGURATION
############################################################################

parser = argparse.ArgumentParser(description="Run Keras Tuner Experiment")
parser.add_argument("--input_folder", type=str, default="generated_scalograms_ALL_gray_overlap0.733_extra_input_example", help="Input folder name")
parser.add_argument("--output_folder", type=str, default="generic_keras_search_example", help="Output folder name inside OUTPUT_DIR")
parser.add_argument("--params_file", type=str, default="parameter_searches/keras_search_example.yaml", help="YAML parameters file name")
parser.add_argument("--model", type=str, default="v2", choices=["v0", "v1", "v2"], help="Model version to use")
parser.add_argument("--model_runner", type=str, default="v1", choices=["v0", "v1", "v2"], help="Model runner version to use")
parser.add_argument("--max_trials", type=int, default=5, help="max_trials is the number of different models tested")

args = parser.parse_args()

############################################################################
## SEARCH PARAMETERS
############################################################################

INPUT_FOLDER  = args.input_folder
OUTPUT_FOLDER = args.output_folder
PARAMS_FILE   = args.params_file
MODEL         = args.model
MODEL_RUNNER  = args.model_runner
MAX_TRIALS    = args.max_trials

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
        input_folder=INPUT_FOLDER,
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
                f"Rank {i} | Trial ID: {trial.trial_id} | val_loss={trial.score:.5f} | "
                f"params={trial.hyperparameters.values}"
            )

            logger.info(line)

            f.write(line + "\n")