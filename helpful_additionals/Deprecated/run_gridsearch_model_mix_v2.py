import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import numpy as np
import itertools
import gc
import json
from scalogram_cnn_project.model_runners.model_runner_mix_v2 import run_model
import scalogram_cnn_project.settings.config as config
from scalogram_cnn_project.models.model_mix import create_model


OVERLAP = 0.85
CMAP = "gray"
CHANNELS = ["C3", "C4"]

channel_string = "".join(CHANNELS)

LOSO_SUBJECT = 1
INPUT_FOLDER = f"generated_scalograms_C3C4_{CMAP}_overlap{OVERLAP}"
OUTPUT_FOLDER = f"gridsearch_mix_v2_{channel_string}_{CMAP}_overlap{OVERLAP}_loso{LOSO_SUBJECT}"
PROGRESS_FILE = config.OUTPUT_DIR / OUTPUT_FOLDER / "progress.json"


import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("scalogram_cnn_project").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    optimizers = [
    ("adam", keras.optimizers.Adam),
    ("sgd", keras.optimizers.SGD),
    ("rmsprop", keras.optimizers.RMSprop)
    ]

    fixed_params = {
        "loso_subject": LOSO_SUBJECT,
        "cmap": CMAP,
        "channels": CHANNELS,
        "seed": 42,
        "epsilon": 1e-3,
        "momentum": 0.99,
        "overlap": OVERLAP
    }

    learning_rates = [1e-3, 1e-4, 1e-5]
    batch_sizes = [16, 32, 64]

    grid_training_parameters = []

    for lr, bs, (opt_name, opt_class) in itertools.product(
        learning_rates,
        batch_sizes,
        optimizers):

        model_name = f"model_{opt_name}_lr{lr}_bs{bs}"
        params = fixed_params.copy()

        params.update({
            "model_name": model_name,
            "learning_rate": lr,
            "batch_size": bs,
            "optimizer_name": opt_name,
            "optimizer": opt_class(learning_rate=lr)
        })

        grid_training_parameters.append(params)

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

    # ==============================
    # GRID SEARCH LOOP
    # ==============================


    for params in grid_training_parameters:
        
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



