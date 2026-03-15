import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import numpy as np
import itertools
import time
from scalogram_cnn_project.model_runners.model_runner_separate_v2 import run_model
import scalogram_cnn_project.settings.config as config

OVERLAP = 0.85
CMAP = "gray"
CHANNELS = ["C3", "C4"]

channel_string = "".join(CHANNELS)

LOSO_SUBJECT = 1
INPUT_FOLDER = f"generated_scalograms_C3C4_{CMAP}_overlap{OVERLAP}"
OUTPUT_FOLDER = f"gridsearch_separate_v2_{channel_string}_{CMAP}_overlap{OVERLAP}_loso{LOSO_SUBJECT}"


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

    grid_master_parameters = []

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

        grid_master_parameters.append(params)


    start_time = time.perf_counter()

    results = []

    for params in grid_master_parameters:
        acc = run_model(master_parameters=params, 
                        input_folder = config.DATA_DIR / INPUT_FOLDER,
                        output_folder = config.OUTPUT_DIR / OUTPUT_FOLDER)
        results.append((params["model_name"], acc))



    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    logger.info("Elapsed time: %s seconds\n\n", elapsed_time)


    logger.info("Total combinations: %s\n\n", len(grid_master_parameters))
    logger.info(results)


