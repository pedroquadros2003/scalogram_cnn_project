import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import numpy as np
import time
from scalogram_cnn_project.model_runners.model_runner_separate_v2 import run_model
import scalogram_cnn_project.settings.config as config

OVERLAP = 0.85
CMAP = "gray"
CHANNELS = ["C3", "C4"]

channel_string = "".join(CHANNELS)

INPUT_FOLDER = f"generated_scalograms_C3C4_{CMAP}_overlap{OVERLAP}"
OUTPUT_FOLDER = f"cross_validation_v2_{channel_string}_{CMAP}_overlap{OVERLAP}"

LOSO_SUBJECTS = [1, 2, 3, 4, 5, 6, 8, 11, 14]
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
OPTIMIZER = "adam"


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
        "cmap": CMAP,
        "channels": CHANNELS,
        "seed": 42,
        "epsilon": 1e-3,
        "momentum": 0.99,
        "overlap": OVERLAP,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "optimizer_name": OPTIMIZER,
        "optimizer": optimizers[OPTIMIZER](learning_rate=LEARNING_RATE)
    }



    grid_master_parameters = []

    for subject in LOSO_SUBJECTS: 

        params = fixed_params.copy()

        params.update({
            "loso_subject": subject,
            "model_name": f"model_{OPTIMIZER}_lr{LEARNING_RATE}_bs{BATCH_SIZE}_loso{subject}",
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
    logger.info(f"Elapsed time: {elapsed_time:.4f} seconds\n\n")



    logger.info("Results of each LOSO: %s\n\n", results)
    mean = sum(row[1] for row in results) / len(results)
    logger.info("The mean accuracy is: %s", mean)


