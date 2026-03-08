import tensorflow as tf 
from tensorflow import keras
import numpy as np
import matplotlib
matplotlib.use("Agg")
import itertools
import time
from scalogram_cnn_project.model_runners.model_runner_mix_v0 import run_model
import scalogram_cnn_project.settings.config as config

OVERLAP = 0.85
CMAP = "gray"
INPUT_FOLDER = f"generated_scalograms_C3C4_{CMAP}_overlap_{OVERLAP}"
OUTPUT_FOLDER = "useless"
CHANNELS = ["C3", "C4"]


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
        "momentum": 0.99
    }

    learning_rates = [1e-3, 5e-4, 3e-4, 1e-4, 5e-5, 3e-5, 1e-5]
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
        acc = run_model(master_parameters = params, 
                        input_folder= config.DATA_DIR / INPUT_FOLDER,
                        output_folder= config.OUTPUT_DIR / OUTPUT_FOLDER)
        results.append((params["model_name"], acc))



    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds\n\n")


    print(f"Total combinations: {len(grid_master_parameters)}\n\n")
    print(results)


