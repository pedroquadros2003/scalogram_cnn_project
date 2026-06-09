import tensorflow as tf
from sklearn.model_selection import train_test_split
from scalogram_cnn_project.utils.balance_indices_undersampling import balanced_indices_undersmp
from scalogram_cnn_project.utils.generic_operations_list_of_numpy import index_X

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from scalogram_cnn_project.utils_drozy import (
    load_data_mix as drozy_mix,
    load_data_separate as drozy_sep
)

from scalogram_cnn_project.utils_seed_vig import (
    load_data_mix as seedvig_mix,
    load_data_separate as seedvig_sep
)


LOADERS = {
    "DROZY": {
        "mix": drozy_mix.load_data,
        "separate": drozy_sep.load_data,
    },

    "SEED-VIG": {
        "mix": seedvig_mix.load_data,
        "separate": seedvig_sep.load_data,
    }
}



import logging
logger = logging.getLogger(__name__)

def run_model(parameters, model, callback, input_folder, output_folder):

    cmap = parameters["cmap"]
    channels = parameters["channels"]
    model_id = parameters["model_id"]
    seed = parameters["seed"]
    batch_size = parameters["batch_size"]
    mode = parameters["mode"]
    subjects = parameters["subjects"]
    from_logit = parameters["from_logit"]
    num_epochs = parameters["num_epochs"]
    preprocessing = parameters["preprocessing"]

    additional_features = False
    if "n_additional_features" in parameters and parameters["n_additional_features"] > 0:
        additional_features = True

    ## Determining the dataset from which the scalograms are
    dataset_config_path = input_folder / "dataset_config.json"
    with open(dataset_config_path) as f:
        dataset_config = json.load(f)

    # Comparing YAML parameters with dataset_config and raising error if mismatch
    if "scalogram" in dataset_config:
        ds_w = dataset_config["scalogram"].get("final_width_px")
        ds_h = dataset_config["scalogram"].get("final_height_px")
        yaml_w = parameters.get("final_width_px")
        yaml_h = parameters.get("final_height_px")
        
        if ds_w is not None and yaml_w is not None and ds_w != yaml_w:
            raise ValueError(f"Width mismatch: YAML specifies {yaml_w}, but dataset_config has {ds_w}.")
            
        if ds_h is not None and yaml_h is not None and ds_h != yaml_h:
            raise ValueError(f"Height mismatch: YAML specifies {yaml_h}, but dataset_config has {ds_h}.")



    if preprocessing == "rpca_juxtaposed" and mode == "separate":
        mode = "mix"
        logger.warning("Separate mode doesn't work with rpca_juxtaposed, changing to 'mix' mode.")



    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    #os.environ["TF_DETERMINISTIC_OPS"] = "1"
    #tf.config.experimental.enable_op_determinism()


    ## Determining the dataset from which the scalograms are

    dataset_config_path = input_folder / "dataset_config.json"

    with open(dataset_config_path) as f:
        dataset_config = json.load(f)

    dataset_name = dataset_config["dataset"]

    load_data = LOADERS[dataset_name][mode]

    ## Loading data with the appropriate loader

    X, y, _, _ = load_data(folder_path=input_folder,
                       channels=channels,
                       cmap=cmap,
                       subjects=subjects,
                       additional_features=additional_features)


    indices = balanced_indices_undersmp(y, seed)
    X = X[indices]
    y = y[indices]


    x_train, x_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.30,
        random_state=seed
    )


    history = model.fit(x = x_train, y = y_train,
                        epochs=num_epochs,
                        batch_size=batch_size, 
                        validation_data=(x_test, y_test),
                        callbacks=[callback],
                        )


    if from_logit:
       predictions = tf.math.sigmoid( model.predict(x_test) ).numpy()
    else:
       predictions = model.predict(x_test)

    error_classification = []

    for i in range(len(predictions)):
      if round(float(predictions[i][0])) != int(y_test[i][0]):
        error_classification.append(i)

    final_accuracy = ( 100*float( 1- len(error_classification)/len(predictions)))
    logger.info(f'\n\nFinal Accuracy: { final_accuracy }\n\n')

    metrics = [
        ("accuracy", "Accuracy"),
        ("loss", "Loss"),
    ]

    os.makedirs(output_folder, exist_ok=True)

    for key, title in metrics:
        plt.figure()
        plt.plot(history.history[key])
        plt.plot(history.history["val_" + key])
        plt.title(f"Model {title}")
        plt.ylabel(title)
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"])
        plt.savefig(output_folder / f"{model_id}_{title}.png")
        plt.close()


    return final_accuracy, history
