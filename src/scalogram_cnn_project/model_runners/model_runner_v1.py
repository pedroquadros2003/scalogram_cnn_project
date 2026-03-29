import tensorflow as tf
from scalogram_cnn_project.utils.train_test_splitter_in_time import train_test_split
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import os

from scalogram_cnn_project.utils import load_data_mix, load_data_separate

loaders = {
    "mix": load_data_mix.load_data,
    "separate": load_data_separate.load_data,
}


import logging
logger = logging.getLogger(__name__)

def run_model(parameters, model, callback, input_folder, output_folder):

    cmap = parameters["cmap"]
    channels = parameters["channels"]
    model_id = parameters["model_id"]
    seed = parameters["seed"]
    batch_size = parameters["batch_size"]
    overlap = parameters["overlap"]
    mode = parameters["mode"]
    subjects = parameters["subjects"]

    additional_features = False
    if "n_additional_features" in parameters:
        additional_features = True


    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    #os.environ["TF_DETERMINISTIC_OPS"] = "1"
    #tf.config.experimental.enable_op_determinism()


    load_data = loaders[mode]

    X, y, Subject_array, Epoch_array = load_data(folder_path=input_folder,
                       channels=channels,
                       cmap=cmap,
                       subjects=subjects,
                       additional_features=additional_features)


    ## As mix version of the model does not differentiate channels,
    ## it is necessary to have a greater step for neglected epochs
    ## in order to skip the right amount of epochs for every channel.
    neglected_epochs_step = 1 
    if mode == "separate":
        neglected_epochs_step = 1
    elif mode == "mix":
        neglected_epochs_step = len(channels)
    else:
       neglected_epochs_step = 1 


    x_train, x_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.30,
        random_state=seed,
        overlap=overlap,
        subject_array = Subject_array,
        epoch_array = Epoch_array,
        neglected_epochs_step = neglected_epochs_step
    )

    history = model.fit(x = x_train, y = y_train,
                        epochs=50,
                        batch_size=batch_size, 
                        validation_data=(x_test, y_test),
                        callbacks=[callback],
                        )


    predictions = tf.math.sigmoid( model.predict(x_test) ).numpy()

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


    return final_accuracy

