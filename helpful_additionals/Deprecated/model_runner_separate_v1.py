import tensorflow as tf
from tensorflow.math import sigmoid
from tensorflow import keras


from scalogram_cnn_project.utils.train_test_splitter_in_time import train_test_split
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scalogram_cnn_project.utils.load_data_separate import load_data
from pathlib import Path
import os

import logging
logger = logging.getLogger(__name__)

def run_model(training_parameters, model, callback, input_folder, output_folder):

    training_cmap = training_parameters["cmap"]
    training_channels = training_parameters["channels"]
    training_model_name = training_parameters["model_name"]
    training_seed = training_parameters["seed"]
    training_batch_size = training_parameters["batch_size"]
    training_overlap = training_parameters["overlap"]


    os.environ["PYTHONHASHSEED"] = str(training_seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    np.random.seed(training_seed)
    tf.random.set_seed(training_seed)
    tf.config.experimental.enable_op_determinism()


    X, y, Subject_array, Epoch_array = load_data(folder=input_folder,
                       channels=training_channels,
                       cmap=training_cmap)


    x_train, x_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.30,
        random_state=training_seed,
        overlap=training_overlap,
        subject_array = Subject_array,
        epoch_array = Epoch_array
    )

    history = model.fit(x = x_train, y = y_train,
                        epochs=50,
                        batch_size=training_batch_size, 
                        validation_data=(x_test, y_test),
                        callbacks=[callback],
                        )


    predictions = sigmoid( model.predict(x_test) ).numpy()

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
        plt.savefig(output_folder / f"{training_model_name}_{title}.png")
        plt.close()


    return final_accuracy

