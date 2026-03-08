import tensorflow as tf
from tensorflow.math import sigmoid
from tensorflow import keras


from scalogram_cnn_project.utils.train_test_splitter_in_subjects import train_test_split
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scalogram_cnn_project.utils.load_data_separate import load_data
from pathlib import Path
import os
from scalogram_cnn_project.models.model_separate_v1 import create_model



def run_model(master_parameters, input_folder, output_folder):

    master_cmap = master_parameters["cmap"]
    master_channels = master_parameters["channels"]
    master_model_name = master_parameters["model_name"]
    master_seed = master_parameters["seed"]
    master_batch_size = master_parameters["batch_size"]
    master_loso_subject = master_parameters["loso_subject"]

    os.environ["PYTHONHASHSEED"] = str(master_seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    np.random.seed(master_seed)
    tf.random.set_seed(master_seed)
    tf.config.experimental.enable_op_determinism()


    (X, y, Subject_array) = load_data(folder=input_folder,
                       channels=master_channels,
                       cmap=master_cmap)


    x_train, x_test, y_train, y_test = train_test_split(
        X, y,
        random_state=master_seed,
        subject_array = Subject_array,
        loso_subject = master_loso_subject
    )

    model, callback = create_model(master_parameters)

    history = model.fit(x = x_train, y = y_train,
                        epochs=50,
                        batch_size=master_batch_size, 
                        validation_data=(x_test, y_test),
                        callbacks=[callback],
                        )


    predictions = sigmoid( model.predict(x_test) ).numpy()

    error_classification = []

    for i in range(len(predictions)):
      if round(float(predictions[i][0])) != int(y_test[i][0]):
        error_classification.append(i)

    final_accuracy = ( 100*float( 1- len(error_classification)/len(predictions)))
    print(f'\n\nFinal Accuracy: { final_accuracy }\n\n')

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
        plt.savefig(output_folder / f"{master_model_name}_{title}.png")
        plt.close()


    return final_accuracy

