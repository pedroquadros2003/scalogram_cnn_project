import tensorflow as tf
from sklearn.model_selection import train_test_split
from scalogram_cnn_project.utils.balance_indices_undersampling import balanced_indices_undersmp
from scalogram_cnn_project.utils.generic_operations_list_of_numpy import index_X


from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    model_name = parameters["model_name"]
    seed = parameters["seed"]
    batch_size = parameters["batch_size"]
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
        plt.savefig(output_folder / f"{model_name}_{title}.png")
        plt.close()


    return final_accuracy
