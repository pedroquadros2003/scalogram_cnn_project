import tensorflow as tf
from keras.metrics import BinaryAccuracy
from keras.losses import BinaryCrossentropy
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import BatchNormalization
from scalogram_cnn_project.utils.validate_dict_params  import validate_dict_params

from pathlib import Path
import numpy as np

import logging
logger = logging.getLogger(__name__)


from keras.optimizers import Adam, SGD, RMSprop
OPTIMIZERS = {
    "adam"   : Adam,
    "sgd"    : SGD,
    "rmsprop": RMSprop,
}


def create_model(parameters):

    REQUIRED_TRAIN_KEYS = ["seed", "optimizer_name", "batch_size", "subjects", "overlap", "learning_rate"]
    REQUIRED_MODEL_KEYS = ["channels", "epsilon", "momentum", "cmap", "mode", \
                            "kernel_size", "extra_layer", "extra_layer_num_filters", "num_neurons_dense", \
                            "first_layer_num_filters", "second_layer_num_filters", "from_logit"]


    logger.info("Validating training parameters...")
    validate_dict_params(parameters, REQUIRED_TRAIN_KEYS)


    seed = parameters["seed"]

    # Optimizer
    opt_name = parameters["optimizer_name"]
    opt_class = OPTIMIZERS[opt_name]
    optimizer = opt_class(learning_rate=parameters["learning_rate"])


    logger.info("Validating model parameters...")
    validate_dict_params(parameters, REQUIRED_MODEL_KEYS)

    channels = parameters["channels"]
    epsilon = parameters["epsilon"]
    momentum = parameters["momentum"]
    cmap = parameters["cmap"]
    mode = parameters["mode"]
    kernel_size = parameters["kernel_size"]
    extra_layer = parameters["extra_layer"]
    extra_layer_num_filters = parameters["extra_layer_num_filters"]
    num_neurons_dense = parameters["num_neurons_dense"]
    first_layer_num_filters = parameters["first_layer_num_filters"]
    second_layer_num_filters = parameters["second_layer_num_filters"]
    from_logit = parameters["from_logit"]




    color_channels_per_image = 1 if cmap == "gray" else 3
    mode_multiplier = 1 if mode == "mix" else len(channels)



    model = Sequential()
    model.add( Input(shape=(64,64,color_channels_per_image*mode_multiplier)) ),
    model.add(Conv2D(first_layer_num_filters, (kernel_size,kernel_size), activation='relu')),
    model.add(BatchNormalization(
                                momentum=momentum,
                                epsilon=epsilon,
                                center=True,
                                scale=True,
                                beta_initializer="zeros",
                                gamma_initializer="ones",
                                moving_mean_initializer="zeros",
                                moving_variance_initializer="ones",
                                )),
    model.add(MaxPooling2D(2,2)),
    model.add(Dropout(0.5, seed = seed)),
    model.add(Conv2D(second_layer_num_filters, (kernel_size,kernel_size), activation='relu')),
    model.add(BatchNormalization(
                                momentum=momentum,
                                epsilon=epsilon,
                                center=True,
                                scale=True,
                                beta_initializer="zeros",
                                gamma_initializer="ones",
                                moving_mean_initializer="zeros",
                                moving_variance_initializer="ones",
                                )),
    model.add(MaxPooling2D(2,2)),
    model.add(Dropout(0.5, seed = seed)),
    
    if extra_layer:
    
        model.add(Conv2D(extra_layer_num_filters, (kernel_size,kernel_size), activation='relu')),
        model.add(BatchNormalization(
                                    momentum=momentum,
                                    epsilon=epsilon,
                                    center=True,
                                    scale=True,
                                    beta_initializer="zeros",
                                    gamma_initializer="ones",
                                    moving_mean_initializer="zeros",
                                    moving_variance_initializer="ones",
                                    )),
        model.add(MaxPooling2D(2,2)),
        model.add(Dropout(0.5, seed = seed)),
    

    model.add(Flatten()),
    model.add(Dense(num_neurons_dense, activation='relu')),
    model.add(Dropout(0.7, seed = seed)),
    
    if from_logit:
        model.add(Dense(1))
    else:
        model.add(Dense(1, activation='sigmoid'))


    metrics=[
            BinaryAccuracy(threshold=0.0 if from_logit else 0.5, name="accuracy"),
            ]

    model.compile(optimizer=optimizer,
                loss=BinaryCrossentropy(from_logits=from_logit),
                metrics = metrics
                )

    callback = EarlyStopping(
                monitor="val_loss",
                min_delta=0,
                patience=10,
                mode="min",
                restore_best_weights=False,
                start_from_epoch=0, 
    )

    return model, callback




if __name__ == "__main__":
    from keras.optimizers import Adam

    params = {}

    params["channels"] = ["C3", "C4", "Cz", "Pz", "Fz"]
    params["seed"] = 42
    params["epsilon"] = 1e-3
    params["momentum"] = 0.99
    params["optimizer_name"] = "adam"
    params["batch_size"] = 32
    params["subjects"] = [1, 2, 3]
    params["overlap"] = 0.733
    params["learning_rate"] = 0.001
    params["cmap"] = "gray"
    params["mode"] = "mix"
    params["from_logit"] = False


    params["extra_layer"] = True
    params["extra_layer_num_filters"] = 16
    params["first_layer_num_filters"] = 64
    params["second_layer_num_filters"] = 64
    params["kernel_size"] = 2
    params["num_neurons_dense"] = 128
     

    model, callback = create_model(params)
    model.summary()
