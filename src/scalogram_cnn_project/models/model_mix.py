import tensorflow as tf
from tensorflow.math import sigmoid
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Input


from pathlib import Path
import numpy as np

import logging
logger = logging.getLogger(__name__)

def create_model(training_parameters):

    training_seed = training_parameters["seed"]
    training_epsilon = training_parameters["epsilon"]
    training_momentum = training_parameters["momentum"]
    training_optimizer = training_parameters["optimizer"]
    training_cmap = training_parameters["cmap"]
    
    
    color_channels_per_image = 3

    if training_cmap == "gray":
        color_channels_per_image=1
    elif training_cmap == "viridis":
        color_channels_per_image=3
    else:
        color_channels_per_image=3


    model = Sequential()
    model.add( Input(shape=(64,64,color_channels_per_image)) ),
    model.add(Conv2D(64, (3,3), activation='relu')),
    model.add(keras.layers.BatchNormalization(
                                momentum=training_momentum,
                                epsilon=training_epsilon,
                                center=True,
                                scale=True,
                                beta_initializer="zeros",
                                gamma_initializer="ones",
                                moving_mean_initializer="zeros",
                                moving_variance_initializer="ones",
                                )),
    model.add(MaxPooling2D(2,2)),
    model.add(Dropout(0.25, seed = training_seed)),
    model.add(Conv2D(32, (3,3), activation='relu')),
    model.add(keras.layers.BatchNormalization(
                                momentum=training_momentum,
                                epsilon=training_epsilon,
                                center=True,
                                scale=True,
                                beta_initializer="zeros",
                                gamma_initializer="ones",
                                moving_mean_initializer="zeros",
                                moving_variance_initializer="ones",
                                )),
    model.add(MaxPooling2D(2,2)),
    model.add(Dropout(0.25, seed = training_seed)),
    model.add(Flatten()),
    model.add(Dense(128, activation='relu')),
    model.add(Dropout(0.5, seed = training_seed)),
    model.add(Dense(1))


    metrics=[
            keras.metrics.BinaryAccuracy(threshold=0.0, name="accuracy"),
            ]

    model.compile(optimizer=training_optimizer,
                loss=keras.losses.BinaryCrossentropy(from_logits=True),
                metrics = metrics
                )

    callback = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0,
                patience=10,
                mode="min",
                restore_best_weights=False,
                start_from_epoch=0, 
    )

    return model, callback