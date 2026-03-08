import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Input

import numpy as np
from pathlib import Path
import os


def create_model(master_parameters):

    master_channels = master_parameters["channels"]
    master_seed = master_parameters["seed"]
    master_epsilon = master_parameters["epsilon"]
    master_momentum = master_parameters["momentum"]
    master_optimizer = master_parameters["optimizer"]
    master_cmap = master_parameters["cmap"]
    

    color_channels_per_image = 3

    if master_cmap == "gray":
        color_channels_per_image=1
    elif master_cmap == "viridis":
        color_channels_per_image=3
    else:
        color_channels_per_image=3



    model = Sequential()
    model.add( Input(shape = (64,64,color_channels_per_image*len(master_channels)) ) ),
    model.add(Conv2D(64, (3,3), activation='relu')),
    model.add(keras.layers.BatchNormalization(
                                momentum=master_momentum,
                                epsilon=master_epsilon,
                                center=True,
                                scale=True,
                                beta_initializer="zeros",
                                gamma_initializer="ones",
                                moving_mean_initializer="zeros",
                                moving_variance_initializer="ones",
                                )),
    model.add(MaxPooling2D(2,2)),
    model.add(Dropout(0.25, seed = master_seed)),
    model.add(Conv2D(32, (3,3), activation='relu')),
    model.add(keras.layers.BatchNormalization(
                                momentum=master_momentum,
                                epsilon=master_epsilon,
                                center=True,
                                scale=True,
                                beta_initializer="zeros",
                                gamma_initializer="ones",
                                moving_mean_initializer="zeros",
                                moving_variance_initializer="ones",
                                )),
    model.add(MaxPooling2D(2,2)),
    model.add(Dropout(0.25, seed = master_seed)),
    model.add(Flatten()),
    model.add(Dense(128, activation='relu')),
    model.add(Dropout(0.5, seed = master_seed)),
    model.add(Dense(1))


    metrics=[
            keras.metrics.BinaryAccuracy(threshold=0.0, name="accuracy"),
            keras.metrics.AUC(num_thresholds=200, curve="ROC", from_logits=True, name="auc"),
            ]

    model.compile(optimizer=master_optimizer,
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