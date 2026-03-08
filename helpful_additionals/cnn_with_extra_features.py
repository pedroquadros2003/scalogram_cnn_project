"""
Example: CNN with Additional Manual Features (Keras Functional API)

This script demonstrates how to extend a convolutional neural network (CNN)
so that it can receive two types of inputs:

1) An image input (e.g., 64x64x2 scalograms)
2) Additional handcrafted features chosen by the user

The CNN processes the image normally. After the Flatten layer, we concatenate
the CNN features with the external features and feed them to dense layers.

This approach is common in scientific machine learning where:
- CNNs learn visual patterns
- Domain knowledge is injected through engineered features
"""

from tensorflow import keras
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense,
    Concatenate
)
from tensorflow.keras.models import Model

# ------------------------------------------------------------------
# Hyperparameters (example values)
# In practice these could come from a configuration file or grid search
# ------------------------------------------------------------------

master_seed = 42
master_momentum = 0.99
master_epsilon = 1e-3

# Number of additional handcrafted features
k = 5


# ------------------------------------------------------------------
# IMAGE INPUT BRANCH (CNN)
# ------------------------------------------------------------------

# Define the image input tensor
image_input = Input(shape=(64, 64, 2), name="image_input")

x = Conv2D(64, (3, 3), activation='relu')(image_input)

# Batch normalization helps stabilize training
x = BatchNormalization(
    momentum=master_momentum,
    epsilon=master_epsilon,
    center=True,
    scale=True,
    beta_initializer="zeros",
    gamma_initializer="ones",
    moving_mean_initializer="zeros",
    moving_variance_initializer="ones",
)(x)

# Downsample spatial dimensions
x = MaxPooling2D((2, 2))(x)

# Dropout for regularization
x = Dropout(0.25, seed=master_seed)(x)

# Second convolutional block
x = Conv2D(32, (3, 3), activation='relu')(x)

x = BatchNormalization(
    momentum=master_momentum,
    epsilon=master_epsilon,
    center=True,
    scale=True,
    beta_initializer="zeros",
    gamma_initializer="ones",
    moving_mean_initializer="zeros",
    moving_variance_initializer="ones",
)(x)

x = MaxPooling2D((2, 2))(x)

x = Dropout(0.25, seed=master_seed)(x)

# Flatten the CNN output into a vector
x = Flatten()(x)


# ------------------------------------------------------------------
# EXTRA FEATURE INPUT
# ------------------------------------------------------------------

# This input contains manually selected features
# Example: statistics, physical parameters, spectral descriptors, etc.
extra_input = Input(shape=(k,), name="extra_features")


# Optional: process the extra features with a small MLP
# before merging them with the CNN features
extra = Dense(16, activation='relu')(extra_input)
extra = Dense(8, activation='relu')(extra)


# ------------------------------------------------------------------
# FEATURE FUSION
# ------------------------------------------------------------------

# Concatenate CNN features with the engineered features
x = Concatenate()([x, extra])


# ------------------------------------------------------------------
# CLASSIFIER / REGRESSION HEAD
# ------------------------------------------------------------------

# Fully connected layer
x = Dense(128, activation='relu')(x)

# Additional dropout for regularization
x = Dropout(0.5, seed=master_seed)(x)

# Output layer producing a single logit
# (no sigmoid applied yet)
output = Dense(1, name="logit_output")(x)


# ------------------------------------------------------------------
# MODEL DEFINITION
# ------------------------------------------------------------------

model = Model(
    inputs=[image_input, extra_input],
    outputs=output,
    name="cnn_with_feature_injection"
)


# ------------------------------------------------------------------
# PRINT MODEL SUMMARY
# ------------------------------------------------------------------

model.summary()


# ------------------------------------------------------------------
# TRAINING USAGE EXAMPLE
# ------------------------------------------------------------------

"""
Training requires passing both inputs.

Example shapes:

X_images : (N, 64, 64, 2)
X_extra  : (N, k)
y        : (N,)

Example:

model.fit(
    [X_images, X_extra],
    y,
    batch_size=32,
    epochs=10
)
"""
