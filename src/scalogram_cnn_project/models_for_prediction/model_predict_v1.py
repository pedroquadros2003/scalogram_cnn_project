import tensorflow as tf
from keras.models import Sequential
from keras.layers import GRU, Dense, Input, Reshape
from keras.optimizers import Adam, SGD, RMSprop
import logging

logger = logging.getLogger(__name__)

OPTIMIZERS = {
    "adam": Adam,
    "sgd": SGD,
    "rmsprop": RMSprop,
}

def create_model(parameters):
    """
    Creates a GRU prediction model with direct Dense projection to output.
    Required keys in parameters:
        - input_steps (int): number of time steps in input sequence.
        - output_steps (int): number of time steps in output sequence.
        - latent_dim (int): units for GRU layers.
        - optimizer_name (str): optimizer to use.
        - learning_rate (float): learning rate.
        
    Returns:
        keras.Model: Compiled Keras Sequential model.
    """
    logger.info("Initializing GRU Direct prediction model...")
    
    input_steps = parameters["input_steps"]
    output_steps = parameters["output_steps"]
    latent_dim = parameters.get("latent_dim", 64)
    opt_name = parameters.get("optimizer_name", "adam")
    lr = parameters.get("learning_rate", 0.001)
    
    features = 1  # 1D single-channel forecasting
    
    model = Sequential([
        Input(shape=(input_steps, features)),
        GRU(latent_dim, activation='tanh', return_sequences=True),
        GRU(latent_dim, activation='tanh', return_sequences=False),
        Dense(output_steps * features),
        Reshape((output_steps, features))
    ])
    
    opt_class = OPTIMIZERS.get(opt_name.lower(), Adam)
    optimizer = opt_class(learning_rate=lr)
    
    model.compile(optimizer=optimizer, loss='mse')
    return model
