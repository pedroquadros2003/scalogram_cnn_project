import logging
import tensorflow as tf

logger = logging.getLogger(__name__)

OPTIMIZERS = {
    "adam": tf.keras.optimizers.Adam,
    "sgd": tf.keras.optimizers.SGD,
    "rmsprop": tf.keras.optimizers.RMSprop,
}

def create_joint_model(
    predict_version: str = "v0",
    classifier_version: str = "v0",
    parameters: dict = None
) -> tf.keras.Model:
    """
    Creates a unified multi-output Functional Keras model that couples temporal signal forecasting
    and drowsiness state classification over a shared recurrent backbone (LSTM or GRU).
    
    All layers are initialized completely from scratch with random weights to ensure zero data leakage.
    
    Args:
        predict_version (str): Forecasting backbone architecture ("v0" for LSTM, "v1" for GRU).
        classifier_version (str): Classification head architecture ("v0" for standard MLP, "v1" for BN-MLP).
        parameters (dict, optional): Hyperparameter dictionary:
            - input_len (int): Length of past input signal window (default: 100 samples).
            - forecast_steps (int): Number of steps ahead to predict for forecasting (default: 1).
            - latent_dim (int): Number of recurrent units in shared backbone (default: 32).
            - loss_alpha (float): Trade-off parameter alpha in [0.0, 1.0] balancing classification and forecasting:
                                  Loss_total = alpha * Loss_clf + (1 - alpha) * Loss_forecast (default: 0.5).
            - learning_rate (float): Optimizer learning rate (default: 0.001).
            - optimizer_name (str): Optimizer to use ("adam", "rmsprop", "sgd") (default: "adam").
            - hidden_units_1 (int): First dense layer units in classifier head (default: 64 for v0, 128 for v1).
            - hidden_units_2 (int): Second dense layer units in classifier head (default: 32 for v0, 64 for v1).
            - dropout_1 (float): Dropout rate for first classifier dense layer (default: 0.3).
            - dropout_2 (float): Dropout rate for second classifier dense layer (default: 0.2).
            
    Returns:
        tf.keras.Model: Compiled multi-output Keras Functional Model with outputs:
            - "forecast_output": Linear prediction of future voltage (shape: (batch, forecast_steps)).
            - "classification_output": Sigmoid vigilance/drowsiness probability (shape: (batch, 1)).
    """
    if parameters is None:
        parameters = {}
        
    p_ver = str(predict_version).lower().strip().replace("predict_", "")
    if not p_ver.startswith("v"):
        p_ver = f"v{p_ver}"
        
    c_ver = str(classifier_version).lower().strip().replace("coupled_", "").replace("classifier_", "")
    if not c_ver.startswith("v"):
        c_ver = f"v{c_ver}"
        
    input_len = parameters.get("input_len", 100)
    forecast_steps = parameters.get("forecast_steps", 1)
    latent_dim = parameters.get("latent_dim", 32)
    alpha = float(parameters.get("loss_alpha", 0.5))
    alpha = max(0.0, min(1.0, alpha)) # Clip to [0.0, 1.0]
    
    lr = float(parameters.get("learning_rate", 0.001))
    opt_name = str(parameters.get("optimizer_name", "adam")).lower()
    
    default_h1 = 128 if c_ver == "v1" else 64
    default_h2 = 64 if c_ver == "v1" else 32
    h1 = parameters.get("hidden_units_1", default_h1)
    h2 = parameters.get("hidden_units_2", default_h2)
    d1 = parameters.get("dropout_1", 0.3)
    d2 = parameters.get("dropout_2", 0.2)
    
    logger.info(
        f"Building Joint Model [Predictor: {p_ver.upper()} | Classifier: {c_ver.upper()}] "
        f"(input_len={input_len}, latent_dim={latent_dim}, forecast_steps={forecast_steps}, alpha={alpha:.3f}, lr={lr})..."
    )
    
    # 1. Shared Input Layer
    inp = tf.keras.layers.Input(shape=(input_len, 1), name="signal_input")
    
    # 2. Shared Recurrent Backbone (Initialized from scratch with random weights)
    if p_ver == "v1" or "gru" in p_ver:
        recurrent_layer = tf.keras.layers.GRU(
            latent_dim,
            activation="tanh",
            return_sequences=False,
            kernel_initializer="glorot_uniform",
            name="shared_gru_backbone"
        )
    else:
        recurrent_layer = tf.keras.layers.LSTM(
            latent_dim,
            activation="tanh",
            return_sequences=False,
            kernel_initializer="glorot_uniform",
            name="shared_lstm_backbone"
        )
        
    latent_state = recurrent_layer(inp) # Shape: (batch, latent_dim)
    
    # 3. Branch 1: Forecasting Output Head (Linear voltage prediction)
    if forecast_steps == 1:
        forecast_output = tf.keras.layers.Dense(
            1,
            activation="linear",
            kernel_initializer="glorot_uniform",
            name="forecast_output"
        )(latent_state)
    else:
        forecast_dense = tf.keras.layers.Dense(
            forecast_steps,
            activation="linear",
            kernel_initializer="glorot_uniform",
            name="forecast_dense"
        )(latent_state)
        forecast_output = tf.keras.layers.Reshape((forecast_steps,), name="forecast_output")(forecast_dense)
        
    # 4. Branch 2: Classification Output Head (Vigilance / Drowsiness probability)
    clf_x = latent_state
    if c_ver == "v1":
        # v1 includes Batch Normalization for regularization
        clf_x = tf.keras.layers.BatchNormalization(name="classifier_bn_1")(clf_x)
        
    clf_x = tf.keras.layers.Dense(
        h1,
        activation="relu",
        kernel_initializer="glorot_uniform",
        name="classifier_dense_1"
    )(clf_x)
    clf_x = tf.keras.layers.Dropout(d1, name="classifier_dropout_1")(clf_x)
    clf_x = tf.keras.layers.Dense(
        h2,
        activation="relu",
        kernel_initializer="glorot_uniform",
        name="classifier_dense_2"
    )(clf_x)
    clf_x = tf.keras.layers.Dropout(d2, name="classifier_dropout_2")(clf_x)
    
    classification_output = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        kernel_initializer="glorot_uniform",
        name="classification_output"
    )(clf_x)
    
    # 5. Build Functional Model
    model = tf.keras.Model(
        inputs=inp,
        outputs={
            "forecast_output": forecast_output,
            "classification_output": classification_output
        },
        name=f"joint_model_{p_ver}_{c_ver}"
    )
    
    # 6. Compile Model with Single Balanced alpha Weighting
    # Loss_total = alpha * Loss_clf + (1 - alpha) * Loss_forecast
    loss_weights = {
        "classification_output": alpha,
        "forecast_output": 1.0 - alpha
    }
    
    losses = {
        "classification_output": "binary_crossentropy",
        "forecast_output": "mse"
    }
    
    metrics = {
        "classification_output": [
            "accuracy",
            tf.keras.metrics.AUC(name="auc")
        ],
        "forecast_output": [
            "mae",
            tf.keras.metrics.RootMeanSquaredError(name="rmse")
        ]
    }
    
    opt_class = OPTIMIZERS.get(opt_name, tf.keras.optimizers.Adam)
    optimizer = opt_class(learning_rate=lr)
    
    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics
    )
    
    logger.info(
        f"Joint model compiled successfully. Loss weights: "
        f"Classification={loss_weights['classification_output']:.3f}, Forecast={loss_weights['forecast_output']:.3f}"
    )
    
    return model
