import logging
import tensorflow as tf

logger = logging.getLogger(__name__)

def create_model(rnn_model, parameters: dict):
    """
    Creates an Anticipatory Coupled Classifier (v1) with Batch Normalization
    and high-capacity dense projection over the pre-trained recurrent latent state.
    
    Args:
        rnn_model (keras.Model): Pretrained forecasting model.
        parameters (dict): Hyperparameters for the classifier:
            - input_len (int): Sequence length in samples.
            - learning_rate (float): Learning rate for classification.
            - hidden_units_1 (int, optional): Neurons in first dense layer (default: 128).
            - dropout_1 (float, optional): Dropout rate for first dense layer (default: 0.3).
            - hidden_units_2 (int, optional): Neurons in second dense layer (default: 64).
            - dropout_2 (float, optional): Dropout rate for second dense layer (default: 0.2).
            - fine_tune_rnn (bool, optional): Unfreezes recurrent backbone if True (default: False).
            
    Returns:
        keras.Model: Compiled coupled classification model.
    """
    input_len = parameters.get("input_len")
    if input_len is None:
        if hasattr(rnn_model, "input_shape") and rnn_model.input_shape:
            input_len = rnn_model.input_shape[1]
        else:
            input_len = 100
            
    lr = parameters.get("learning_rate", 0.001)
    h1 = parameters.get("hidden_units_1", 128)
    d1 = parameters.get("dropout_1", 0.3)
    h2 = parameters.get("hidden_units_2", 64)
    d2 = parameters.get("dropout_2", 0.2)
    fine_tune = parameters.get("fine_tune_rnn", False)
    
    logger.info(f"Building Coupled Classifier v1 (BN + High Capacity, Input steps: {input_len}, Fine-tune: {fine_tune}, LR: {lr})...")
    
    # Check if we should initialize RNN backbone from scratch with random weights
    is_random_backbone = rnn_model is None or (isinstance(rnn_model, str) and rnn_model.lower() in ["none", "null", "random", "scratch"])
    
    inp = tf.keras.layers.Input(shape=(input_len, 1), name="signal_input")
    x = inp

    if is_random_backbone:
        latent_dim = parameters.get("latent_dim", 32)
        rnn_type = str(parameters.get("rnn_type", "lstm")).lower()
        logger.info(f"Instantiating fresh {rnn_type.upper()} backbone from scratch with random weights ({latent_dim} units)...")
        if rnn_type == "gru":
            x = tf.keras.layers.GRU(latent_dim, name="random_gru_backbone")(inp)
        else:
            x = tf.keras.layers.LSTM(latent_dim, name="random_lstm_backbone")(inp)
            
        feature_extractor = tf.keras.Model(inputs=inp, outputs=x, name="rnn_latent_backbone")
        # Freshly initialized backbone is trainable by default unless fine_tune_rnn is explicitly False
        feature_extractor.trainable = parameters.get("fine_tune_rnn", True)
    else:
        # Locate recurrent layer in pre-trained RNN model
        recurrent_layer = None
        for layer in rnn_model.layers:
            if isinstance(layer, (tf.keras.layers.LSTM, tf.keras.layers.GRU, tf.keras.layers.RNN)):
                recurrent_layer = layer
                break
                
        if recurrent_layer is not None:
            latent_dim = getattr(recurrent_layer, "units", 32)
            logger.info(f"Extracting latent state vector from recurrent layer '{recurrent_layer.name}' ({latent_dim} units)...")
            for l in rnn_model.layers:
                x = l(x)
                if l == recurrent_layer:
                    break
        else:
            logger.info("Using penultimate layer as feature extractor backbone...")
            for l in rnn_model.layers[:-1]:
                x = l(x)
                
        feature_extractor = tf.keras.Model(inputs=inp, outputs=x, name="rnn_latent_backbone")
        feature_extractor.trainable = fine_tune
    
    # Classification Head with Batch Normalization
    clf_model = tf.keras.Sequential([
        feature_extractor,
        tf.keras.layers.BatchNormalization(name="classifier_bn_1"),
        tf.keras.layers.Dense(h1, activation="relu", name="classifier_dense_1"),
        tf.keras.layers.Dropout(d1, name="classifier_dropout_1"),
        tf.keras.layers.Dense(h2, activation="relu", name="classifier_dense_2"),
        tf.keras.layers.Dropout(d2, name="classifier_dropout_2"),
        tf.keras.layers.Dense(1, activation="sigmoid", name="classifier_output")
    ])
    
    opt_name = parameters.get("optimizer_name", "adam").lower()
    if opt_name == "rmsprop":
        opt = tf.keras.optimizers.RMSprop(learning_rate=lr)
    elif opt_name == "sgd":
        opt = tf.keras.optimizers.SGD(learning_rate=lr)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        
    clf_model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    
    return clf_model
