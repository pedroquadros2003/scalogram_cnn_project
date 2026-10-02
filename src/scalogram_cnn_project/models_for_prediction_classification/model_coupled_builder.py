import importlib
import logging

logger = logging.getLogger(__name__)

def create_coupled_classifier_model(model_version: str, rnn_model, parameters: dict):
    """
    Dynamically loads and instantiates a coupled anticipatory classifier model
    based on a pre-trained forecasting RNN backbone and an MLP classification head.
    
    Args:
        model_version (str): The classifier architecture version (e.g. "v0", "v1").
        rnn_model (keras.Model): Pre-trained forecasting RNN model (or model path).
        parameters (dict): Hyperparameters for the classification model.
        
    Returns:
        keras.Model: Instantiated and compiled coupled classifier model.
    """
    version = str(model_version).lower().strip()
    if not version.startswith("v"):
        if "v" in version:
            version = version.split("v")[-1]
        version = f"v{version}"
        
    module_name = f"scalogram_cnn_project.models_for_prediction_classification.model_coupled_{version}"
    logger.info(f"Loading coupled classification model from module: {module_name}")
    
    try:
        model_module = importlib.import_module(module_name)
    except ImportError as e:
        logger.error(f"Failed to import coupled classification model module: {module_name}")
        raise ValueError(f"Unknown coupled model version: '{model_version}'. Error: {e}")
        
    if not hasattr(model_module, "create_model"):
        raise AttributeError(f"Module {module_name} does not have a 'create_model' function.")
        
    return model_module.create_model(rnn_model, parameters)
