import importlib
import logging

logger = logging.getLogger(__name__)

def create_prediction_model(model_version: str, parameters: dict):
    """
    Dynamically loads and instantiates a forecasting model.
    
    Args:
        model_version (str): The model version, e.g. "v0", "v1"
        parameters (dict): Hyperparameters passed to the model creation function
        
    Returns:
        keras.Model: Instantiated and compiled model
    """
    # Normalize model version (e.g. "v0" or "model_predict_v0")
    version = str(model_version).lower().strip()
    if not version.startswith("v"):
        # Extract version number if it was passed like "model_predict_v0" or just a number
        if "v" in version:
            version = version.split("v")[-1]
        version = f"v{version}"
        
    module_name = f"scalogram_cnn_project.models_for_prediction.model_predict_{version}"
    logger.info(f"Loading prediction model from module: {module_name}")
    
    try:
        model_module = importlib.import_module(module_name)
    except ImportError as e:
        logger.error(f"Failed to import prediction model module: {module_name}")
        raise ValueError(f"Unknown model version: '{model_version}'. Error: {e}")
        
    if not hasattr(model_module, "create_model"):
        raise AttributeError(f"Module {module_name} does not have a 'create_model' function.")
        
    return model_module.create_model(parameters)
