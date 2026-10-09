import math
import numpy as np
import tensorflow as tf

class JointSequence(tf.keras.utils.Sequence):
    """
    Keras Sequence generator for multi-task Joint Training (Signal Forecasting + Drowsiness Classification).
    
    Generates batches with:
        X: Tensor of shape (batch_size, input_len, 1)
        y: Dictionary with dual targets:
            - "forecast_output": Tensor of shape (batch_size, forecast_steps) or (batch_size, 1)
            - "classification_output": Tensor of shape (batch_size, 1)
    """
    def __init__(
        self,
        loaded_signals,
        indices_with_dual_labels,
        input_len: int = 100,
        forecast_steps: int = 1,
        batch_size: int = 32,
        shuffle: bool = True
    ):
        """
        Args:
            loaded_signals (list): List of 1D numpy arrays (standardized signals).
            indices_with_dual_labels (list): List of tuples: (sig_idx, start_idx, y_forecast, y_clf).
            input_len (int): Number of time samples in past input window.
            forecast_steps (int): Number of future steps to predict.
            batch_size (int): Mini-batch size.
            shuffle (bool): Whether to shuffle sample order after each epoch.
        """
        self.loaded_signals = loaded_signals
        self.indices_with_dual_labels = list(indices_with_dual_labels)
        self.input_len = int(input_len)
        self.forecast_steps = int(forecast_steps)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        
        self.indexes = np.arange(len(self.indices_with_dual_labels))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return math.ceil(len(self.indices_with_dual_labels) / self.batch_size)

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        b_size = len(batch_indexes)
        
        batch_x = np.zeros((b_size, self.input_len, 1), dtype=np.float32)
        batch_y_forecast = np.zeros((b_size, self.forecast_steps), dtype=np.float32)
        batch_y_clf = np.zeros((b_size, 1), dtype=np.float32)
        
        for i, item_idx in enumerate(batch_indexes):
            sig_idx, start_idx, y_forecast, y_clf = self.indices_with_dual_labels[item_idx]
            sig = self.loaded_signals[sig_idx]
            
            # Extract input window
            batch_x[i, :, 0] = sig[start_idx : start_idx + self.input_len]
            
            # Set forecast target
            if np.isscalar(y_forecast) or (isinstance(y_forecast, (np.ndarray, list)) and len(y_forecast) == 1):
                batch_y_forecast[i, 0] = float(np.squeeze(y_forecast))
            else:
                batch_y_forecast[i, :] = np.asarray(y_forecast, dtype=np.float32)
                
            # Set classification target
            batch_y_clf[i, 0] = float(y_clf)
            
        return batch_x, {
            "forecast_output": batch_y_forecast,
            "classification_output": batch_y_clf
        }

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
