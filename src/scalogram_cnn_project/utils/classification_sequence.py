import numpy as np
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

class ClassificationSequence(tf.keras.utils.Sequence):
    """
    Keras Sequence providing signal inputs (past window) and target binary
    classification labels (alert/drowsy) for training a coupled classification model.
    """
    def __init__(self, loaded_signals, indices_with_labels, input_len, batch_size, shuffle=True, **kwargs):
        """
        Args:
            loaded_signals (list): List of 1D normalized signal arrays (one per file).
            indices_with_labels (list): List of tuples: (sig_idx, start_idx, label).
            input_len (int): Duration of the input signal window in samples.
            batch_size (int): Batch size.
            shuffle (bool): Shuffles batch indices on epoch end.
        """
        super().__init__(**kwargs)
        self.loaded_signals = loaded_signals
        self.indices = list(indices_with_labels)
        self.input_len = input_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))
        
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        X_batch = []
        y_batch = []
        
        for sig_idx, start_idx, label in batch_indices:
            signal = self.loaded_signals[sig_idx]
            X_batch.append(signal[start_idx : start_idx + self.input_len])
            y_batch.append(label)
            
        X_batch = np.array(X_batch, dtype=np.float32)
        y_batch = np.array(y_batch, dtype=np.float32)
        
        # Reshape inputs to (batch, input_steps, features=1)
        X_batch = np.expand_dims(X_batch, axis=-1)
        # Reshape targets to (batch, 1)
        y_batch = np.expand_dims(y_batch, axis=-1)
        
        return X_batch, y_batch
        
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
