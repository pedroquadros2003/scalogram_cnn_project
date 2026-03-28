
import numpy as np

import logging
logger = logging.getLogger(__name__)

def index_X(X, indices):
    """
    Apply indexing to X, whether it is a numpy array or a list of arrays.
    """
    if isinstance(X, list):
        return [x[indices, ...] for x in X]
    return X[indices, ...]


def sort_X(X, indices):
    """
    Apply sorting to X.
    """
    return index_X(X, indices)


def slice_X(X, start, end=None):
    """
    Slice X safely (supports list or array).
    """
    if isinstance(X, list):
        return [x[start:end, ...] for x in X]
    return X[start:end, ...]



def concat_X(X_list):
    """
    Concatenate list of Xs (supports list-of-arrays structure).
    """
    if isinstance(X_list[0], list):
        # multi-input case
        return [
            np.concatenate([x[i] for x in X_list], axis=0)
            for i in range(len(X_list[0]))
        ]
    else:
        return np.concatenate(X_list, axis=0)
    

def get_num_epochs(X):
    """
    Returns the number of samples (epochs) in X.

    Supports:
    - numpy array: shape (N, ...)
    - list of numpy arrays: [X1, X2, ...] where each has shape (N, ...)
    """
    if isinstance(X, list):
        if len(X) == 0:
            raise ValueError("X list is empty")
        
        n = X[0].shape[0]

        # Safety check: ensure all inputs have same number of samples
        if not all(x.shape[0] == n for x in X):
            raise ValueError("All elements in X must have the same number of samples")

        return n

    return X.shape[0]