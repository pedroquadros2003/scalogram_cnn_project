
import numpy as np
from scalogram_cnn_project.utils.balance_indices_undersampling import balanced_indices_undersmp

import logging
logger = logging.getLogger(__name__)

def train_test_split(X,
                     y,
                     random_state,
                     subject_array,
                     loso_subject):
    
    test_indexes  = np.where(subject_array == loso_subject)[0]

    train_indexes = np.where(subject_array != loso_subject)[0]
    
    x_train = X[ train_indexes, ...]
    x_test  = X[ test_indexes , ...]
    y_train = y[ train_indexes, ...]
    y_test  = y[ test_indexes , ...]

    indices = balanced_indices_undersmp(y_train, random_state)
    x_train = x_train[indices]
    y_train = y_train[indices]


    logger.info("Shapes:")
    logger.info("x_train : %s", x_train.shape)
    logger.info("x_test  : %s", x_test.shape)
    logger.info("y_train : %s", y_train.shape)
    logger.info("y_test  : %s", y_test.shape)


    logger.debug("The number of 0s in the training set is: %d", len(np.where( y_train == 0 )[0]))
    logger.debug("The number of 1s in the training set is: %d", len(np.where( y_train == 1 )[0]))

    return x_train, x_test, y_train, y_test 