
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from math import ceil

import logging
logger = logging.getLogger(__name__)

def train_test_split(X, y, test_size, random_state, overlap):
    
    threshold_epochs = int(X.shape[0] * (1 - test_size))
    neglected_epochs =  ceil( 1/ (1-overlap) )
    
    x_train = X[ :threshold_epochs, ...]
    x_test  = X[ threshold_epochs + neglected_epochs: , ...]
    y_train = y[ :threshold_epochs, ...]
    y_test  = y[ threshold_epochs + neglected_epochs: , ...]

    rus = RandomUnderSampler(random_state=random_state)

    rus.fit_resample(np.zeros(len(y_test)).reshape(-1,1), y_test)
    indices = rus.sample_indices_
    x_test = x_test[indices]
    y_test = y_test[indices]

    rus.fit_resample(np.zeros(len(y_train)).reshape(-1,1), y_train)
    indices = rus.sample_indices_
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