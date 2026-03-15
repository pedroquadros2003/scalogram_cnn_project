
import numpy as np
from scalogram_cnn_project.utils.balance_indices_undersampling import balanced_indices_undersmp
from math import ceil

import logging
logger = logging.getLogger(__name__)



def train_test_split_aux(X, y, test_size, random_state, overlap, neglected_epochs_step):
    '''
    This function considers a simplification of our problem. It takes for granted that X and y from one 
    subject only, and that both arrays are ordered according to the epochs
    '''
    
    threshold_epochs = int(X.shape[0] * (1 - test_size))

    # Number of epochs to discard to avoid temporal leakage caused by overlapping windows
    # Example: overlap=0.85 -> discard ~7 epochs
    neglected_epochs =  ceil( 1/ (1-overlap) * neglected_epochs_step)
    
    x_train = X[ :threshold_epochs, ...]
    x_test  = X[ threshold_epochs + neglected_epochs: , ...]
    y_train = y[ :threshold_epochs, ...]
    y_test  = y[ threshold_epochs + neglected_epochs: , ...]

    indices = balanced_indices_undersmp(y_train, random_state)
    x_train = x_train[indices]
    y_train = y_train[indices]

    logger.debug("Subject split shapes:")
    logger.debug("x_train : %s", x_train.shape)
    logger.debug("x_test  : %s", x_test.shape)
    logger.debug("y_train : %s", y_train.shape)
    logger.debug("y_test  : %s", y_test.shape)


    logger.debug("The number of 0s in the training set for a specific subject is: %d", np.sum(y_train == 0))
    logger.debug("The number of 1s in the training set for a specific subject is: %d", np.sum(y_train == 1))

    return x_train, x_test, y_train, y_test 


def train_test_split(X, y, test_size, random_state, overlap, subject_array, epoch_array, neglected_epochs_step=1):
    '''
    This function solves the problem of splitting. In order to the separate the last few minutes of a recording for 
    testing, it creates a pair of temporary variables, X_temp and y_temp, one pair per subject, ordering them according
    to the epoch numbering, and then applies the auxiliary splitting method. 
    '''


    subject_label_array = np.unique(subject_array)

    # List accumulators
    x_train_list = []
    x_test_list = []
    y_train_list = []
    y_test_list = []

    for subject in subject_label_array:

        subject_indexes = np.where(subject_array == subject)[0]

        X_temp = X[subject_indexes, ...]
        y_temp = y[subject_indexes, ...]
        epoch_array_temp = epoch_array[subject_indexes, ...]

        indexes_sorted = np.argsort(epoch_array_temp.ravel())

        X_temp = X_temp[indexes_sorted, ...]
        y_temp = y_temp[indexes_sorted, ...]

        temp_x_train, temp_x_test, temp_y_train, temp_y_test = \
            train_test_split_aux(X_temp, y_temp, test_size, random_state, overlap, neglected_epochs_step)

        # Accumulate
        x_train_list.append(temp_x_train)
        x_test_list.append(temp_x_test)
        y_train_list.append(temp_y_train)
        y_test_list.append(temp_y_test)

    # Concatenate everything
    x_train = np.concatenate(x_train_list, axis=0)
    x_test  = np.concatenate(x_test_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    y_test  = np.concatenate(y_test_list, axis=0)

    logger.info("Shapes:")
    logger.info("x_train : %s", x_train.shape)
    logger.info("x_test  : %s", x_test.shape)
    logger.info("y_train : %s", y_train.shape)
    logger.info("y_test  : %s", y_test.shape)

    logger.debug("The number of 0s in the training set is: %d", np.sum(y_train == 0))
    logger.debug("The number of 1s in the training set is: %d", np.sum(y_train == 1))

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":

    from scalogram_cnn_project.utils.load_data_separate import load_data
    import scalogram_cnn_project.settings.config as config

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s:%(name)s:%(message)s"
    )

    X, Y, Subject_array, Epoch_array =load_data(folder=config.DATA_DIR / "generated_scalograms_C3C4_gray_overlap_0.85",
                                                channels=["C3", "C4"],
                                                cmap="gray")


    train_test_split(X, Y, test_size=0.30, random_state=42, overlap=0.85, subject_array=Subject_array, epoch_array=Epoch_array)