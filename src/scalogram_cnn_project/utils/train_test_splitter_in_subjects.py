
import numpy as np
from imblearn.under_sampling import RandomUnderSampler

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

    rus = RandomUnderSampler(random_state=random_state)

    rus.fit_resample(np.zeros(len(y_train)).reshape(-1,1), y_train)
    indices = rus.sample_indices_
    x_train = x_train[indices]
    y_train = y_train[indices]


    print("Shapes:")
    print(f"x_train : {x_train.shape}")
    print(f"x_test  : {x_test.shape}")
    print(f"y_train : {y_train.shape}")
    print(f"y_test  : {y_test.shape}")


    print("For debug:\n")
    print(len(np.where( y_train == 0 )[0]))
    print(len(np.where( y_train == 1 )[0]))

    return x_train, x_test, y_train, y_test 