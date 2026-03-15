import numpy as np
from imblearn.under_sampling import RandomUnderSampler

import logging
logger = logging.getLogger(__name__)


## This function is responsible for balancing the number of labels in the dataset. It undersamples the dominant
## class until the dataset becomes balanced
def balanced_indices_undersmp(y, random_state):

    ## This is for the case when, possibly, all the subject's labels are equal to 0, or
    ## all of them are equal to 1.
    if len(np.unique(y)) < 2:  
        return np.arange(len(y))

    rus = RandomUnderSampler(random_state=random_state)

    rus.fit_resample(
        np.arange(len(y)).reshape(-1,1),
        y
    )

    return rus.sample_indices_
