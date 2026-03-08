from scalogram_cnn_project.epoch_rejection.generate_epoch_rejection_object import generate_epoch_object
import scalogram_cnn_project.settings.config as config
from pathlib import Path
from autoreject import AutoReject, get_rejection_threshold


import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("scalogram_cnn_project").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)



if __name__ == "__main__": 

    epochs = generate_epoch_object(subject=2, session=2, verbose=False)

    ar = AutoReject()
    epochs_clean = ar.fit_transform(epochs)

    reject = get_rejection_threshold(epochs)  
    reject_log = ar.get_reject_log(epochs)
    reject_log.plot()

    data_cleaned = epochs_clean.get_data()
    logger.info(data_cleaned.shape)