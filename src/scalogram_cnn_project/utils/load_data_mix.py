from pathlib import Path
import numpy as np
import cv2
import os
from scalogram_cnn_project.utils.list_files import list_files
import scalogram_cnn_project.settings.config as config

import logging
logger = logging.getLogger(__name__)

def load_data(folder = "GeneratedScalograms",
              channels=["C3", "C4"],
              cmap="viridis"): 

    images = []
    Y = []
    Epoch_list = []
    Subject_list = []


    # Read file list
    selected_files = list_files(folder)


    for file in selected_files:

        # Check if the scalogram comes from one of the selected channels 
        include_file = False
        for ch in channels:
            if f"channel{ch}" in file:
                include_file = True
                break
            else:
                continue

        # Skip file if it should not be included
        if not include_file:
            continue
        
        ## drownsinessLevel0_subject1_session1_channelC3_epoch0.png
        splitted_filename = file.split("_")

        # Extract metadata
        subject_part = splitted_filename[1]
        subject = subject_part.replace("subject", "")
        Subject_list.append(int(subject))

        epoch_part = splitted_filename[4]
        epoch = epoch_part.replace("epoch", "").replace(".png", "")
        Epoch_list.append(int(epoch))

        full_path = os.path.join(folder, file)

        if not os.path.exists(full_path):
            logger.warning(f"Warning: {file} not found, skipping.")
            continue

        if cmap=="viridis":
            img = cv2.imread(full_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif cmap=="gray":
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)


        images.append(img)
        ## 16 is the exact position of the label (0 awake, 1 drowsy)
        Y.append(int(file[16])) 

    X = np.array(images)
    X = X / 255.0

    if cmap=="gray":
        X = X[..., np.newaxis]

    Y = np.array(Y)
    Y = Y[:, np.newaxis]

    Epoch_array = np.array(Epoch_list)
    Subject_array = np.array(Subject_list)

    logger.debug("Subject_array: %s", Subject_array)
    logger.debug("Epoch_array: %s", Epoch_array)
    logger.info("Dataset shape: %s", X.shape)
    logger.info("Labels shape: %s", Y.shape)

    return X, Y, Subject_array, Epoch_array



if __name__ == "__main__":

    import scalogram_cnn_project.settings.config as config
    from scalogram_cnn_project.utils.train_test_splitter_in_time import train_test_split

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s:%(name)s:%(message)s"
    )

    X, Y, Subject_array, Epoch_array = load_data(folder=config.DATA_DIR / "generated_scalograms_C3C4_gray_overlap_0.85",
                                                channels=["C3", "C4"],
                                                cmap="gray")


    train_test_split(X, Y, test_size=0.30, random_state=42,\
                    overlap=0.85, subject_array=Subject_array,\
                    epoch_array=Epoch_array, 
                    neglected_epochs_step = 2) ## It is equal to the number of channels in the mixed model
