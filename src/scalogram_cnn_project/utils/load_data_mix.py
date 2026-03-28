from pathlib import Path
import numpy as np
import cv2
import os
from scalogram_cnn_project.utils.list_files import list_files
import scalogram_cnn_project.settings.config as config

import logging
logger = logging.getLogger(__name__)

def load_data(folder_path="GeneratedScalograms",
              channels=["C3", "C4"],
              cmap="viridis",
              subjects = range(1,15),
              additional_features=False
              ):

    images = []
    Y = []
    Epoch_list = []
    Subject_list = []
    Extra_features_list = []

    # Get list of files
    selected_files = list_files(folder_path)

    # Load feature array if needed
    if additional_features:
        
        features_array = np.load(folder_path / "data.npy")
        # Expected shape:
        # (subject, session, channel, epoch, features)

        # Create channel → index mapping (must match generator logic)
        # +1 because index 0 is reserved as dummy channel
        channel_map = {ch: i + 1 for i, ch in enumerate(channels)}

    for file in selected_files:


        # Check if file belongs to selected channels
        if not any(f"channel{ch}" in file for ch in channels):
            continue

        # Check if file belongs to selected subjects
        if not any(f"subject{sub}" in file for sub in subjects):
            continue

        # Ensure file is an image
        if not (".jpg" in file or ".png" in file):
            continue


        splitted = file.split("_")

        # -------------------------
        # EXTRACT METADATA
        # -------------------------
        # Example filename:
        # drownsinessLevel0_subject1_session1_channelC3_epoch0.png

        subject = int(splitted[1].replace("subject", ""))
        session = int(splitted[2].replace("session", ""))
        channel_str = splitted[3].replace("channel", "")
        epoch = int(splitted[4].replace("epoch", "").replace(".png", ""))

        Subject_list.append(subject)
        Epoch_list.append(epoch)

        full_path = os.path.join(folder_path, file)

        if not os.path.exists(full_path):
            logger.warning(f"{file} not found, skipping.")
            continue

        # -------------------------
        # LOAD IMAGE
        # -------------------------
        if cmap == "viridis":
            img = cv2.imread(full_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

        images.append(img)

        # -------------------------
        # EXTRACT LABEL
        # -------------------------
        # Assumes label is encoded at fixed position in filename
        Y.append(int(file[16]))

        # -------------------------
        # LOAD ADDITIONAL FEATURES
        # -------------------------
        if additional_features:

            if channel_str not in channel_map:
                raise ValueError(f"Channel {channel_str} not found in channel_map")

            ch_idx = channel_map[channel_str]

            # Extract feature vector aligned with this image
            extra_feat = features_array[
                subject,
                session,
                ch_idx,
                epoch
            ]

            Extra_features_list.append(extra_feat)

    # -------------------------
    # FINAL FORMATTING
    # -------------------------
    X = np.array(images) / 255.0

    if cmap == "gray":
        X = X[..., np.newaxis]

    Y = np.array(Y)[:, np.newaxis]

    Epoch_array = np.array(Epoch_list)
    Subject_array = np.array(Subject_list)

    if additional_features:
        X_extra = np.array(Extra_features_list)

        # Combine inputs into a list for Keras multi-input model
        X = [X, X_extra]

        logger.info("Image tensor shape: %s", X[0].shape)
        logger.info("Extra features shape: %s", X[1].shape)
        logger.info("Labels shape: %s", Y.shape)

        return X, Y, Subject_array, Epoch_array
    
    else:
        logger.info("Image tensor shape: %s", X.shape)
        logger.info("Labels shape: %s", Y.shape)

        return X, Y, Subject_array, Epoch_array


if __name__ == "__main__":

    import scalogram_cnn_project.settings.config as config
    from scalogram_cnn_project.utils.train_test_splitter_in_time import train_test_split

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s:%(name)s:%(message)s"
    )

    X, Y, Subject_array, Epoch_array = load_data(folder_path=config.DATA_DIR / "generated_scalograms_ALL_gray_overlap0.733_s1",
                                                channels=["C3", "C4"],
                                                cmap="gray",
                                                additional_features=True
                                                )


    train_test_split(X, Y, test_size=0.30, random_state=42,\
                    overlap=0.85, subject_array=Subject_array,\
                    epoch_array=Epoch_array, 
                    neglected_epochs_step = 2) ## It is equal to the number of channels in the mixed model
