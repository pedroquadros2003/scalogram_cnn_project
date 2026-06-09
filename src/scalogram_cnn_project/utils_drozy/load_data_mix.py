from pathlib import Path
import numpy as np
import cv2
import os
from scalogram_cnn_project.utils.list_files import list_files
import scalogram_cnn_project.settings.config as config
import json


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
    index = {}


    # Load index dictionary that has all metadata for each scalogram
    with open(folder_path / "index.json") as f:
        index = json.load(f)

    # Check preprocessing from dataset_config.json
    dataset_config_path = folder_path / "dataset_config.json"
    is_juxtaposed = False

    logger.debug(f"Checking dataset_config at: {dataset_config_path}")
    if dataset_config_path.exists():
        with open(dataset_config_path) as f:
            d_config = json.load(f)
            logger.debug(f"dataset_config content: {d_config}")
            if d_config.get("preprocessing") == "rpca_juxtaposed":
                is_juxtaposed = True
    else:
        logger.debug("dataset_config.json not found!")

    logger.debug(f"is_juxtaposed flag evaluated to: {is_juxtaposed}")

    if is_juxtaposed and additional_features:
        logger.warning("rpca_juxtaposed is not compatible with additional_features. Setting additional_features to False.")
        additional_features = False


    # -------------------------
    # LOAD FEATURES IF NEEDED
    # -------------------------

    if additional_features:
        
        features_array = np.load(folder_path / "data.npy")
        # Expected shape:
        # (subject, session, channel, epoch, features)

        # Create subject → index mapping. +1 is for dummy subject
        subject_map = {s: i + 1 for i, s in enumerate(subjects)}

        # Channel mapping (must match generator)
        channel_map = {ch: i + 1 for i, ch in enumerate(channels)}


    # -------------------------
    # LOOP FOR SCALOGRAMS
    # -------------------------

    for image_id, meta in index.items():

        subject = meta["subject"]
        if subject not in subjects:
            logger.debug(f"Skipping image {image_id}: Subject {subject} not in requested subjects {subjects}")
            continue

        channel_str = meta["channel"]

        # Separate juxtaposed cases from the rest
        if is_juxtaposed:
            if channel_str != "merged":
                logger.debug(f"Skipping image {image_id}: Expected channel 'merged', but got '{channel_str}'")
                continue

        else:
            if channel_str not in channels:
                logger.debug(f"Skipping image {image_id}: Channel '{channel_str}' not in requested channels {channels}")
                continue
            if additional_features:
                ch_idx = channel_map[channel_str]

        file_name = f"{image_id}.png"
        full_path = os.path.join(folder_path, file_name)

        if not os.path.exists(full_path):
            logger.warning(f"{file_name} not found, skipping.")
            continue

        session = meta["session"]
        epoch = meta["epoch"]
        label = meta["label"]

        Subject_list.append(subject)
        Epoch_list.append(epoch)

        # Load image

        if cmap == "gray":

            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            img = np.expand_dims(img, axis=-1)

        else:

            img = cv2.imread(full_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        
        img = img / 255.0
        images.append(img)
        Y.append(label)

        # extra features
        if additional_features:
            subject_idx = subject_map[subject]

            extra_feat = features_array[
                subject_idx,
                session,
                ch_idx,
                epoch
            ]

            Extra_features_list.append(extra_feat)


    # -------------------------
    # FINAL FORMATTING
    # -------------------------
    X = np.array(images)

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

    X, Y, Subject_array, Epoch_array = load_data(folder_path=config.DATA_DIR / "generated_scalograms_ALL_gray_overlap0.733_subject1",
                                                channels=["C3", "C4"],
                                                cmap="gray",
                                                additional_features=True
                                                )


    train_test_split(X, Y, test_size=0.30, random_state=42,\
                    overlap=0.85, subject_array=Subject_array,\
                    epoch_array=Epoch_array, 
                    neglected_epochs_step = 2) ## It is equal to the number of channels in the mixed model
