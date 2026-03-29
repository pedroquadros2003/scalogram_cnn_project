import numpy as np
import cv2
import os
from collections import defaultdict
from scalogram_cnn_project.utils.list_files import list_files
import scalogram_cnn_project.settings.config as config
from pathlib import Path


import logging
logger = logging.getLogger(__name__)

def load_data(folder_path="GeneratedScalograms",
              channels=["C3", "C4"],
              cmap="viridis",
              subjects=range(1, 15),
              additional_features=False):

    # -------------------------
    # READ FILE LIST
    # -------------------------
    all_files = list_files(folder_path)

    grouped = defaultdict(dict)
    labels = {}
    metadata = {}

    # -------------------------
    # LOAD FEATURES IF NEEDED
    # -------------------------
    if additional_features:
        features_array = np.load(folder_path / "data.npy")
        # shape: (subject, session, channel, epoch, features)

        # Create subject → index mapping. +1 is for dummy subject
        subject_map = {s: i + 1 for i, s in enumerate(subjects)}

        # Channel mapping (must match generator)
        channel_map = {ch: i + 1 for i, ch in enumerate(channels)}


    # -------------------------
    # PARSE FILENAMES
    # -------------------------
    for file in all_files:


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

        # Example:
        # drownsinessLevel0_subject1_session1_channelC3_epoch0.png

        label   = int(splitted[0].replace("drownsinessLevel", ""))
        subject = int(splitted[1].replace("subject", ""))
        session = int(splitted[2].replace("session", ""))
        epoch   = int(splitted[4].replace("epoch", "").replace(".png", ""))

        # Extract channel
        channel = None
        for ch in channels:
            if f"channel{ch}" in file:
                channel = ch
                break

        if channel is None:
            continue

        # Build sample_id (same sample across channels)
        sample_id = file.replace(f"_channel{channel}", "")

        grouped[sample_id][channel] = file
        labels[sample_id] = label
        metadata[sample_id] = {
            "subject": subject,
            "session": session,
            "epoch": epoch
        }

    # -------------------------
    # BUILD DATASET
    # -------------------------
    X_list = []
    Y_list = []
    Subject_list = []
    Epoch_list = []
    Extra_features_list = []

    for sample_id in grouped:

        imgs = []
        extra_feats_per_sample = []

        for ch in channels:

            if ch not in grouped[sample_id]:
                break  # skip incomplete sample

            file = grouped[sample_id][ch]
            full_path = os.path.join(folder_path, file)

            if not os.path.exists(full_path):
                break

            # -------------------------
            # LOAD IMAGE
            # -------------------------
            if cmap == "viridis":
                img = cv2.imread(full_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                img = img[..., np.newaxis]

            img = img / 255.0
            imgs.append(img)

            # -------------------------
            # LOAD FEATURES PER CHANNEL
            # -------------------------
            if additional_features:
                subject = metadata[sample_id]["subject"]
                session = metadata[sample_id]["session"]
                epoch = metadata[sample_id]["epoch"]

                if ch not in channel_map:
                    raise ValueError(f"Channel {ch} not found in channel_map")

                ch_idx = channel_map[ch]
                subject_idx = subject_map[subject]

                feat = features_array[
                    subject_idx,
                    session,
                    ch_idx,
                    epoch
                ]

                extra_feats_per_sample.append(feat)

        # Only keep complete samples
        if len(imgs) == len(channels):

            # Stack images along channel dimension
            stacked = np.concatenate(imgs, axis=-1)
            X_list.append(stacked)

            Y_list.append(labels[sample_id])
            Subject_list.append(metadata[sample_id]["subject"])
            Epoch_list.append(metadata[sample_id]["epoch"])

            # Combine features from all channels
            if additional_features:
                combined_feat = np.concatenate(extra_feats_per_sample, axis=0)
                Extra_features_list.append(combined_feat)

    # -------------------------
    # FINAL FORMATTING
    # -------------------------
    X = np.stack(X_list)
    Y = np.array(Y_list)[:, np.newaxis]
    Subject_array = np.array(Subject_list)
    Epoch_array = np.array(Epoch_list)

    if additional_features:
        X_extra = np.array(Extra_features_list)

        # Multi-input format for Keras
        X = [X, X_extra]

        logger.info("Image tensor shape: %s", X[0].shape)
        logger.info("Extra features shape: %s", X[1].shape)
        logger.info("Labels shape: %s", Y.shape)

        return X, Y, Subject_array, Epoch_array

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
                                                additional_features=True)
    

    train_test_split(X, Y, test_size=0.30, random_state=42,\
                overlap=0.85, subject_array=Subject_array,\
                epoch_array=Epoch_array, 
                neglected_epochs_step = 2) ## It is equal to the number of channels in the mixed model