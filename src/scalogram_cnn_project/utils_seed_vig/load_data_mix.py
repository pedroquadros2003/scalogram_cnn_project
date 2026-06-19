from pathlib import Path
import numpy as np
import cv2
import os
import json

import scalogram_cnn_project.settings.config as config

import logging
logger = logging.getLogger(__name__)


def load_data(
    folder_path="GeneratedScalograms",
    channels=["FT8"],
    cmap="viridis",
    subjects=range(1, 24),
    additional_features=False
):

    images = []
    Y = []
    Epoch_list = []
    Subject_list = []
    Extra_features_list = []
    index = {}

    # -------------------------
    # LOAD INDEX
    # -------------------------

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
    # LOAD EXTRA FEATURES
    # -------------------------

    if additional_features:

        features_array = np.load(folder_path / "data.npy")

        # shape:
        # (subject, channel, epoch, features)

        subject_map = {
            s: i + 1 for i, s in enumerate(subjects)
        }

        channel_map = {
            ch: i + 1 for i, ch in enumerate(channels)
        }

    # -------------------------
    # LOOP OVER SCALOGRAMS
    # -------------------------

    for image_id, meta in index.items():

        # filter subjects
        if meta["subject"] not in subjects:
            continue

        channel_str = meta["channel"]

        # Separate juxtaposed cases from the rest
        if is_juxtaposed:
            if channel_str != "merged":
                continue
        else:
            if channel_str not in channels:
                continue
            if additional_features:
                ch_idx = channel_map[channel_str]

        file_name = f"{image_id}.png"
        full_path = os.path.join(folder_path, file_name)

        if not os.path.exists(full_path):

            logger.warning(f"{file_name} not found, skipping.")
            continue

        subject = meta["subject"]
        epoch = meta["epoch"]
        label = meta["label"]

        Subject_list.append(subject)
        Epoch_list.append(epoch)

        # -------------------------
        # LOAD IMAGE
        # -------------------------

        if cmap == "gray":

            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            img = np.expand_dims(img, axis=-1)

        else:

            img = cv2.imread(full_path)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        img = img / 255.0
        images.append(img)

        Y.append(label)

        # -------------------------
        # LOAD EXTRA FEATURES
        # -------------------------

        if additional_features:

            subject_idx = subject_map[subject]

            extra_feat = features_array[
                subject_idx,
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

    # -------------------------
    # MULTI INPUT
    # -------------------------

    if additional_features:

        X_extra = np.array(Extra_features_list)

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

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s:%(name)s:%(message)s"
    )

    X, Y, Subject_array, Epoch_array = load_data(
        folder_path=config.DATA_DIR / "seedvig_scalograms",
        channels=["FT8", "T7"],
        subjects=[1,3],
        cmap="gray",
        additional_features=True
    )

    print(X[0].shape)

    print(X[1].shape)

    print(Y.shape)