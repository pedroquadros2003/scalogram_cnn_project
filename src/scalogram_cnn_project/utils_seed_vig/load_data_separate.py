import numpy as np
import cv2
import os
from collections import defaultdict
import scalogram_cnn_project.settings.config as config
from pathlib import Path
import json

import logging
logger = logging.getLogger(__name__)


def load_data(
    folder_path="seedvig_scalograms",
    channels=["FT8"],
    cmap="viridis",
    subjects=[1, 3],
    additional_features=False
):

    index = {}

    # -------------------------
    # LOAD INDEX
    # -------------------------

    with open(folder_path / "index.json") as f:
        index = json.load(f)

    grouped = defaultdict(dict)

    labels = {}

    metadata = {}

    # -------------------------
    # LOAD FEATURES IF NEEDED
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

        subject = meta["subject"]

        epoch = meta["epoch"]

        channel = meta["channel"]

        label = meta["label"]

        # filters

        if subject not in subjects:
            continue

        if channel not in channels:
            continue

        # same epoch across channels
        sample_id = (subject, epoch)

        file_name = f"{image_id}.png"

        grouped[sample_id][channel] = file_name

        labels[sample_id] = label

        metadata[sample_id] = {
            "subject": subject,
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

            # skip incomplete sample
            if ch not in grouped[sample_id]:
                break

            file = grouped[sample_id][ch]

            full_path = os.path.join(folder_path, file)

            if not os.path.exists(full_path):
                break

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
            imgs.append(img)

            # -------------------------
            # LOAD EXTRA FEATURES
            # -------------------------

            if additional_features:

                subject = metadata[sample_id]["subject"]

                epoch = metadata[sample_id]["epoch"]

                subject_idx = subject_map[subject]

                ch_idx = channel_map[ch]

                feat = features_array[
                    subject_idx,
                    ch_idx,
                    epoch
                ]

                extra_feats_per_sample.append(feat)

        # -------------------------
        # KEEP COMPLETE SAMPLES
        # -------------------------

        if len(imgs) == len(channels):

            # concatenate channels in last dimension
            stacked = np.concatenate(imgs, axis=-1)

            X_list.append(stacked)

            Y_list.append(labels[sample_id])

            Subject_list.append(
                metadata[sample_id]["subject"]
            )

            Epoch_list.append(
                metadata[sample_id]["epoch"]
            )

            if additional_features:

                combined_feat = np.concatenate(
                    extra_feats_per_sample,
                    axis=0
                )

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

        X = [X, X_extra]

        logger.info(
            "Image tensor shape: %s",
            X[0].shape
        )

        logger.info(
            "Extra features shape: %s",
            X[1].shape
        )

        logger.info(
            "Labels shape: %s",
            Y.shape
        )

        return X, Y, Subject_array, Epoch_array

    logger.info("Image tensor shape: %s", X.shape)

    logger.info("Labels shape: %s", Y.shape)

    return X, Y, Subject_array, Epoch_array



if __name__ == "__main__":

    import logging

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s:%(name)s:%(message)s"
    )

    X, Y, Subject_array, Epoch_array = load_data(
        folder_path=config.DATA_DIR / "seedvig_scalograms",
        channels=["FT8", "T7"],
        cmap="gray",
        subjects=[1, 3],
        additional_features=True
    )

    print(X[0].shape)

    print(X[1].shape)

    print(Y.shape)