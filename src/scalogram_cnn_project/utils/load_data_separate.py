import numpy as np
import cv2
import os
from collections import defaultdict
from scalogram_cnn_project.utils.list_files import list_files
import scalogram_cnn_project.settings.config as config
from pathlib import Path


import logging
logger = logging.getLogger(__name__)

def load_data(folder="GeneratedScalograms",
              channels=["C3", "C4"],
              cmap="viridis"):

    # -------------------------
    # Read file list
    # -------------------------
    all_files = list_files(folder)

    grouped = defaultdict(dict)
    labels = {}
    metadata = {}

    # -------------------------
    # Parse file names
    # -------------------------
    for file in all_files:
        
        ## drownsinessLevel0_subject1_session1_channelC3_epoch0.png
        splitted_filename = file.split("_")

        # Extract label
        label_part = splitted_filename[0]
        label = int(label_part.replace("drownsinessLevel", ""))

        # Extract metadata
        subject_part = splitted_filename[1]
        subject = subject_part.replace("subject", "")

        epoch_part = splitted_filename[4]
        epoch = epoch_part.replace("epoch", "").replace(".png", "")


        # Extract channel
        for ch in channels:
            if f"channel{ch}" in file:
                channel = ch
                break
        else:
            continue  # skip if channel not found

        # Build sample_id (remove channel part)
        sample_id = file.replace(f"_channel{channel}", "")

        grouped[sample_id][channel] = file  # dict inside dict
        labels[sample_id] = label
        metadata[sample_id] = {"subject": int(subject), "epoch": int(epoch) }

    # -------------------------
    # Build X, Y and Subject_list
    # -------------------------
    X_list = []
    Y_list = []
    Subject_list = []
    Epoch_list = []

    for sample_id in grouped:

        imgs = []

        for ch in channels:

            if ch not in grouped[sample_id]:
                break  # skip incomplete samples

            full_path = os.path.join(folder, grouped[sample_id][ch])

            if not os.path.exists(full_path):
                break
            
            if cmap=="viridis":
                img = cv2.imread(full_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif cmap=="gray":
                img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                img = img[..., np.newaxis]
            
            img = img / 255.0

            imgs.append(img)

        if len(imgs) == len(channels):
            stacked = np.concatenate(imgs, axis=-1)
            X_list.append(stacked)
            Y_list.append(labels[sample_id])
            Subject_list.append(metadata[sample_id]["subject"])
            Epoch_list.append(metadata[sample_id]["epoch"])

    X = np.stack(X_list)
    Y = np.array(Y_list)
    Y = Y[:, np.newaxis]
    Subject_array= np.array(Subject_list)
    Subject_array = Subject_array[:, np.newaxis]
    Epoch_array = np.array(Epoch_list)
    Epoch_array = Epoch_array[:, np.newaxis]


    logger.info("Dataset shape: %s", X.shape)
    logger.info("Labels shape: %s", Y.shape)
    logger.debug("Subject_array: %s", Subject_array)
    logger.debug("Epoch_array: %s", Epoch_array)

    return X, Y, Subject_array, Epoch_array




if __name__ == "__main__":

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s:%(name)s:%(message)s"
    )

    load_data(folder=config.DATA_DIR / "generated_scalograms_C3C4_gray_overlap_0.85",
              channels=["C3", "C4"],
              cmap="gray")