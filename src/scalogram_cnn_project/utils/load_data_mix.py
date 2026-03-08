from pathlib import Path
import numpy as np
import cv2
import os
from scalogram_cnn_project.utils.list_files import list_files
import scalogram_cnn_project.settings.config as config


def load_data(folder = "GeneratedScalograms",
              channels=["C3", "C4"],
              cmap="viridis"): 

    images = []
    Y = []

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
        

        full_path = os.path.join(folder, file)

        if not os.path.exists(full_path):
            print(f"Warning: {file} not found, skipping.")
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

    print("Final dataset shape:", X.shape)
    print("Labels shape:", Y.shape)

    return (X, Y)



if __name__ == "__main__":

    print("hello")
    load_data(folder=config.DATA_DIR / "generated_scalograms_C3C4_gray_overlap_0.85",
              channels=["C3", "C4"],
              cmap="gray")