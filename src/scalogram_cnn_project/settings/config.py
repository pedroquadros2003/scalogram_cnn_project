from pathlib import Path
import platform
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent

DATA_DIR = PROJECT_DIR / "data"

OUTPUT_DIR = PROJECT_DIR / "outputs"

# Linux path mapping
DATASET_DIR = Path("/mnt/c/Users/peuqu/OneDrive/Desktop/IC Harlei-Sarah/DataSets")

DROZY_DIR = DATASET_DIR / "DROZY" / "DROZY"

drozy_kss_scale = np.array([
    [-1, -1, -1, -1],
    [-1, 3, 6, 7], ## subject 1
    [-1, 3, 7, 6], ## subject 2 
    [-1, 2, 3, 4], ## subject 3
    [-1, 4, 8, 9], ## subject 4
    [-1, 3, 7, 8], ## subject 5
    [-1, 2, 3, 7], ## subject 6
    [-1, 0, 4, 9], ## subject 7
    [-1, 2, 6, 8], ## subject 8
    [-1, 2, 6, 8], ## subject 9
    [-1, 3, 6, 7], ## subject 10
    [-1, 4, 7, 7], ## subject 11
    [-1, 2, 5, 6], ## subject 12
    [-1, 6, 3, 7], ## subject 13
    [-1, 5, 7, 8]  ## subject 14
])

drozy_valid_tests = np.array([
    [-1, -1, -1, -1],
    [-1, 1, 1, 1], ## subject 1
    [-1, 1, 1, 1], ## subject 2
    [-1, 1, 1, 1], ## subject 3
    [-1, 1, 1, 1], ## subject 4
    [-1, 1, 1, 1], ## subject 5
    [-1, 1, 1, 1], ## subject 6
    [-1, 0, 1, 1], ## subject 7
    [-1, 1, 1, 1], ## subject 8
    [-1, 0, 1, 1], ## subject 9
    [-1, 1, 0, 1], ## subject 10
    [-1, 1, 1, 1], ## subject 11
    [-1, 1, 0, 0], ## subject 12
    [-1, 1, 1, 0], ## subject 13
    [-1, 1, 1, 1]  ## subject 14
])


ITA_PILOT_DIR = DATASET_DIR / "ITA_PILOT"


if __name__ == "__main__":

    print(DATA_DIR)

    print(drozy_kss_scale[13][1])
    print(drozy_valid_tests[13][1])