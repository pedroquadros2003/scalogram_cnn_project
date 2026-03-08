from pathlib import Path
import platform
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent

DATA_DIR = PROJECT_DIR / "data"

OUTPUT_DIR = PROJECT_DIR / "outputs"

# Linux path mapping
DATASET_DIR = Path("/mnt/c/Users/peuqu/OneDrive/Desktop/IC Harlei/DataSets")

DROZY_DIR = DATASET_DIR / "DROZY" / "DROZY"

drozy_kss_scale = np.array([
    [-1, -1, -1, -1],
    [-1, 3, 6, 7], 
    [-1, 3, 7, 6], 
    [-1, 2, 3, 4], 
    [-1, 4, 8, 9], 
    [-1, 3, 7, 8], 
    [-1, 2, 3, 7], 
    [-1, 0, 4, 9], 
    [-1, 2, 6, 8], 
    [-1, 2, 6, 8], 
    [-1, 3, 6, 7], 
    [-1, 4, 7, 7], 
    [-1, 2, 5, 6], 
    [-1, 6, 3, 7], 
    [-1, 5, 7, 8]  
])

drozy_valid_tests = np.array([
    [-1, -1, -1, -1],
    [-1, 1, 1, 1], 
    [-1, 1, 1, 1],
    [-1, 1, 1, 1],
    [-1, 1, 1, 1],
    [-1, 1, 1, 1],
    [-1, 1, 1, 1],
    [-1, 0, 1, 1],
    [-1, 1, 1, 1],
    [-1, 0, 1, 1],
    [-1, 1, 0, 1],
    [-1, 1, 1, 1],
    [-1, 1, 0, 0],
    [-1, 1, 1, 0],
    [-1, 1, 1, 1]
])


ITA_PILOT_DIR = DATASET_DIR / "ITA_PILOT"


if __name__ == "__main__":

    print(DATA_DIR)

    print(drozy_kss_scale[13][1])
    print(drozy_valid_tests[13][1])