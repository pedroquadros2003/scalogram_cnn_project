from pathlib import Path
import platform
import numpy as np
import os
# Helper to load environmental variables from a local .env file
def _load_dotenv(dotenv_path):
    if dotenv_path.exists():
        with open(dotenv_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if key not in os.environ:
                    os.environ[key] = val

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent
_load_dotenv(PROJECT_DIR / ".env")

DATA_DIR = PROJECT_DIR / "data"

OUTPUT_DIR = PROJECT_DIR / "outputs"

PARAM_SEARCH_DIR = PROJECT_DIR / "configs" / "hyperparameter_search"

# Resolve Dataset directories from environment variables
def _get_path_env(env_name):
    path_str = os.getenv(env_name)
    if not path_str:
        raise ValueError(
            f"{env_name} is not set. Please define it in your '.env' file in the project root "
            f"(see '.env_example' for reference)."
        )
    return Path(path_str)


################ DROZY DATASET ################

DROZY_DIR = _get_path_env("DROZY_DIR")

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

################ ITA PILOT DATASET ################

ITA_PILOT_DIR = _get_path_env("ITA_PILOT_DIR")

################ SEED_VIG DATASET ################

SEED_VIG_DIR = _get_path_env("SEED_VIG_DIR")

SEED_VIG_LABELS = _get_path_env("SEED_VIG_LABELS")

seed_vig_filenames = [
    "null",
    "10_20151125_noon.mat",
    "11_20151024_night.mat",
    "12_20150928_noon.mat",
    "13_20150929_noon.mat",
    "14_20151014_night.mat",
    "15_20151126_night.mat",
    "16_20151128_night.mat",
    "17_20150925_noon.mat",
    "18_20150926_noon.mat",
    "19_20151114_noon.mat",
    "1_20151124_noon_2.mat",
    "20_20151129_night.mat",
    "21_20151016_noon.mat",
    "2_20151106_noon.mat",
    "3_20151024_noon.mat",
    "4_20151105_noon.mat",
    "4_20151107_noon.mat",
    "5_20141108_noon.mat",
    "5_20151012_night.mat",
    "6_20151121_noon.mat",
    "7_20151015_night.mat",
    "8_20151022_noon.mat",
    "9_20151017_night.mat",
]


if __name__ == "__main__":

    print(DATA_DIR)

    print(drozy_kss_scale[13][1])
    print(drozy_valid_tests[13][1])