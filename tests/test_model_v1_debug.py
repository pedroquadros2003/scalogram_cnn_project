import yaml
import sys
import json
from pathlib import Path

# Add src to pythonpath
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from scalogram_cnn_project.utils_seed_vig.load_data_separate import load_data

input_folder = PROJECT_ROOT / "data/seedvig_complete_scalograms"

# Let's inspect the dataset_config.json first
print("Checking dataset_config.json...")
config_path = input_folder / "dataset_config.json"
if config_path.exists():
    with open(config_path) as f:
        print(json.load(f))
else:
    print("dataset_config.json NOT found!")

# Let's inspect index.json presence
print("Checking index.json...")
index_path = input_folder / "index.json"
print("index.json exists:", index_path.exists())

try:
    print("Loading data for channel FT7...")
    X, y, Subject_array, Epoch_array = load_data(
        folder_path=input_folder,
        channels=['FT7'],
        cmap='gray'
    )
    print("Data loaded successfully!")
    print("X shape:", X[0].shape if isinstance(X, list) else X.shape)
    print("y shape:", y.shape)
except Exception as e:
    import traceback
    print("Error during data loading:")
    traceback.print_exc()
