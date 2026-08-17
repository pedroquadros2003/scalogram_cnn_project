import sys
import os
import numpy as np
from pathlib import Path

# Add src to pythonpath
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

# Load dotenv to get dataset path
from scalogram_cnn_project.settings import config
from scalogram_cnn_project.utils.signal_loader import SignalLoader

print("SEED_VIG_DIR:", config.SEED_VIG_DIR)
files = list(config.SEED_VIG_DIR.glob("*.mat"))
if files:
    print("Found files:", len(files))
    test_file = files[0]
    print("Loading test file:", test_file.name)
    signal_data = SignalLoader.load_signal(str(test_file), "seed_vig")
    
    # Let's inspect some channel stats
    channels_to_check = ["CP2", "O1", "FT8"]
    for ch in channels_to_check:
        try:
            sig = signal_data.get_channel_signal(ch)
            print(f"Channel {ch}:")
            print(f"  - Mean: {np.mean(sig):.4f}")
            print(f"  - Std:  {np.std(sig):.4f}")
            print(f"  - Var:  {np.var(sig):.4f}")
            print(f"  - Min:  {np.min(sig):.4f}")
            print(f"  - Max:  {np.max(sig):.4f}")
        except Exception as e:
            print(f"Channel {ch} check failed: {e}")
else:
    print("No SEED-VIG files found!")
