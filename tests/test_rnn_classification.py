import unittest
import numpy as np
from scipy.io import savemat
import tempfile
import shutil
from pathlib import Path
import subprocess
import os
import sys
import logging

class TestRNNClassification(unittest.TestCase):
    
    def setUp(self):
        # Create temp dir for temporary test files
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        
        # Create separate directories to avoid overwriting files with same name
        self.signals_dir = self.test_path / "Raw_Data"
        self.signals_dir.mkdir()
        self.labels_dir = self.test_path / "Raw_Data_Labels"
        self.labels_dir.mkdir()
        
        # Create a mock SEED-VIG signal file
        self.sfreq = 100.0
        self.num_channels = 3
        self.num_samples = 6000  # 60 seconds of signal
        self.channels = ["C3", "C4", "CP2"]
        
        self.mock_data = np.random.randn(self.num_samples, self.num_channels).astype(np.float32)
        
        # Save in matlab struct format
        eeg_struct = {
            "chn": self.channels,
            "sample_rate": self.sfreq,
            "data": self.mock_data
        }
        self.mat_file_name = "mock_seed_vig.mat"
        self.mat_file_path = self.signals_dir / self.mat_file_name
        savemat(str(self.mat_file_path), {"EEG": eeg_struct})
        
        # Create a mock PERCLOS labels file (with same filename)
        # PERCLOS contains an array representing drowsiness scores per epoch (e.g. 8-second segments)
        num_epochs = int(np.ceil(self.num_samples / (8.0 * self.sfreq))) + 5
        self.mock_perclos = np.random.rand(num_epochs).astype(np.float32)
        savemat(str(self.labels_dir / self.mat_file_name), {"perclos": self.mock_perclos})
        
    def tearDown(self):
        # Clean up temp dir
        shutil.rmtree(self.test_dir)
        
    def test_classification_pipeline(self):
        # 1. Train forecasting model first (needed for rnn-model-path)
        rnn_model_path = self.test_path / "test_forecasting_model.h5"
        
        train_rnn_cmd = [
            sys.executable, "experiments/train_rnn.py",
            "--dataset-type", "seed_vig",
            "--channel", "CP2",
            "--model-version", "v0",
            "--input-min", "0.4",    # 24 seconds of input
            "--predict-min", "0.1",  # 6 seconds of prediction
            "--stride-sec", "5.0",
            "--epochs", "1",
            "--batch-size", "4",
            "--latent-dim", "8",
            "--output-model", str(rnn_model_path),
            "--force-cpu"
        ]
        
        env = os.environ.copy()
        workspace_root = Path(__file__).parent.parent
        env["PYTHONPATH"] = str(workspace_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
        env["SEED_VIG_DIR"] = str(self.signals_dir)
        env["SEED_VIG_LABELS"] = str(self.labels_dir)
        
        logging.info("Training forecasting RNN via subprocess...")
        res_rnn = subprocess.run(train_rnn_cmd, capture_output=True, text=True, env=env)
        if res_rnn.returncode != 0:
            print("STDOUT:", res_rnn.stdout)
            print("STDERR:", res_rnn.stderr)
        self.assertEqual(res_rnn.returncode, 0)
        self.assertTrue(rnn_model_path.exists())
        
        # 2. Train coupled RNN-MLP classifier model
        classifier_model_path = self.test_path / "test_coupled_model.h5"
        
        train_classifier_cmd = [
            sys.executable, "experiments/train_rnn_classifier.py",
            "--dataset-type", "seed_vig",
            "--channel", "CP2",
            "--rnn-model-path", str(rnn_model_path),
            "--input-min", "0.4",
            "--predict-min", "0.1",
            "--stride-sec", "5.0",
            "--epochs", "1",
            "--batch-size", "4",
            "--learning-rate", "0.01",
            "--train-split", "0.8",
            "--output-model", str(classifier_model_path),
            "--force-cpu"
        ]
        
        logging.info("Training coupled RNN-MLP classifier via subprocess...")
        res_clf = subprocess.run(train_classifier_cmd, capture_output=True, text=True, env=env)
        if res_clf.returncode != 0:
            print("STDOUT:", res_clf.stdout)
            print("STDERR:", res_clf.stderr)
        self.assertEqual(res_clf.returncode, 0)
        self.assertTrue(classifier_model_path.exists())

if __name__ == "__main__":
    unittest.main()
