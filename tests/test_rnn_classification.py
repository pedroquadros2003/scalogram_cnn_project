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

    def test_classification_from_scratch_lstm_and_gru_pipeline(self):
        env = os.environ.copy()
        workspace_root = Path(__file__).parent.parent
        env["PYTHONPATH"] = str(workspace_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
        env["SEED_VIG_DIR"] = str(self.signals_dir)
        env["SEED_VIG_LABELS"] = str(self.labels_dir)

        # 1. Train from scratch using LSTM backbone (rnn-model-path null)
        lstm_model_path = self.test_path / "test_scratch_lstm_model.h5"
        train_lstm_cmd = [
            sys.executable, "experiments/train_rnn_classifier.py",
            "--dataset-type", "seed_vig",
            "--channel", "CP2",
            "--model-version", "v0",
            "--rnn-model-path", "null",
            "--rnn-type", "lstm",
            "--latent-dim", "16",
            "--input-sec", "1.0",
            "--lead-time-sec", "5.0",
            "--stride-sec", "5.0",
            "--epochs", "1",
            "--batch-size", "4",
            "--learning-rate", "0.01",
            "--output-model", str(lstm_model_path),
            "--force-cpu"
        ]
        logging.info("Training classifier from scratch (LSTM) via subprocess...")
        res_lstm = subprocess.run(train_lstm_cmd, capture_output=True, text=True, env=env)
        if res_lstm.returncode != 0:
            print("STDOUT:", res_lstm.stdout)
            print("STDERR:", res_lstm.stderr)
        self.assertEqual(res_lstm.returncode, 0)
        self.assertTrue(lstm_model_path.exists())

        # 2. Train from scratch using GRU backbone and v1 architecture
        gru_model_path = self.test_path / "test_scratch_gru_model.h5"
        train_gru_cmd = [
            sys.executable, "experiments/train_rnn_classifier.py",
            "--dataset-type", "seed_vig",
            "--channel", "CP2",
            "--model-version", "v1",
            "--rnn-model-path", "random",
            "--rnn-type", "gru",
            "--latent-dim", "8",
            "--input-sec", "1.0",
            "--lead-time-sec", "0.0",
            "--stride-sec", "5.0",
            "--epochs", "1",
            "--batch-size", "4",
            "--learning-rate", "0.01",
            "--output-model", str(gru_model_path),
            "--force-cpu"
        ]
        logging.info("Training classifier from scratch (GRU v1) via subprocess...")
        res_gru = subprocess.run(train_gru_cmd, capture_output=True, text=True, env=env)
        if res_gru.returncode != 0:
            print("STDOUT:", res_gru.stdout)
            print("STDERR:", res_gru.stderr)
        self.assertEqual(res_gru.returncode, 0)
        self.assertTrue(gru_model_path.exists())

    def test_direct_coupled_model_creation_from_scratch(self):
        from scalogram_cnn_project.models_for_prediction_classification import create_coupled_classifier_model
        
        for version in ["v0", "v1"]:
            for rnn_type in ["lstm", "gru"]:
                params = {
                    "input_len": 100,
                    "latent_dim": 24,
                    "rnn_type": rnn_type,
                    "learning_rate": 0.001,
                    "hidden_units_1": 32,
                    "hidden_units_2": 16,
                    "dropout_1": 0.2,
                    "dropout_2": 0.1,
                    "fine_tune_rnn": True
                }
                # Test with rnn_model=None
                model_none = create_coupled_classifier_model(version, None, params)
                self.assertIsNotNone(model_none)
                # Test with rnn_model="random"
                model_rand = create_coupled_classifier_model(version, "random", params)
                self.assertIsNotNone(model_rand)
                
                # Check model input/output shapes
                self.assertEqual(model_none.input_shape, (None, 100, 1))
                self.assertEqual(model_none.output_shape, (None, 1))

if __name__ == "__main__":
    unittest.main()
