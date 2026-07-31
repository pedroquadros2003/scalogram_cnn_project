import unittest
import numpy as np
from scipy.io import savemat
import tempfile
import shutil
from pathlib import Path
import subprocess
import os
import logging

from scalogram_cnn_project.utils.signal_data import SignalData
from scalogram_cnn_project.utils.signal_loader import SignalLoader
from scalogram_cnn_project.models_for_prediction.model_predict_builder import create_prediction_model

class TestRNNPipeline(unittest.TestCase):
    
    def setUp(self):
        # Create temp dir for temporary test files
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        
        # Create a mock SEED-VIG .mat file
        # SEED-VIG format has an "EEG" struct with chn, sample_rate, and data (num_samples, num_channels)
        self.sfreq = 100.0
        self.num_channels = 3
        self.num_samples = 6000  # 60 seconds of signal
        self.channels = ["C3", "C4", "O1"]
        
        self.mock_data = np.random.randn(self.num_samples, self.num_channels).astype(np.float32)
        
        # Save in matlab struct format
        eeg_struct = {
            "chn": self.channels,
            "sample_rate": self.sfreq,
            "data": self.mock_data
        }
        self.mat_file_path = self.test_path / "mock_seed_vig.mat"
        savemat(str(self.mat_file_path), {"EEG": eeg_struct})
        
    def tearDown(self):
        # Clean up temp dir
        shutil.rmtree(self.test_dir)
        
    def test_signal_data(self):
        # Test shape transpose and window slicing
        data_2d = self.mock_data.T
        sig_data = SignalData(data_2d, self.channels, self.sfreq)
        
        self.assertEqual(sig_data.sfreq, self.sfreq)
        self.assertEqual(sig_data.channels, self.channels)
        self.assertTrue(np.array_equal(sig_data.get_channel_signal("C3"), data_2d[0]))
        
        # Test slice
        # 0.1 min to 0.5 min = 6s to 30s -> 600 to 3000 samples
        sliced = sig_data.get_channel_window("C3", 0.1, 0.5)
        self.assertEqual(len(sliced), int(0.4 * 60.0 * self.sfreq))
        
    def test_signal_loader(self):
        # Test load seed_vig
        sig_data = SignalLoader.load_signal(str(self.mat_file_path), "seed_vig")
        self.assertEqual(sig_data.sfreq, self.sfreq)
        self.assertEqual(sig_data.channels, self.channels)
        # Check transpose worked: data shape should be (num_channels, num_samples)
        self.assertEqual(sig_data.data.shape, (self.num_channels, self.num_samples))
        
    def test_model_builder(self):
        # Create parameters
        params = {
            "input_steps": 100,
            "output_steps": 50,
            "latent_dim": 16,
            "optimizer_name": "adam",
            "learning_rate": 0.01
        }
        # Test LSTM builder
        model_v0 = create_prediction_model("v0", params)
        self.assertEqual(model_v0.input_shape, (None, 100, 1))
        self.assertEqual(model_v0.output_shape, (None, 50, 1))
        
        # Test GRU builder
        model_v1 = create_prediction_model("v1", params)
        self.assertEqual(model_v1.input_shape, (None, 100, 1))
        self.assertEqual(model_v1.output_shape, (None, 50, 1))
        
    def test_pipeline_integration(self):
        # Run model training for 1 epoch and then run inference
        model_path = self.test_path / "test_model.h5"
        
        import sys
        train_cmd = [
            sys.executable, "experiments/train_rnn.py",
            "--dataset-type", "seed_vig",
            "--channel", "C3",
            "--model-version", "v0",
            "--input-min", "0.5",    # 30 seconds of input
            "--predict-min", "0.2",  # 12 seconds of prediction
            "--stride-sec", "5.0",
            "--epochs", "1",
            "--batch-size", "4",
            "--latent-dim", "8",
            "--output-model", str(model_path)
        ]
        
        env = os.environ.copy()
        # Set PYTHONPATH so the code can import scalogram_cnn_project
        workspace_root = Path(__file__).parent.parent
        env["PYTHONPATH"] = str(workspace_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
        env["SEED_VIG_DIR"] = self.test_dir
        
        logging.info("Running training script via subprocess...")
        res = subprocess.run(train_cmd, capture_output=True, text=True, env=env)
        
        if res.returncode != 0:
            print("STDOUT:", res.stdout)
            print("STDERR:", res.stderr)
        self.assertEqual(res.returncode, 0)
        self.assertTrue(model_path.exists())
        
        # 2. Run inference
        run_cmd = [
            sys.executable, "experiments/run_pipeline.py",
            "--file", str(self.mat_file_path),
            "--dataset-type", "seed_vig",
            "--channel", "C3",
            "--start-min", "0.1",
            "--end-min", "0.6",
            "--predict-min", "0.2",
            "--model-path", str(model_path),
            "--output-dir", self.test_dir
        ]
        
        logging.info("Running run_pipeline.py script via subprocess...")
        res_run = subprocess.run(run_cmd, capture_output=True, text=True, env=env)
        if res_run.returncode != 0:
            print("STDOUT:", res_run.stdout)
            print("STDERR:", res_run.stderr)
        self.assertEqual(res_run.returncode, 0)
        
        # Verify output files exist
        predicted_mat = self.test_path / "predicted_mock_seed_vig_C3.mat"
        predicted_plot = self.test_path / "plot_mock_seed_vig_C3.png"
        self.assertTrue(predicted_mat.exists())
        self.assertTrue(predicted_plot.exists())

if __name__ == "__main__":
    unittest.main()
