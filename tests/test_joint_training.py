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

class TestJointTraining(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        
        self.signals_dir = self.test_path / "Raw_Data"
        self.signals_dir.mkdir()
        self.labels_dir = self.test_path / "Raw_Data_Labels"
        self.labels_dir.mkdir()
        
        # Create mock SEED-VIG file
        self.sfreq = 100.0
        self.num_samples = 6000  # 60 seconds
        self.channels = ["C3", "C4", "CP2"]
        self.mock_data = np.random.randn(self.num_samples, len(self.channels)).astype(np.float32)
        
        eeg_struct = {
            "chn": self.channels,
            "sample_rate": self.sfreq,
            "data": self.mock_data
        }
        self.mat_file_name = "mock_seed_vig.mat"
        self.mat_file_path = self.signals_dir / self.mat_file_name
        savemat(str(self.mat_file_path), {"EEG": eeg_struct})
        
        num_epochs = int(np.ceil(self.num_samples / (8.0 * self.sfreq))) + 5
        self.mock_perclos = np.random.rand(num_epochs).astype(np.float32)
        savemat(str(self.labels_dir / self.mat_file_name), {"perclos": self.mock_perclos})
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_joint_model_creation_all_architectures(self):
        from scalogram_cnn_project.models_for_joint_training import create_joint_model
        
        for p_ver in ["v0", "v1"]:
            for c_ver in ["v0", "v1"]:
                for alpha in [0.0, 0.5, 1.0]:
                    params = {
                        "input_len": 100,
                        "forecast_steps": 1,
                        "latent_dim": 16,
                        "loss_alpha": alpha,
                        "learning_rate": 0.001,
                        "hidden_units_1": 32,
                        "hidden_units_2": 16,
                        "dropout_1": 0.2,
                        "dropout_2": 0.1
                    }
                    model = create_joint_model(p_ver, c_ver, params)
                    self.assertIsNotNone(model)
                    self.assertEqual(model.input_shape, (None, 100, 1))
                    self.assertIn("forecast_output", model.output_names)
                    self.assertIn("classification_output", model.output_names)
                    
    def test_joint_sequence_batch_generation(self):
        from scalogram_cnn_project.utils.joint_sequence import JointSequence
        
        signals = [np.random.randn(2000).astype(np.float32)]
        indices = [(0, i, float(np.random.randn()), int(np.random.choice([0, 1]))) for i in range(0, 1500, 50)]
        
        seq = JointSequence(
            loaded_signals=signals,
            indices_with_dual_labels=indices,
            input_len=100,
            forecast_steps=1,
            batch_size=8,
            shuffle=True
        )
        self.assertGreater(len(seq), 0)
        batch_x, batch_y = seq[0]
        
        self.assertEqual(batch_x.shape, (8, 100, 1))
        self.assertIn("forecast_output", batch_y)
        self.assertIn("classification_output", batch_y)
        self.assertEqual(batch_y["forecast_output"].shape, (8, 1))
        self.assertEqual(batch_y["classification_output"].shape, (8, 1))

    def test_joint_training_pipeline_subprocess(self):
        env = os.environ.copy()
        workspace_root = Path(__file__).parent.parent
        env["PYTHONPATH"] = str(workspace_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
        env["SEED_VIG_DIR"] = str(self.signals_dir)
        env["SEED_VIG_LABELS"] = str(self.labels_dir)
        
        joint_model_path = self.test_path / "test_joint_model.h5"
        plot_path = self.test_path / "test_joint_history.png"
        metrics_json_path = self.test_path / "test_joint_metrics.json"
        
        train_joint_cmd = [
            sys.executable, "experiments/train_joint_model.py",
            "--dataset-type", "seed_vig",
            "--channel", "CP2",
            "--predict-version", "v0",
            "--classifier-version", "v1",
            "--loss-alpha", "0.5",
            "--lead-time-sec", "0.0",
            "--forecast-steps", "1",
            "--input-sec", "1.0",
            "--stride-sec", "5.0",
            "--epochs", "1",
            "--batch-size", "4",
            "--learning-rate", "0.01",
            "--latent-dim", "16",
            "--output-model", str(joint_model_path),
            "--output-plot", str(plot_path),
            "--metrics-json-path", str(metrics_json_path),
            "--force-cpu"
        ]
        
        res = subprocess.run(train_joint_cmd, capture_output=True, text=True, env=env)
        if res.returncode != 0:
            print("STDOUT:", res.stdout)
            print("STDERR:", res.stderr)
        self.assertEqual(res.returncode, 0)
        self.assertTrue(joint_model_path.exists())
        self.assertTrue(plot_path.exists())
        self.assertTrue(metrics_json_path.exists())

if __name__ == "__main__":
    unittest.main()
