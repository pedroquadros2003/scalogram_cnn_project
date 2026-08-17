import unittest
import numpy as np
from scipy.io import savemat
import tempfile
import shutil
from pathlib import Path
import subprocess
import os
import sys
import json
import logging
import yaml

class TestRNNGridSearch(unittest.TestCase):
    
    def setUp(self):
        # Create temp dir for temporary test files
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        
        # Create subdirectories for SEED-VIG signals and labels
        self.signals_dir = self.test_path / "Raw_Data"
        self.signals_dir.mkdir()
        self.labels_dir = self.test_path / "Raw_Data_Labels"
        self.labels_dir.mkdir()
        
        # Create mock SEED-VIG signal file
        self.sfreq = 100.0
        self.num_channels = 3
        self.num_samples = 6000  # 60 seconds
        self.channels = ["C3", "C4", "CP2"]
        
        self.mock_data = np.random.randn(self.num_samples, self.num_channels).astype(np.float32)
        
        eeg_struct = {
            "chn": self.channels,
            "sample_rate": self.sfreq,
            "data": self.mock_data
        }
        self.mat_file_name = "mock_seed_vig.mat"
        savemat(str(self.signals_dir / self.mat_file_name), {"EEG": eeg_struct})
        
        # Create mock PERCLOS labels file
        num_epochs = int(np.ceil(self.num_samples / (8.0 * self.sfreq))) + 5
        self.mock_perclos = np.random.rand(num_epochs).astype(np.float32)
        savemat(str(self.labels_dir / self.mat_file_name), {"perclos": self.mock_perclos})
        
    def tearDown(self):
        # Clean up temp dir
        shutil.rmtree(self.test_dir)
        
        # Clean up workspace test output folders
        workspace_root = Path(__file__).parent.parent
        for path in [workspace_root / "outputs" / "test_forecast_search", workspace_root / "outputs" / "test_classifier_search"]:
            if path.exists():
                shutil.rmtree(path)
        
    def test_rnn_grid_searches(self):
        env = os.environ.copy()
        workspace_root = Path(__file__).parent.parent
        env["PYTHONPATH"] = str(workspace_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
        env["SEED_VIG_DIR"] = str(self.signals_dir)
        env["SEED_VIG_LABELS"] = str(self.labels_dir)
        
        # 1. Create a search space YAML config for forecasting
        forecast_config = {
            "MODEL_HYPER_PARAMS": {
                "latent_dim": {
                    "mode": "choice",
                    "values": [4, 8]
                },
                "model_version": {
                    "mode": "fixed",
                    "values": ["v0"]
                }
            },
            "MODEL_TRAIN_PARAMS": {
                "epochs": {
                    "mode": "fixed",
                    "values": [1]
                },
                "batch_size": {
                    "mode": "fixed",
                    "values": [4]
                },
                "dataset_type": {
                    "mode": "fixed",
                    "values": ["seed_vig"]
                },
                "channel": {
                    "mode": "fixed",
                    "values": ["CP2"]
                },
                "input_min": {
                    "mode": "fixed",
                    "values": [0.4]
                },
                "predict_min": {
                    "mode": "fixed",
                    "values": [0.1]
                },
                "stride_sec": {
                    "mode": "fixed",
                    "values": [5.0]
                },
                "train_split": {
                    "mode": "fixed",
                    "values": [0.8]
                }
            }
        }
        
        forecast_yaml_path = self.test_path / "forecast_search.yaml"
        with open(forecast_yaml_path, "w") as f:
            yaml.dump(forecast_config, f)
            
        # Run forecasting grid search
        grid_forecast_cmd = [
            sys.executable, "experiments/run_rnn_gridsearch.py",
            "--output_folder", "test_forecast_search",
            "--params_file", str(forecast_yaml_path),
            "--force-cpu"
        ]
        
        logging.info("Running forecasting grid search...")
        res_fc = subprocess.run(grid_forecast_cmd, capture_output=True, text=True, env=env)
        if res_fc.returncode != 0:
            print("STDOUT:", res_fc.stdout)
            print("STDERR:", res_fc.stderr)
        self.assertEqual(res_fc.returncode, 0)
        
        search_out_dir = workspace_root / "outputs" / "test_forecast_search"
        self.assertTrue((search_out_dir / "progress.json").exists())
        self.assertTrue((search_out_dir / "param_registry.json").exists())
        
        # Verify that model weights files are created
        self.assertTrue((search_out_dir / "rnn_predict_cand_00000.h5").exists())
        self.assertTrue((search_out_dir / "rnn_predict_cand_00001.h5").exists())
        
        # 2. Create a search space YAML config for classification
        classifier_config = {
            "MODEL_HYPER_PARAMS": {
                "learning_rate": {
                    "mode": "choice",
                    "values": [0.01, 0.005]
                }
            },
            "MODEL_TRAIN_PARAMS": {
                "epochs": {
                    "mode": "fixed",
                    "values": [1]
                },
                "batch_size": {
                    "mode": "fixed",
                    "values": [4]
                },
                "dataset_type": {
                    "mode": "fixed",
                    "values": ["seed_vig"]
                },
                "channel": {
                    "mode": "fixed",
                    "values": ["CP2"]
                },
                "input_min": {
                    "mode": "fixed",
                    "values": [0.4]
                },
                "predict_min": {
                    "mode": "fixed",
                    "values": [0.1]
                },
                "stride_sec": {
                    "mode": "fixed",
                    "values": [5.0]
                },
                "train_split": {
                    "mode": "fixed",
                    "values": [0.8]
                },
                "rnn_model_path": {
                    "mode": "fixed",
                    "values": [str(search_out_dir / "rnn_predict_cand_00000.h5")]
                }
            }
        }
        
        classifier_yaml_path = self.test_path / "classifier_search.yaml"
        with open(classifier_yaml_path, "w") as f:
            yaml.dump(classifier_config, f)
            
        # Run classification grid search
        grid_classifier_cmd = [
            sys.executable, "experiments/run_rnn_classifier_gridsearch.py",
            "--output_folder", "test_classifier_search",
            "--params_file", str(classifier_yaml_path),
            "--force-cpu"
        ]
        
        logging.info("Running coupled classification grid search...")
        res_clf = subprocess.run(grid_classifier_cmd, capture_output=True, text=True, env=env)
        if res_clf.returncode != 0:
            print("STDOUT:", res_clf.stdout)
            print("STDERR:", res_clf.stderr)
        self.assertEqual(res_clf.returncode, 0)
        
        clf_out_dir = workspace_root / "outputs" / "test_classifier_search"
        self.assertTrue((clf_out_dir / "progress.json").exists())
        self.assertTrue((clf_out_dir / "param_registry.json").exists())
        self.assertTrue((clf_out_dir / "combined_predict_classifier_cand_00000.h5").exists())
        self.assertTrue((clf_out_dir / "combined_predict_classifier_cand_00001.h5").exists())

if __name__ == "__main__":
    unittest.main()
