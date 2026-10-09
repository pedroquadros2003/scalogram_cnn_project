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
import yaml


class TestJointGridSearch(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)

        self.signals_dir = self.test_path / "Raw_Data"
        self.signals_dir.mkdir()
        self.labels_dir = self.test_path / "Raw_Data_Labels"
        self.labels_dir.mkdir()

        # Create mock SEED-VIG files for subjects 1 and 2
        self.sfreq = 100.0
        self.num_samples = 4000  # 40 seconds
        self.channels = ["C3", "C4", "CP2"]

        for subj_id in [1, 2]:
            mock_data = np.random.randn(self.num_samples, len(self.channels)).astype(np.float32)
            eeg_struct = {
                "chn": self.channels,
                "sample_rate": self.sfreq,
                "data": mock_data
            }
            mat_file_name = f"{subj_id}_20151012_mock.mat"
            savemat(str(self.signals_dir / mat_file_name), {"EEG": eeg_struct})

            num_epochs = int(np.ceil(self.num_samples / (8.0 * self.sfreq))) + 5
            mock_perclos = np.random.rand(num_epochs).astype(np.float32)
            savemat(str(self.labels_dir / mat_file_name), {"perclos": mock_perclos})

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_joint_gridsearch_temporal_subprocess(self):
        env = os.environ.copy()
        workspace_root = Path(__file__).parent.parent
        env["PYTHONPATH"] = str(workspace_root / "src") + os.pathsep + str(workspace_root) + os.pathsep + env.get("PYTHONPATH", "")
        env["SEED_VIG_DIR"] = str(self.signals_dir)
        env["SEED_VIG_LABELS"] = str(self.labels_dir)

        # Create minimal temporal search config
        search_config = {
            "MODEL_HYPER_PARAMS": {
                "predict_version": {"mode": "fixed", "values": ["v0"]},
                "classifier_version": {"mode": "fixed", "values": ["v0"]},
                "loss_alpha": {"mode": "choice", "values": [0.3, 0.7]},
                "latent_dim": {"mode": "fixed", "values": [16]},
                "learning_rate": {"mode": "fixed", "values": [0.01]},
                "hidden_units_1": {"mode": "fixed", "values": [16]},
                "hidden_units_2": {"mode": "fixed", "values": [8]},
                "dropout_1": {"mode": "fixed", "values": [0.1]},
                "dropout_2": {"mode": "fixed", "values": [0.1]},
                "optimizer_name": {"mode": "fixed", "values": ["adam"]}
            },
            "MODEL_TRAIN_PARAMS": {
                "dataset_type": {"mode": "fixed", "values": ["seed_vig"]},
                "channel": {"mode": "fixed", "values": ["CP2"]},
                "lead_time_sec": {"mode": "fixed", "values": [0.0]},
                "forecast_steps": {"mode": "fixed", "values": [1]},
                "input_sec": {"mode": "fixed", "values": [1.0]},
                "stride_sec": {"mode": "fixed", "values": [5.0]},
                "epochs": {"mode": "fixed", "values": [1]},
                "batch_size": {"mode": "fixed", "values": [4]},
                "train_ratio": {"mode": "fixed", "values": [0.8]},
                "drowsiness_threshold": {"mode": "fixed", "values": [0.5]}
            }
        }

        config_path = self.test_path / "temp_temporal_grid.yaml"
        with open(config_path, "w") as f:
            yaml.dump(search_config, f)

        output_folder_name = "test_joint_temporal_grid_out"

        cmd = [
            sys.executable, "experiments/run_joint_gridsearch.py",
            "--params-file", str(config_path),
            "--output-folder", output_folder_name,
            "--force-cpu",
            "--no-save-plot"
        ]

        res = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if res.returncode != 0:
            print("STDOUT:", res.stdout)
            print("STDERR:", res.stderr)
        self.assertEqual(res.returncode, 0)

        import scalogram_cnn_project.settings.config as proj_config
        out_dir = proj_config.OUTPUT_DIR / output_folder_name
        self.assertTrue((out_dir / "progress.json").exists())
        self.assertTrue((out_dir / "results.jsonl").exists())
        self.assertTrue((out_dir / "gridsearch_summary.json").exists())

        with open(out_dir / "progress.json", "r") as f:
            progress = json.load(f)
        self.assertEqual(len(progress), 2)  # 2 candidates (alpha 0.3 and 0.7)

        # Cleanup output test dir
        if out_dir.exists():
            shutil.rmtree(out_dir)

    def test_joint_gridsearch_loso_subprocess(self):
        env = os.environ.copy()
        workspace_root = Path(__file__).parent.parent
        env["PYTHONPATH"] = str(workspace_root / "src") + os.pathsep + str(workspace_root) + os.pathsep + env.get("PYTHONPATH", "")
        env["SEED_VIG_DIR"] = str(self.signals_dir)
        env["SEED_VIG_LABELS"] = str(self.labels_dir)

        # Create minimal LOSO search config
        search_config = {
            "MODEL_HYPER_PARAMS": {
                "predict_version": {"mode": "fixed", "values": ["v0"]},
                "classifier_version": {"mode": "fixed", "values": ["v1"]},
                "loss_alpha": {"mode": "fixed", "values": [0.5]},
                "latent_dim": {"mode": "fixed", "values": [16]},
                "learning_rate": {"mode": "fixed", "values": [0.01]},
                "hidden_units_1": {"mode": "fixed", "values": [16]},
                "hidden_units_2": {"mode": "fixed", "values": [8]},
                "dropout_1": {"mode": "fixed", "values": [0.1]},
                "dropout_2": {"mode": "fixed", "values": [0.1]},
                "optimizer_name": {"mode": "fixed", "values": ["adam"]}
            },
            "MODEL_TRAIN_PARAMS": {
                "dataset_type": {"mode": "fixed", "values": ["seed_vig"]},
                "channel": {"mode": "fixed", "values": ["CP2"]},
                "lead_time_sec": {"mode": "fixed", "values": [0.0]},
                "forecast_steps": {"mode": "fixed", "values": [1]},
                "input_sec": {"mode": "fixed", "values": [1.0]},
                "stride_sec": {"mode": "fixed", "values": [5.0]},
                "epochs": {"mode": "fixed", "values": [1]},
                "batch_size": {"mode": "fixed", "values": [4]},
                "drowsiness_threshold": {"mode": "fixed", "values": [0.5]},
                "subjects": {"mode": "fixed", "values": [[1, 2]]}
            }
        }

        config_path = self.test_path / "temp_loso_grid.yaml"
        with open(config_path, "w") as f:
            yaml.dump(search_config, f)

        output_folder_name = "test_joint_loso_grid_out"

        cmd = [
            sys.executable, "experiments/run_joint_gridsearch.py",
            "--params-file", str(config_path),
            "--output-folder", output_folder_name,
            "--force-cpu",
            "--no-save-plot"
        ]

        res = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if res.returncode != 0:
            print("STDOUT:", res.stdout)
            print("STDERR:", res.stderr)
        self.assertEqual(res.returncode, 0)

        import scalogram_cnn_project.settings.config as proj_config
        out_dir = proj_config.OUTPUT_DIR / output_folder_name
        self.assertTrue((out_dir / "progress.json").exists())
        self.assertTrue((out_dir / "results.jsonl").exists())
        self.assertTrue((out_dir / "gridsearch_summary.json").exists())

        with open(out_dir / "progress.json", "r") as f:
            progress = json.load(f)
        self.assertEqual(len(progress), 1)

        cand_data = list(progress.values())[0]
        self.assertIn("mean_val_accuracy", cand_data)
        self.assertIn("std_val_accuracy", cand_data)
        self.assertEqual(len(cand_data["evaluated_subjects"]), 2)

        # Cleanup output test dir
        if out_dir.exists():
            shutil.rmtree(out_dir)


if __name__ == "__main__":
    unittest.main()
