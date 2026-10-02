import unittest
import shutil
from pathlib import Path
import numpy as np

from experiments.run_rnn_classifier_loso import discover_dataset_subjects, plot_loso_evolution_overview


class TestRNNLOSO(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path("outputs/test_loso_unit")
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_discover_subjects(self):
        seed_subjs = discover_dataset_subjects("seed_vig")
        self.assertIsInstance(seed_subjs, list)
        self.assertTrue(len(seed_subjs) > 0)
        self.assertIn(1, seed_subjs)

    def test_plot_loso_evolution_overview(self):
        # Create mock fold metrics
        mock_fold_metrics = {
            1: {
                "val_accuracy": 0.75,
                "accuracy": 0.82,
                "val_loss": 0.55,
                "loss": 0.42,
                "history": {
                    "loss": [0.65, 0.52, 0.42],
                    "val_loss": [0.70, 0.61, 0.55],
                    "accuracy": [0.60, 0.72, 0.82],
                    "val_accuracy": [0.55, 0.68, 0.75]
                }
            },
            2: {
                "val_accuracy": 0.68,
                "accuracy": 0.85,
                "val_loss": 0.62,
                "loss": 0.38,
                "history": {
                    "loss": [0.62, 0.48, 0.38],
                    "val_loss": [0.68, 0.64, 0.62],
                    "accuracy": [0.65, 0.78, 0.85],
                    "val_accuracy": [0.58, 0.62, 0.68]
                }
            },
            3: {
                "val_accuracy": 0.81,
                "accuracy": 0.88,
                "val_loss": 0.49,
                "loss": 0.35,
                "history": {
                    "loss": [0.58, 0.44, 0.35],
                    "val_loss": [0.61, 0.53, 0.49],
                    "accuracy": [0.70, 0.80, 0.88],
                    "val_accuracy": [0.64, 0.75, 0.81]
                }
            }
        }

        output_plot = self.test_dir / "test_loso_overview.png"
        plot_loso_evolution_overview(mock_fold_metrics, output_plot, title_prefix="Test LOSO")

        self.assertTrue(output_plot.exists())
        self.assertGreater(output_plot.stat().st_size, 1000)

    def test_loso_gridsearch_config(self):
        import yaml
        from scalogram_cnn_project.utils.dict_product import dict_product
        from scalogram_cnn_project.utils.simplify_config_space import simplify_config_space

        cfg_path = Path("configs/hyperparameter_search_rnn/seedvig_loso_classifier_grid_example.yaml")
        self.assertTrue(cfg_path.exists())

        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        model_hp = simplify_config_space(cfg.get("MODEL_HYPER_PARAMS", {}))
        train_hp = simplify_config_space(cfg.get("MODEL_TRAIN_PARAMS", {}))

        model_configs = list(dict_product(model_hp))
        train_configs = list(dict_product(train_hp))

        self.assertGreater(len(model_configs), 0)
        self.assertGreater(len(train_configs), 0)


if __name__ == "__main__":
    unittest.main()
