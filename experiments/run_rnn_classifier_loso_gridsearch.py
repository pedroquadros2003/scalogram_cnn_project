import argparse
import itertools
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import yaml

from scalogram_cnn_project.utils.dict_product import dict_product
from scalogram_cnn_project.utils.simplify_config_space import simplify_config_space
from scalogram_cnn_project.utils.make_hash_id import make_hash_id
import scalogram_cnn_project.settings.config as config
from experiments.run_rnn_classifier_loso import discover_dataset_subjects, plot_loso_evolution_overview

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run Hyperparameter Grid Search for Coupled RNN-MLP Classifier under Leave-One-Subject-Out (LOSO) Cross-Validation"
    )
    parser.add_argument("--output_folder", "--output-folder", type=str, default="rnn_classifier_loso_gridsearch", help="Output folder name inside OUTPUT_DIR")
    parser.add_argument("--params_file", "--params-file", "--config", type=str, default="configs/hyperparameter_search_rnn/seedvig_loso_classifier_grid_example.yaml", help="YAML search space configuration file path")
    parser.add_argument("--subjects", type=int, nargs="+", default=None, help="Override subject IDs evaluated in each LOSO fold")
    parser.add_argument("--save-plot", action=argparse.BooleanOptionalAction, default=True, help="Whether to generate per-candidate multi-panel overview plots")
    parser.add_argument("--force-cpu", action="store_true", help="Force candidate executions to run on CPU")

    args = parser.parse_args()

    params_file_path = Path(args.params_file)
    if not params_file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {params_file_path}")

    with open(params_file_path, "r") as f:
        search_config = yaml.safe_load(f)

    model_hp = simplify_config_space(search_config.get("MODEL_HYPER_PARAMS", {}))
    train_hp = simplify_config_space(search_config.get("MODEL_TRAIN_PARAMS", {}))

    model_configs = list(dict_product(model_hp))
    train_configs = list(dict_product(train_hp))

    logger.info("=" * 75)
    logger.info("STARTING LOSO HYPERPARAMETER GRID SEARCH")
    logger.info(f"  - Config Space: {len(model_configs)} model configs x {len(train_configs)} train configs = {len(model_configs) * len(train_configs)} total candidates")
    logger.info(f"  - Params File:  {params_file_path}")
    logger.info("=" * 75)

    grid_candidates = []
    candidate_id_counter = 0
    param_registry = {}

    for m_hp, t_hp in itertools.product(model_configs, train_configs):
        cand_params = {}
        cand_params.update(m_hp)
        cand_params.update(t_hp)

        cand_id = f"cand_{candidate_id_counter:05d}"
        candidate_id_counter += 1
        cand_params["candidate_id"] = cand_id

        grid_candidates.append(cand_params)
        param_registry[cand_id] = {**m_hp, **t_hp}

    output_dir = config.OUTPUT_DIR / args.output_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = output_dir / "log.txt"
    file_handler = logging.FileHandler(log_file_path, mode="a")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)

    progress_file = output_dir / "progress.json"
    registry_file = output_dir / "param_registry.json"
    results_jsonl = output_dir / "results.jsonl"

    with open(registry_file, "w") as f:
        json.dump(param_registry, f, indent=2)

    progress_data = {}
    if progress_file.exists():
        try:
            with open(progress_file, "r") as f:
                progress_data = json.load(f)
            logger.info(f"Resuming grid search. Found {len(progress_data)} previously evaluated candidates.")
        except Exception as e:
            logger.warning(f"Could not read progress.json: {e}")

    for idx, cand_params in enumerate(grid_candidates):
        cand_id = cand_params["candidate_id"]
        hash_id = make_hash_id(param_registry[cand_id])

        if cand_id in progress_data:
            logger.info(f"[{idx + 1}/{len(grid_candidates)}] Candidate {cand_id} ({hash_id}) already processed. Skipping.")
            continue

        logger.info("=" * 75)
        logger.info(f"[{idx + 1}/{len(grid_candidates)}] EVALUATING LOSO CANDIDATE: {cand_id} (hash: {hash_id})")
        logger.info(f"  Hyperparameters: {param_registry[cand_id]}")
        logger.info("=" * 75)

        cand_dir = output_dir / f"{cand_id}_{hash_id}"
        cand_dir.mkdir(parents=True, exist_ok=True)

        dataset_type = cand_params.get("dataset_type", "seed_vig")
        channel = cand_params.get("channel", "CP2")
        lead_time_sec = float(cand_params.get("lead_time_sec", 0.0))

        # Determine subjects to evaluate
        if args.subjects:
            loso_subjects = sorted(list(set(args.subjects)))
        elif "subjects" in cand_params and cand_params["subjects"]:
            if isinstance(cand_params["subjects"], list):
                loso_subjects = sorted(cand_params["subjects"])
            else:
                loso_subjects = [int(cand_params["subjects"])]
        else:
            loso_subjects = discover_dataset_subjects(dataset_type)

        logger.info(f"Candidate {cand_id} evaluating across {len(loso_subjects)} LOSO folds: {loso_subjects}")

        fold_metrics = {}
        candidate_failed = False

        for fold_idx, val_subj in enumerate(loso_subjects):
            logger.info(f"--- [{cand_id}] FOLD {fold_idx + 1}/{len(loso_subjects)}: Validation Subject S{val_subj} ---")

            fold_dir = cand_dir / f"fold_subj_{val_subj:02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            fold_config = dict(cand_params)
            fold_config["validation_subject"] = val_subj
            fold_config["save_plot"] = args.save_plot

            temp_config_path = fold_dir / "config.yaml"
            with open(temp_config_path, "w") as f:
                yaml.dump(fold_config, f, default_flow_style=False)

            metrics_json_path = fold_dir / "metrics.json"
            plot_path = fold_dir / "history.png"
            model_path = fold_dir / f"model_fold_subj_{val_subj:02d}.h5"

            cmd = [
                sys.executable, "experiments/train_rnn_classifier.py",
                "--config", str(temp_config_path),
                "--metrics-json-path", str(metrics_json_path),
                "--output-model", str(model_path),
                "--output-plot", str(plot_path),
                "--validation-subject", str(val_subj)
            ]
            if not args.save_plot:
                cmd.append("--no-save-plot")
            if args.force_cpu:
                cmd.append("--force-cpu")

            env = os.environ.copy()
            workspace_root = Path(__file__).parent.parent
            env["PYTHONPATH"] = str(workspace_root / "src") + os.pathsep + str(workspace_root) + os.pathsep + env.get("PYTHONPATH", "")

            try:
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )

                with open(log_file_path, "a") as log_file:
                    for line in process.stdout:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                        log_file.write(line)
                        log_file.flush()

                return_code = process.wait()

                if return_code == 0 and metrics_json_path.exists():
                    with open(metrics_json_path, "r") as mf:
                        metrics = json.load(mf)
                    fold_metrics[val_subj] = metrics
                    val_acc = metrics.get("val_accuracy", metrics.get("accuracy", 0.0))
                    val_loss = metrics.get("val_loss", metrics.get("loss", 0.0))
                    logger.info(f"[{cand_id}] Fold S{val_subj} complete. Val Acc: {val_acc*100:.2f}%, Val Loss: {val_loss:.4f}")
                else:
                    logger.error(f"[{cand_id}] Fold S{val_subj} failed with exit code {return_code}.")
                    fold_metrics[val_subj] = {"val_accuracy": 0.0, "val_loss": 99.0, "status": "FAILED"}
                    candidate_failed = True

            except Exception as e:
                logger.error(f"[{cand_id}] Exception during fold S{val_subj}: {e}")
                fold_metrics[val_subj] = {"val_accuracy": 0.0, "val_loss": 99.0, "status": "ERROR"}
                candidate_failed = True

        # Aggregate candidate statistics across folds
        val_acc_list = [m.get("val_accuracy", m.get("accuracy", 0.0)) for m in fold_metrics.values() if m.get("status") not in ["FAILED", "ERROR"]]
        val_loss_list = [m.get("val_loss", m.get("loss", 0.0)) for m in fold_metrics.values() if m.get("status") not in ["FAILED", "ERROR"]]
        train_acc_list = [m.get("accuracy", 0.0) for m in fold_metrics.values() if "accuracy" in m]
        train_loss_list = [m.get("loss", 0.0) for m in fold_metrics.values() if "loss" in m]

        mean_val_acc = float(np.mean(val_acc_list)) if val_acc_list else 0.0
        std_val_acc = float(np.std(val_acc_list)) if val_acc_list else 0.0
        mean_val_loss = float(np.mean(val_loss_list)) if val_loss_list else 99.0
        std_val_loss = float(np.std(val_loss_list)) if val_loss_list else 0.0

        mean_train_acc = float(np.mean(train_acc_list)) if train_acc_list else 0.0
        mean_train_loss = float(np.mean(train_loss_list)) if train_loss_list else 99.0

        logger.info(f"=== CANDIDATE {cand_id} ({hash_id}) LOSO RESULTS ===")
        logger.info(f"  Mean Val Accuracy: {mean_val_acc * 100.0:.2f}% ± {std_val_acc * 100.0:.2f}%")
        logger.info(f"  Mean Val Loss:     {mean_val_loss:.4f} ± {std_val_loss:.4f}")
        logger.info(f"  Mean Train Acc:    {mean_train_acc * 100.0:.2f}%")

        # Save candidate summary
        cand_summary_path = cand_dir / "loso_summary.json"
        cand_summary = {
            "candidate_id": cand_id,
            "hash_id": hash_id,
            "hyperparameters": param_registry[cand_id],
            "dataset_type": dataset_type,
            "channel": channel,
            "lead_time_sec": lead_time_sec,
            "total_folds": len(loso_subjects),
            "evaluated_subjects": loso_subjects,
            "mean_val_accuracy": mean_val_acc,
            "std_val_accuracy": std_val_acc,
            "mean_val_loss": mean_val_loss,
            "std_val_loss": std_val_loss,
            "mean_train_accuracy": mean_train_acc,
            "mean_train_loss": mean_train_loss,
            "per_subject_results": {sid: fold_metrics[sid] for sid in loso_subjects}
        }
        with open(cand_summary_path, "w") as csf:
            json.dump(cand_summary, csf, indent=2)

        # Multi-panel overview plot
        cand_plot_path = cand_dir / "loso_evolution_overview.png"
        if args.save_plot:
            try:
                title = f"{cand_id} ({hash_id}) - {dataset_type.upper()} ({channel}) Lead {int(lead_time_sec)}s"
                plot_loso_evolution_overview(fold_metrics, cand_plot_path, title_prefix=title)
            except Exception as pe:
                logger.error(f"Could not generate overview plot for {cand_id}: {pe}")

        # Record candidate results
        result_entry = {
            "candidate_id": cand_id,
            "hash_id": hash_id,
            "mean_val_accuracy": mean_val_acc,
            "std_val_accuracy": std_val_acc,
            "mean_val_loss": mean_val_loss,
            "std_val_loss": std_val_loss,
            "mean_train_accuracy": mean_train_acc,
            "mean_train_loss": mean_train_loss,
            "summary_path": str(cand_summary_path),
            "plot_path": str(cand_plot_path) if args.save_plot else None,
            "hyperparameters": param_registry[cand_id],
            "status": "FAILED" if candidate_failed else "SUCCESS"
        }

        with open(results_jsonl, "a") as rf:
            rf.write(json.dumps(result_entry) + "\n")

        progress_data[cand_id] = {
            "status": "FAILED" if candidate_failed else "SUCCESS",
            "mean_val_accuracy": mean_val_acc,
            "std_val_accuracy": std_val_acc,
            "mean_val_loss": mean_val_loss,
            "hash_id": hash_id
        }

        with open(progress_file, "w") as pf:
            json.dump(progress_data, pf, indent=2)

    logger.info("=" * 75)
    logger.info("LOSO HYPERPARAMETER GRID SEARCH COMPLETED SUCCESSFULLY")
    logger.info(f"Results registry saved to: {results_jsonl}")
    logger.info("=" * 75)


if __name__ == "__main__":
    main()
