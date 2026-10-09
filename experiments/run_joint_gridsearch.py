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
from experiments.run_rnn_classifier_loso import discover_dataset_subjects

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run Hyperparameter Grid Search for Joint Multi-Task Model (Forecasting + Classification) with Temporal or LOSO Validation"
    )
    parser.add_argument("--output_folder", "--output-folder", type=str, default="joint_model_gridsearch", help="Output folder name inside OUTPUT_DIR")
    parser.add_argument("--params_file", "--params-file", "--config", type=str, default="configs/hyperparameter_search_rnn/joint_gridsearch_temporal_example.yaml", help="YAML search space configuration file path")
    parser.add_argument("--subjects", type=int, nargs="+", default=None, help="Override validation subject IDs for LOSO mode")
    parser.add_argument("--save-plot", action=argparse.BooleanOptionalAction, default=True, help="Whether to generate diagnostic plots for candidates/folds")
    parser.add_argument("--force-cpu", action="store_true", help="Force candidate executions to run on CPU")
    parser.add_argument("--loso", action="store_true", help="Force Leave-One-Subject-Out (LOSO) mode even if not specified in YAML")

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
    summary_file = output_dir / "gridsearch_summary.json"

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

    # Determine validation mode
    first_cand = grid_candidates[0] if grid_candidates else {}
    is_loso = (
        args.loso or
        args.subjects is not None or
        "subjects" in first_cand or
        first_cand.get("validation_mode") == "loso"
    )

    logger.info("=" * 80)
    logger.info("STARTING JOINT MULTI-TASK HYPERPARAMETER GRID SEARCH")
    logger.info(f"  - Validation Mode: {'Leave-One-Subject-Out (LOSO)' if is_loso else 'Temporal Split (Intra-Session)'}")
    logger.info(f"  - Total Candidates: {len(grid_candidates)}")
    logger.info(f"  - Search File:      {params_file_path}")
    logger.info(f"  - Output Folder:    {output_dir}")
    logger.info("=" * 80)

    best_candidate_id = None
    best_metric_val = -1.0 if is_loso else float("inf")

    for idx, cand_params in enumerate(grid_candidates):
        cand_id = cand_params["candidate_id"]
        hash_id = make_hash_id(param_registry[cand_id], prefix="joint", size=10)

        if cand_id in progress_data:
            logger.info(f"[{idx + 1}/{len(grid_candidates)}] Candidate {cand_id} ({hash_id}) already processed. Skipping.")
            continue

        logger.info("=" * 80)
        logger.info(f"[{idx + 1}/{len(grid_candidates)}] EVALUATING CANDIDATE: {cand_id} (hash: {hash_id})")
        logger.info(f"  Hyperparameters: {param_registry[cand_id]}")
        logger.info("=" * 80)

        cand_dir = output_dir / f"{cand_id}_{hash_id}"
        cand_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        workspace_root = Path(__file__).parent.parent
        env["PYTHONPATH"] = str(workspace_root / "src") + os.pathsep + str(workspace_root) + os.pathsep + env.get("PYTHONPATH", "")

        if is_loso:
            # --- LOSO EVALUATION ACROSS FOLDS ---
            dataset_type = cand_params.get("dataset_type", "seed_vig")
            if args.subjects:
                loso_subjects = sorted(list(set(args.subjects)))
            elif "subjects" in cand_params and cand_params["subjects"]:
                if isinstance(cand_params["subjects"], list):
                    loso_subjects = sorted(cand_params["subjects"])
                else:
                    loso_subjects = [int(cand_params["subjects"])]
            else:
                loso_subjects = discover_dataset_subjects(dataset_type)

            logger.info(f"Candidate {cand_id} running across {len(loso_subjects)} LOSO folds: {loso_subjects}")

            fold_metrics = {}
            candidate_failed = False

            for fold_idx, val_subj in enumerate(loso_subjects):
                logger.info(f"--- [{cand_id}] FOLD {fold_idx + 1}/{len(loso_subjects)}: Validation Subject S{val_subj:02d} ---")

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
                    sys.executable, "experiments/train_joint_model.py",
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
                        val_acc = metrics.get("val_classification_output_accuracy", metrics.get("val_accuracy", 0.0))
                        val_loss = metrics.get("val_loss", 0.0)
                        logger.info(f"[{cand_id}] Fold S{val_subj:02d} complete. Val Acc: {val_acc*100:.2f}%, Val Loss: {val_loss:.4f}")
                    else:
                        logger.error(f"[{cand_id}] Fold S{val_subj:02d} failed with exit code {return_code}.")
                        fold_metrics[val_subj] = {"val_accuracy": 0.0, "val_loss": 99.0, "status": "FAILED"}
                        candidate_failed = True

                except Exception as e:
                    logger.error(f"[{cand_id}] Exception during fold S{val_subj:02d}: {e}")
                    fold_metrics[val_subj] = {"val_accuracy": 0.0, "val_loss": 99.0, "status": "ERROR"}
                    candidate_failed = True

            val_acc_list = [m.get("val_classification_output_accuracy", m.get("val_accuracy", 0.0)) for m in fold_metrics.values() if m.get("status") not in ["FAILED", "ERROR"]]
            val_loss_list = [m.get("val_loss", 99.0) for m in fold_metrics.values() if m.get("status") not in ["FAILED", "ERROR"]]
            val_mae_list = [m.get("val_forecast_output_mae", 99.0) for m in fold_metrics.values() if m.get("status") not in ["FAILED", "ERROR"]]

            mean_val_acc = float(np.mean(val_acc_list)) if val_acc_list else 0.0
            std_val_acc = float(np.std(val_acc_list)) if val_acc_list else 0.0
            mean_val_loss = float(np.mean(val_loss_list)) if val_loss_list else 99.0
            std_val_loss = float(np.std(val_loss_list)) if val_loss_list else 0.0
            mean_val_mae = float(np.mean(val_mae_list)) if val_mae_list else 99.0

            cand_summary = {
                "candidate_id": cand_id,
                "hash_id": hash_id,
                "parameters": param_registry[cand_id],
                "mean_val_accuracy": mean_val_acc,
                "std_val_accuracy": std_val_acc,
                "mean_val_loss": mean_val_loss,
                "std_val_loss": std_val_loss,
                "mean_val_mae": mean_val_mae,
                "evaluated_subjects": loso_subjects,
                "fold_metrics": fold_metrics,
                "status": "COMPLETED" if not candidate_failed else "FAILED_PARTIAL"
            }

            cand_summary_path = cand_dir / "loso_summary.json"
            with open(cand_summary_path, "w") as csf:
                json.dump(cand_summary, csf, indent=2)

            progress_data[cand_id] = cand_summary
            with open(progress_file, "w") as pf:
                json.dump(progress_data, pf, indent=2)

            with open(results_jsonl, "a") as rf:
                rf.write(json.dumps(cand_summary) + "\n")

            logger.info(f"=== CANDIDATE {cand_id} ({hash_id}) LOSO RESULT ===")
            logger.info(f"  Mean Val Acc:  {mean_val_acc * 100:.2f}% ± {std_val_acc * 100:.2f}%")
            logger.info(f"  Mean Val Loss: {mean_val_loss:.4f} ± {std_val_loss:.4f}")
            logger.info(f"  Mean Val MAE:  {mean_val_mae:.4f}")

            if mean_val_acc > best_metric_val:
                best_metric_val = mean_val_acc
                best_candidate_id = cand_id

        else:
            # --- TEMPORAL SPLIT EVALUATION ---
            temp_config_path = cand_dir / "config.yaml"
            with open(temp_config_path, "w") as f:
                yaml.dump(cand_params, f, default_flow_style=False)

            metrics_json_path = cand_dir / "metrics.json"
            plot_path = cand_dir / "history.png"
            model_path = cand_dir / "model.h5"

            cmd = [
                sys.executable, "experiments/train_joint_model.py",
                "--config", str(temp_config_path),
                "--metrics-json-path", str(metrics_json_path),
                "--output-model", str(model_path),
                "--output-plot", str(plot_path)
            ]
            if not args.save_plot:
                cmd.append("--no-save-plot")
            if args.force_cpu:
                cmd.append("--force-cpu")

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

                    cand_summary = {
                        "candidate_id": cand_id,
                        "hash_id": hash_id,
                        "parameters": param_registry[cand_id],
                        "metrics": metrics,
                        "status": "COMPLETED"
                    }

                    val_loss = metrics.get("val_loss", float("inf"))
                    val_acc = metrics.get("val_classification_output_accuracy", 0.0)

                    progress_data[cand_id] = cand_summary
                    with open(progress_file, "w") as pf:
                        json.dump(progress_data, pf, indent=2)

                    with open(results_jsonl, "a") as rf:
                        rf.write(json.dumps(cand_summary) + "\n")

                    logger.info(f"[{cand_id}] Complete. Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

                    if val_loss < best_metric_val:
                        best_metric_val = val_loss
                        best_candidate_id = cand_id
                else:
                    logger.error(f"[{cand_id}] Failed with return code {return_code}.")
                    progress_data[cand_id] = {"status": "FAILED"}
                    with open(progress_file, "w") as pf:
                        json.dump(progress_data, pf, indent=2)

            except Exception as e:
                logger.error(f"[{cand_id}] Exception occurred: {e}")
                progress_data[cand_id] = {"status": "ERROR"}
                with open(progress_file, "w") as pf:
                    json.dump(progress_data, pf, indent=2)

    logger.info("=" * 80)
    logger.info("JOINT HYPERPARAMETER GRID SEARCH COMPLETED")
    if is_loso:
        logger.info(f"Best Candidate: {best_candidate_id} with Best Mean Val Accuracy: {best_metric_val * 100:.2f}%")
    else:
        logger.info(f"Best Candidate: {best_candidate_id} with Best Val Loss: {best_metric_val:.6f}")
    if best_candidate_id and best_candidate_id in param_registry:
        logger.info(f"Best Parameters: {param_registry[best_candidate_id]}")
    logger.info("=" * 80)

    # Save final summary file
    with open(summary_file, "w") as sf:
        json.dump({
            "best_candidate_id": best_candidate_id,
            "best_metric_value": best_metric_val,
            "validation_mode": "loso" if is_loso else "temporal",
            "total_candidates": len(grid_candidates),
            "best_parameters": param_registry.get(best_candidate_id, {})
        }, sf, indent=2)


if __name__ == "__main__":
    main()
