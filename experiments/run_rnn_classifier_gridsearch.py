import argparse
import logging
import os
import sys
import json
import itertools
import subprocess
from pathlib import Path
import yaml

from scalogram_cnn_project.utils.dict_product import dict_product
from scalogram_cnn_project.utils.simplify_config_space import simplify_config_space
import scalogram_cnn_project.settings.config as config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run Grid Search for Coupled RNN-MLP Classifier")
    parser.add_argument("--output_folder", type=str, default="rnn_classifier_gridsearch", help="Output folder name inside OUTPUT_DIR")
    parser.add_argument("--params_file", type=str, default="configs/hyperparameter_search_rnn/classifier_gridsearch_example.yaml", help="YAML search space configuration file path")
    parser.add_argument("--force-cpu", action="store_true", help="Force candidate executions to run on CPU")
    
    args = parser.parse_args()
    
    params_file_path = Path(args.params_file)
    if not params_file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {params_file_path}")
        
    with open(params_file_path, "r") as f:
        search_config = yaml.safe_load(f)
        
    model_hp = simplify_config_space(search_config.get("MODEL_HYPER_PARAMS", {}))
    train_hp = simplify_config_space(search_config.get("MODEL_TRAIN_PARAMS", {}))
    
    # Generate Cartesian product of configurations
    model_configs = list(dict_product(model_hp))
    train_configs = list(dict_product(train_hp))
    
    logger.info(f"Generated {len(model_configs)} model configs and {len(train_configs)} train configs.")
    
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
    
    progress_file = output_dir / "progress.json"
    registry_file = output_dir / "param_registry.json"
    
    # Save parameter registry
    with open(registry_file, "w") as f:
        json.dump(param_registry, f, indent=2)
        
    results = {}
    if progress_file.exists():
        with open(progress_file, "r") as f:
            results = json.load(f)
        logger.info(f"Resuming search. {len(results)} candidates already completed.")
        
    best_acc = -1.0
    best_cand = None
    
    for cand in grid_candidates:
        cand_id = cand["candidate_id"]
        if cand_id in results:
            logger.info(f"Skipping {cand_id} (already evaluated)")
            val_acc = results[cand_id].get("val_accuracy") if isinstance(results[cand_id], dict) else results[cand_id]
            if val_acc is not None and val_acc > best_acc:
                best_acc = val_acc
                best_cand = cand_id
            continue
            
        logger.info(f"Evaluating candidate {cand_id}/{len(grid_candidates)-1}...")
        
        # Write candidate config to temporary file
        temp_config_path = output_dir / f"temp_{cand_id}.yaml"
        with open(temp_config_path, "w") as f:
            yaml.dump(param_registry[cand_id], f, default_flow_style=False)
            
        # Target metrics JSON file
        metrics_json_path = output_dir / f"metrics_{cand_id}.json"
        
        # Subprocess command
        cmd = [
            sys.executable, "experiments/train_rnn_classifier.py",
            "--config", str(temp_config_path),
            "--metrics-json-path", str(metrics_json_path),
            # Save weights uniquely per candidate
            "--output-model", str(output_dir / f"combined_predict_classifier_{cand_id}.h5")
        ]
        if args.force_cpu:
            cmd.append("--force-cpu")
            
        env = os.environ.copy()
        # Set PYTHONPATH so the subprocess can import src
        workspace_root = Path(__file__).parent.parent
        env["PYTHONPATH"] = str(workspace_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
        
        try:
            res = subprocess.run(cmd, env=env)
            if res.returncode == 0 and metrics_json_path.exists():
                with open(metrics_json_path, "r") as f:
                    metrics = json.load(f)
                val_acc = metrics.get("val_accuracy", -1.0)
                results[cand_id] = metrics
                logger.info(f"Candidate {cand_id} completed. Metrics: {metrics}")
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_cand = cand_id
            else:
                logger.error(f"Candidate {cand_id} failed with exit code: {res.returncode}")
                results[cand_id] = "FAILED"
        except Exception as e:
            logger.error(f"Error executing candidate {cand_id}: {e}")
            results[cand_id] = "FAILED"
        finally:
            # Cleanup temporary config and metrics files
            if temp_config_path.exists():
                temp_config_path.unlink()
            if metrics_json_path.exists():
                metrics_json_path.unlink()
                
        # Save progress immediately
        with open(progress_file, "w") as f:
            json.dump(results, f, indent=2)
            
    logger.info("Grid Search Completed.")
    logger.info(f"Best Candidate: {best_cand} with Val Accuracy: {best_acc:.6f}")
    if best_cand:
        logger.info(f"Best Params: {param_registry[best_cand]}")

if __name__ == "__main__":
    main()
