import argparse
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Scalogram Generator CLI")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument("--mode", type=str, choices=["batch", "simple"], default="batch", help="Generation mode (batch or simple)")

    args = parser.parse_args()

    # Load configuration
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to read configuration file: {e}")
        exit(1)

    dataset = config.get("dataset")
    if not dataset:
        logger.error("The 'dataset' parameter is missing in the configuration file.")
        exit(1)

    # Route dynamically to the correct module
    if dataset == "DROZY":
        from scalogram_cnn_project.scalogram_generation_drozy.run_generator import run_generation
        run_generation(config, args.mode)
    elif dataset == "SEED-VIG":
        from scalogram_cnn_project.scalogram_generation_seed_vig.run_generator import run_generation
        run_generation(config, args.mode)
    else:
        logger.error(f"Unsupported dataset: {dataset}")
        exit(1)
