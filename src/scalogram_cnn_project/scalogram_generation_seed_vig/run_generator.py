import json
import logging
import os
import numpy as np
from pathlib import Path
import scalogram_cnn_project.settings.config as project_config

logger = logging.getLogger(__name__)

def run_generation(config, mode):
    scalogram_params = config.get("scalogram_params", {})

    if mode == "simple":
        from scalogram_cnn_project.scalogram_generation_seed_vig.generator_scalogram_simple import generate_scalogram
        
        # Merge common and simple parameters
        params = {}
        params.update(scalogram_params)
        params.update(config.get("simple_params", {}))
        
        # Extract keys needed by simple generator
        generate_scalogram(
            subject=params.get("subject"),
            channel=params.get("channel"),
            epoch_index=params.get("epoch_index"),
            epoch_duration=params.get("epoch_duration"),
            wavelet_type=params.get("wavelet_type"),
            freq_min=params.get("freq_min"),
            freq_max=params.get("freq_max"),
            cmap=params.get("cmap"),
            width_px=params.get("width_px"),
            height_px=params.get("height_px"),
            dpi=params.get("dpi"),
            show_bands=params.get("show_bands", True),
            final_width_px=params.get("final_width_px"),
            final_height_px=params.get("final_height_px")
        )
        logger.info("Finished simple scalogram generation.")
        
    elif mode == "batch":
        output_folder = config.get("output_folder")
        if not output_folder:
            raise ValueError("The 'output_folder' parameter is required for batch mode.")

        # Resolve directories
        output_dir = project_config.OUTPUT_DIR / output_folder
        output_dir.mkdir(parents=True, exist_ok=True)

        sample_file_path = output_dir / "samples.jsonl"
        index_file_path = output_dir / "index.json"
        dataset_config_path = output_dir / "dataset_config.json"

        # Delete old samples file if starting fresh in batch mode
        if sample_file_path.exists():
            sample_file_path.unlink()

        subjects = config.get("subjects", [])
        channels = config.get("channels", [])
        extra_input = config.get("extra_input", False)

        if extra_input:
            from scalogram_cnn_project.scalogram_generation_seed_vig.generator_scalogram_batch_and_biomarkers import generate_scalogram as generate_scalogram_biomarkers
            
            subject_map = {s: i+1 for i, s in enumerate(subjects)}
            n_subjects = len(subjects) + 1
            n_channels = len(channels) + 1
            data = None

            for subject in subjects:
                for ch_idx, channel in enumerate(channels, start=1):
                    logger.info(f"Generating SEED-VIG {subject} | {channel}")
                    feature_np = generate_scalogram_biomarkers(
                        subject=subject,
                        channel=channel,
                        images_dir=output_dir,
                        sample_file_path=sample_file_path,
                        **scalogram_params
                    )

                    # Initialize tensor after the first generated features
                    if data is None:
                        n_epochs, n_features = feature_np.shape
                        data = np.zeros((
                            n_subjects,
                            n_channels,
                            n_epochs,
                            n_features
                        ), dtype=np.float32)

                    subject_idx = subject_map[subject]
                    data[subject_idx, ch_idx] = feature_np

            # Save extra features file
            output_data_path = output_dir / "data.npy"
            logger.info(f"Saving feature tensor with shape: {data.shape}")
            np.save(output_data_path, data)

        else:
            from scalogram_cnn_project.scalogram_generation_seed_vig.generator_scalogram_batch import generate_scalogram as generate_scalogram_batch
            
            for subject in subjects:
                for channel in channels:
                    logger.info(f"Generating SEED-VIG {subject} | {channel}")
                    generate_scalogram_batch(
                        subject=subject,
                        channel=channel,
                        images_dir=output_dir,
                        sample_file_path=sample_file_path,
                        **scalogram_params
                    )

        # Save dataset configuration
        with open(dataset_config_path, "w") as f:
            config_dict = {
                "dataset": "SEED-VIG",
                "scalogram": scalogram_params,
                "subjects": list(subjects),
                "channels": list(channels),
                "rpca_preprocessing": "none",
                "extra_input": extra_input
            }
            json.dump(config_dict, f, indent=2)

        # Build image index from jsonl
        index = {}
        if sample_file_path.exists():
            with open(sample_file_path, "r") as f:
                for line in f:
                    if line.strip():
                        sample = json.loads(line)
                        index[sample["image_id"]] = sample

        with open(index_file_path, "w") as f:
            json.dump(index, f, indent=2)

        logger.info("Finished batch dataset generation.")
