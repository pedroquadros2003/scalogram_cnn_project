import os
import cv2
import numpy as np
from pathlib import Path
import logging
import shutil
import json

from scalogram_cnn_project.rpca_preprocessing.rpca import RPCA
import scalogram_cnn_project.settings.config as config


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

def process_folder(input_folder, output_l_folder, output_s_folder, lamb=None, mu=None, tolerance=None, max_iteration=None, cmap="gray"):
    input_path = Path(input_folder)
    output_l_path = Path(output_l_folder)
    output_s_path = Path(output_s_folder)

    # Create output directories if they do not exist
    output_l_path.mkdir(parents=True, exist_ok=True)
    output_s_path.mkdir(parents=True, exist_ok=True)

    # Find all files in the input folder
    all_files = [f for f in input_path.iterdir() if f.is_file()]
    png_files = [f for f in all_files if f.suffix.lower() == ".png"]
    non_png_files = [f for f in all_files if f.suffix.lower() != ".png"]

    # Copy non-PNG files to both output directories
    for file_path in non_png_files:
        if file_path.name == "dataset_config.json":
            logger.info("Updating dataset_config.json for L and S...")
            with open(file_path, "r") as f:
                config_data = json.load(f)
            
            config_l = dict(config_data)
            config_l["dataset"] = config_data.get("dataset", "DROZY")
            config_l["preprocessing"] = "rpca_isolated"
            config_l["rpca_component"] = "L"
            with open(output_l_path / "dataset_config.json", "w") as f:
                json.dump(config_l, f, indent=4)

            config_s = dict(config_data)
            config_s["dataset"] = config_data.get("dataset", "DROZY")
            config_s["preprocessing"] = "rpca_isolated"
            config_s["rpca_component"] = "S"
            with open(output_s_path / "dataset_config.json", "w") as f:
                json.dump(config_s, f, indent=4)
        else:
            logger.info(f"Copying {file_path.name}...")
            shutil.copy2(file_path, output_l_path / file_path.name)
            shutil.copy2(file_path, output_s_path / file_path.name)

    if not png_files:
        logger.warning(f"No .png images found in {input_folder}")
        return

    for img_path in png_files:
        logger.info(f"Processing {img_path.name}...")
        
        if cmap == "gray":
            # Read the image in grayscale (expected format for 2D matrices)
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.error(f"Failed to read {img_path.name}")
                continue

            # Normalize the image to the [0, 1] interval for RPCA stability
            X = img.astype(float) / 255.0

            # Apply RPCA
            L, S = RPCA(X, lamb=lamb, mu=mu, tolerance=tolerance, max_iteration=max_iteration)

            # Convert matrices back to 8-bit images (0 to 255)
            # For S, we take the absolute value as it represents sparse differences/noise
            L_img = cv2.normalize(L, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            S_img = cv2.normalize(np.abs(S), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                logger.error(f"Failed to read {img_path.name}")
                continue

            bgr_channels = cv2.split(img)
            L_channels, S_channels = [], []
            for ch in bgr_channels:
                X = ch.astype(float) / 255.0
                L, S = RPCA(X, lamb=lamb, mu=mu, tolerance=tolerance, max_iteration=max_iteration)
                L_channels.append(cv2.normalize(L, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
                S_channels.append(cv2.normalize(np.abs(S), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

            L_img = cv2.merge(L_channels)
            S_img = cv2.merge(S_channels)

        # Save the generated images in their respective directories
        cv2.imwrite(str(output_l_path / img_path.name), L_img)
        cv2.imwrite(str(output_s_path / img_path.name), S_img)

    logger.info(f"Done! Processed {len(png_files)} images.")