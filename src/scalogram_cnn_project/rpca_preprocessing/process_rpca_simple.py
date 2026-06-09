import os
import cv2
import numpy as np
from pathlib import Path
import logging
import shutil

from scalogram_cnn_project.rpca_preprocessing.rpca import RPCA
import scalogram_cnn_project.settings.config as config

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

def process_single_image(image_path, output_folder, lamb=None, mu=None, tolerance=None, max_iteration=None, cmap="gray"):
    input_path = Path(image_path)
    output_path = Path(output_folder)

    # Create output directory if it does not exist
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.is_file() or input_path.suffix.lower() != ".png":
        logger.error(f"The file {image_path} is not a valid .png image or does not exist.")
        return

    logger.info(f"Processing {input_path.name}...")
    
    if cmap == "gray":
        # Read the image in grayscale (expected format for 2D matrices)
        img = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.error(f"Failed to read {input_path.name}")
            return

        # Normalize the image to the [0, 1] interval for RPCA stability
        X = img.astype(float) / 255.0

        # Apply RPCA
        L, S = RPCA(X, lamb=lamb, mu=mu, tolerance=tolerance, max_iteration=max_iteration)

        # Convert matrices back to 8-bit images (0 to 255)
        # For S, we take the absolute value as it represents sparse differences/noise
        L_img = cv2.normalize(L, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        S_img = cv2.normalize(np.abs(S), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if img is None:
            logger.error(f"Failed to read {input_path.name}")
            return

        bgr_channels = cv2.split(img)
        L_channels, S_channels = [], []
        for ch in bgr_channels:
            X = ch.astype(float) / 255.0
            L, S = RPCA(X, lamb=lamb, mu=mu, tolerance=tolerance, max_iteration=max_iteration)
            L_channels.append(cv2.normalize(L, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
            S_channels.append(cv2.normalize(np.abs(S), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
            
        L_img = cv2.merge(L_channels)
        S_img = cv2.merge(S_channels)

    # Create a prefix string if lambda is provided
    lamb_prefix = f"lamb{lamb}_" if lamb is not None else ""

    # Save the generated images in the output directory
    cv2.imwrite(str(output_path / f"L_{lamb_prefix}{input_path.name}"), L_img)
    cv2.imwrite(str(output_path / f"S_{lamb_prefix}{input_path.name}"), S_img)

    # Copy the original image for comparison
    orig_copy_path = output_path / f"Original_{input_path.name}"
    if not orig_copy_path.exists():
        shutil.copy2(input_path, orig_copy_path)

    logger.info(f"Done! L and S images (lambda={lamb}) saved in {output_folder}")
