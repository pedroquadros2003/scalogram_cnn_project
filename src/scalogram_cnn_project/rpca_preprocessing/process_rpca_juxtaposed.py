import os
import cv2
import numpy as np
from pathlib import Path
import logging
import shutil
import json

from scalogram_cnn_project.rpca_preprocessing.rpca import RPCA
import scalogram_cnn_project.settings.config as config
from scalogram_cnn_project.utils.make_hash_id import make_hash_id

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

def process_juxtaposed(input_folder, output_l_folder, output_s_folder, lamb=None, mu=None, tolerance=None, max_iteration=None, cmap="gray"):
    input_path = Path(input_folder)
    output_l_path = Path(output_l_folder)
    output_s_path = Path(output_s_folder)

    output_l_path.mkdir(parents=True, exist_ok=True)
    output_s_path.mkdir(parents=True, exist_ok=True)

    index_file = input_path / "index.json"
    if not index_file.exists():
        logger.error(f"index.json not found in {input_folder}. Cannot group images.")
        return

    with open(index_file, "r") as f:
        index_data = json.load(f)

    # Grouping logic: group by everything except "channel"
    grouped_images = {}
    for img_id, meta in index_data.items():
        # Create a frozenset of items excluding 'channel' to use as a dictionary key
        group_key_dict = {k: v for k, v in meta.items() if k not in ["channel", "image_id"]}
        group_key = frozenset(group_key_dict.items())

        if group_key not in grouped_images:
            grouped_images[group_key] = {"meta": group_key_dict, "channels": []}
        
        grouped_images[group_key]["channels"].append({
            "image_id": img_id,
            "channel_name": meta["channel"]
        })

    new_index_data = {}
    processed_count = 0

    for group_key, group_info in grouped_images.items():
        meta_base = group_info["meta"]
        channels_info = group_info["channels"]

        # Sort channels alphabetically to guarantee consistent concatenation order
        channels_info.sort(key=lambda x: x["channel_name"])

        images_to_concat = []
        valid_group = True

        for ch_info in channels_info:
            img_path = input_path / f"{ch_info['image_id']}.png"
            if not img_path.exists():
                logger.warning(f"Image {img_path.name} not found. Skipping group.")
                valid_group = False
                break
            
            if cmap == "gray":
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                logger.warning(f"Failed to read {img_path.name}. Skipping group.")
                valid_group = False
                break
            
            images_to_concat.append(img)

        if not valid_group or not images_to_concat:
            continue

        # Horizontal concatenation (frequencies remain vertical, time/channels stack horizontally)
        concatenated_img = np.hstack(images_to_concat)

        logger.info(f"Processing Group: Subject {meta_base.get('subject')} | Epoch {meta_base.get('epoch')} | Channels: {[c['channel_name'] for c in channels_info]}")

        if cmap == "gray":
            # Normalize the image to [0, 1] for RPCA
            X = concatenated_img.astype(float) / 255.0

            # Apply RPCA
            L, S = RPCA(X, lamb=lamb, mu=mu, tolerance=tolerance, max_iteration=max_iteration)

            L_img = cv2.normalize(L, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            S_img = cv2.normalize(np.abs(S), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            bgr_channels = cv2.split(concatenated_img)
            L_channels, S_channels = [], []
            for ch in bgr_channels:
                X = ch.astype(float) / 255.0
                L, S = RPCA(X, lamb=lamb, mu=mu, tolerance=tolerance, max_iteration=max_iteration)
                L_channels.append(cv2.normalize(L, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
                S_channels.append(cv2.normalize(np.abs(S), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
            
            L_img = cv2.merge(L_channels)
            S_img = cv2.merge(S_channels)

        # Build new metadata for this concatenated image
        new_meta = dict(meta_base)
        new_meta.pop("image_id", None)  # Remove old ID if present before hashing
        new_meta["channel"] = "merged"

        new_id = make_hash_id(new_meta)
        new_meta["image_id"] = new_id

        cv2.imwrite(str(output_l_path / f"{new_id}.png"), L_img)
        cv2.imwrite(str(output_s_path / f"{new_id}.png"), S_img)

        new_index_data[new_id] = new_meta

        processed_count += 1

    # Save new index.json and copy dataset config
    with open(output_l_path / "index.json", "w") as f:
        json.dump(new_index_data, f, indent=4)
    with open(output_s_path / "index.json", "w") as f:
        json.dump(new_index_data, f, indent=4)

    # Determine the original channels and their order of juxtaposition
    original_channels = sorted(list({meta.get("channel") for meta in index_data.values() if "channel" in meta}))

    dataset_config_path = input_path / "dataset_config.json"
    if dataset_config_path.exists():
        with open(dataset_config_path, "r") as f:
            config_data = json.load(f)
        
        # Update the final width to reflect the horizontal juxtaposition
        if "scalogram" in config_data and "final_width_px" in config_data["scalogram"]:
            config_data["scalogram"]["final_width_px"] *= len(original_channels)

        config_l = dict(config_data)
        config_l["dataset"] = config_data.get("dataset", "DROZY")
        config_l["preprocessing"] = "rpca_juxtaposed"
        config_l["rpca_component"] = "L"
        config_l["channels"] = original_channels
        with open(output_l_path / "dataset_config.json", "w") as f:
            json.dump(config_l, f, indent=4)

        config_s = dict(config_data)
        config_s["dataset"] = config_data.get("dataset", "DROZY")
        config_s["preprocessing"] = "rpca_juxtaposed"
        config_s["rpca_component"] = "S"
        config_s["channels"] = original_channels
        with open(output_s_path / "dataset_config.json", "w") as f:
            json.dump(config_s, f, indent=4)

    logger.info(f"Done! Formed {processed_count} juxtaposed images.")