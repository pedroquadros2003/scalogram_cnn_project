import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import scalogram_cnn_project.settings.config as config

from scalogram_cnn_project.scalogram_generation_seed_vig.generator_scalogram_batch_and_biomarkers import (
    generate_scalogram
)


CMAP = "gray"

CHANNELS = ['FT8', 'T7']   #['FT8','T7','T8','TP7','TP8','CP1','CP2','P1','PZ','P2','PO3','POZ','PO4','O1','OZ','O2']
SUBJECTS = [1,3]     #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] 

OUTPUT_FOLDER = "seedvig_scalograms"

SAMPLE_FILE_PATH = config.OUTPUT_DIR / OUTPUT_FOLDER / "samples.jsonl"
INDEX_FILE_PATH = config.OUTPUT_DIR / OUTPUT_FOLDER / "index.json"
DATASET_CONFIG_PATH = config.OUTPUT_DIR / OUTPUT_FOLDER / "dataset_config.json"


COMMON_PARAMS = dict(
    freq_min=3,
    freq_max=30,
    epoch_duration=8.0,
    wavelet_type='morl',
    cmap=CMAP,
    final_width_px=64,
    final_height_px=64,
)


if __name__ == "__main__":

    OUTPUT_PATH = config.OUTPUT_DIR / OUTPUT_FOLDER
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Delete old samples file
    if SAMPLE_FILE_PATH.exists():
        SAMPLE_FILE_PATH.unlink()

    subject_map = {s: i+1 for i, s in enumerate(SUBJECTS)}

    N_SUBJECTS = len(SUBJECTS) + 1
    N_CHANNELS = len(CHANNELS) + 1

    data = None

    for subject in SUBJECTS:

        for ch_idx, channel in enumerate(CHANNELS, start=1):

            logger.info(f"Generating {subject} | {channel}")

            feature_np = generate_scalogram(
                subject=subject,
                channel=channel,

                images_dir=OUTPUT_PATH,

                sample_file_path=SAMPLE_FILE_PATH,

                **COMMON_PARAMS
            )

            # initialize tensor after first sample
            if data is None:

                n_epochs, n_features = feature_np.shape

                data = np.zeros(
                    (
                        N_SUBJECTS,
                        N_CHANNELS,
                        n_epochs,
                        n_features
                    ),
                    dtype=np.float32
                )

            subject_idx = subject_map[subject]

            data[subject_idx, ch_idx] = feature_np


    # Save feature tensor
    output_data_path = OUTPUT_PATH / "data.npy"

    logger.info("Feature tensor shape: %s", data.shape)

    ## Numpy array format: (subjects, channels, epochs, features)
    np.save(output_data_path, data)


    # Save dataset configuration
    with open(DATASET_CONFIG_PATH, "w") as f:

        config_dict = {
            "dataset": "SEED-VIG",
            "scalogram": COMMON_PARAMS,
            "channels": CHANNELS,
            "subjects": SUBJECTS,
            "rpca_preprocessing" : "none",
            "extra_input": True
        }

        json.dump(config_dict, f, indent=2)


    # Build image index
    index = {}

    with open(SAMPLE_FILE_PATH) as f:

        for line in f:

            sample = json.loads(line)

            index[sample["image_id"]] = sample

    with open(INDEX_FILE_PATH, "w") as f:
        json.dump(index, f, indent=2)

    logger.info("Finished dataset generation.")