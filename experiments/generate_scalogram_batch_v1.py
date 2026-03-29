import scalogram_cnn_project.settings.config as config
from scalogram_cnn_project.scalogram_generation.generator_scalogram_batch_and_biomarkers import generate_scalogram_and_biomarkers
import json
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("scalogram_cnn_project").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


OVERLAP = 0.733
CMAP = "gray" # gray 1 channel or viridis 3 channels RGB
SUBJECTS = [6] # range(1,15)
SESSIONS = [1, 2, 3]
CHANNELS = ["C3"] # ["C3", "C4", "Cz", "Fz", "Pz"]


OUTPUT_FOLDER = "use_less" # f"generated_scalograms_ALL_{CMAP}_overlap{OVERLAP}_subject2"
SAMPLE_FILE_PATH = config.OUTPUT_DIR / OUTPUT_FOLDER / "samples.jsonl"
INDEX_FILE_PATH = config.OUTPUT_DIR / OUTPUT_FOLDER / "index.json" 
DATASET_CONFIG_PATH = config.OUTPUT_DIR / OUTPUT_FOLDER / "dataset_config.json"


COMMON_PARAMS = dict(
    freq_min=3,
    freq_max=30,
    do_resampling=True,
    resample_freq=128.0,
    epoch_duration=30.0,
    overlap_ratio=OVERLAP,
    wavelet_type='cmor1.5-2.5',
    cmap=CMAP,
    final_width_px=64,
    final_height_px=64,
    drowsiness_threshold=4,
)



if __name__ == "__main__": 

    subject_map = {s: i+1 for i, s in enumerate(SUBJECTS)}

    # dimensions
    N_SUBJECTS = len(SUBJECTS) + 1
    N_SESSIONS = len(SESSIONS) + 1
    N_CHANNELS = len(CHANNELS) + 1  # +1 dummy

    data = None  # we are going to initialize it later

    for subject in SUBJECTS:

        for session in SESSIONS:

            if not config.drozy_valid_tests[subject][session]:
                continue

            for ch_idx, channel in enumerate(CHANNELS, start=1):

                logger.info(f"Starting {subject}-{session} channel {channel}")

                feature_np = generate_scalogram_and_biomarkers(
                    subject=subject, session=session, channel=channel,
                    images_dir=config.OUTPUT_DIR / OUTPUT_FOLDER,
                    sample_file_path = SAMPLE_FILE_PATH,
                    **COMMON_PARAMS
                )

                # finally, initialize data array
                if data is None:
                    n_epochs, n_features = feature_np.shape

                    data = np.zeros((
                        N_SUBJECTS,
                        N_SESSIONS,
                        N_CHANNELS,
                        n_epochs,
                        n_features
                    ), dtype=np.float32)

                # fill the data array
                subject_idx = subject_map[subject]
                data[subject_idx, session, ch_idx] = feature_np


    ## Save extra features file
    output_data_path = config.OUTPUT_DIR / OUTPUT_FOLDER / "data.npy"
    logger.info("The shape of the feature numpy array is %s", data.shape)
    np.save(output_data_path, data)


    with open(DATASET_CONFIG_PATH, "w") as f:
        config_dict = {
            "scalogram"  : COMMON_PARAMS,
            "subjects"   : list(SUBJECTS),
            "channels"   : list(CHANNELS),
            "sessions"   : list(SESSIONS),
            "extra_input": True
        }
        json.dump(config_dict, f, indent=2)


    index = {}

    with open(SAMPLE_FILE_PATH) as f:
        for line in f:
            sample = json.loads(line)
            index[sample["image_id"]] = sample

    with open(INDEX_FILE_PATH, "w") as f:
        json.dump(index, f, indent=2)
