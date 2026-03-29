import scalogram_cnn_project.settings.config as config
from scalogram_cnn_project.scalogram_generation.generator_scalogram_batch import generate_scalogram
import json


import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("scalogram_cnn_project").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


OVERLAP = 0.733
CMAP = "gray" # gray 1 channel or viridis 3 channels RGB
SUBJECTS = [1]
SESSIONS = [1, 2, 3]
CHANNELS = ["C3"] #["C3", "C4", "Fz", "Cz", "Pz"]


OUTPUT_FOLDER = "use_less" # f"generated_scalograms_ALL_{CMAP}_overlap{OVERLAP}"
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
    
    for subject in SUBJECTS:
        for session in SESSIONS:
            for channel in CHANNELS:
                
                logger.info(f"Starting {subject}-{session} channel {channel}")
                if config.drozy_valid_tests[subject][session]:
                

                    generate_scalogram(
                        subject = subject, session = session, channel = channel,
                        images_dir = config.OUTPUT_DIR / OUTPUT_FOLDER,
                        sample_file_path = SAMPLE_FILE_PATH,
                        **COMMON_PARAMS
                        )


    with open(DATASET_CONFIG_PATH, "w") as f:
        config_dict = {
            "scalogram"  : COMMON_PARAMS,
            "subjects"   : list(SUBJECTS),
            "channels"   : list(CHANNELS),
            "sessions"   : list(SESSIONS),
            "extra_input": False
        }
        json.dump(config_dict, f, indent=2)


    index = {}

    with open(SAMPLE_FILE_PATH) as f:
        for line in f:
            sample = json.loads(line)
            index[sample["image_id"]] = sample

    with open(INDEX_FILE_PATH, "w") as f:
        json.dump(index, f, indent=2)


    