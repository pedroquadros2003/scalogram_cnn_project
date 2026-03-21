import scalogram_cnn_project.settings.config as config
from scalogram_cnn_project.scalogram_generation.generator_scalogram_batch_v0 import generate_scalogram

import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("scalogram_cnn_project").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


SEED = 42
OVERLAP = 0.733
CMAP = "gray" # gray 1 channel or viridis 3 channels RGB
SUBJECTS = range(1,15)
SESSIONS = range(1, 4)
CHANNELS = ["C3", "C4", "Fz", "Cz", "Pz"]

channel_string = "".join(CHANNELS)

OUTPUT_FOLDER = f"generated_scalograms_ALL_{CMAP}_overlap{OVERLAP}"



if __name__ == "__main__": 
    drowsiness_threshold = 4


    
    for subject in SUBJECTS:
        for session in SESSIONS:
            for channel in CHANNELS:
                
                logger.info(f"Starting {subject}-{session} channel {channel}")
                if config.drozy_valid_tests[subject][session]:
                
                    drowsiness_level = 1 if config.drozy_kss_scale[subject][session]>=drowsiness_threshold else 0

                    generate_scalogram(
                        subject = subject, session = session, channel = channel,
                        images_dir = config.OUTPUT_DIR / OUTPUT_FOLDER,
                        drowsiness_threshold=drowsiness_threshold,
                        freq_min=3, freq_max=30,
                        do_resampling=True,
                        resample_freq=128.0,
                        epoch_duration = 30.0, 
                        overlap_ratio=OVERLAP,
                        wavelet_type = 'cmor1.5-2.5',
                        cmap=CMAP,
                        width_px = 662,  
                        height_px = 536,
                        dpi = 100,
                        final_width_px = 64,
                        final_height_px = 64,
                        )


