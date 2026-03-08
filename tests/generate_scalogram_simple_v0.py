from scalogram_cnn_project.scalogram_generation.generator_scalogram_simple_v0 import generate_scalogram
import time

import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("scalogram_cnn_project").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    start_time = time.perf_counter()
    # Run the generator
    generate_scalogram(subject=1,
                       session=1, 
                       channel="C3", 
                       epoch_index=10,
                       wavelet_type = 'cmor1.5-2.5',
                       show_bands = True,
                       epoch_duration = 30.0,
                       freq_min=3,
                       freq_max=30,
                       do_resampling = False, 
                       resample_freq = 128.0,
                       drowsiness_threshold=4,
                       cmap="gray", # gray 1 channel or viridis 3 channels RGB
                       width_px = 662,  
                       height_px = 536,
                       dpi = 100,
                       final_width_px = 256,
                       final_height_px = 256,
                       )
    
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    logger.info(f"Elapsed time: {elapsed_time:.4f} seconds")