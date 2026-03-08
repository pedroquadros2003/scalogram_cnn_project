import scalogram_cnn_project.settings.config as config
import time
from scalogram_cnn_project.scalogram_generation.generator_scalogram_batch_v0 import generate_scalogram

SEED = 42
OVERLAP = 0.85
CMAP = "gray" # gray 1 channel or viridis 3 channels RGB
OUTPUT_FOLDER = f"generated_scalograms_C3C4_{CMAP}_overlap_{OVERLAP}"

if __name__ == "__main__": 
    drowsiness_threshold = 4

    subjects = range(1,15)
    sessions = range(1, 4)
    channels = ["C3", "C4"] # +  ["Fz", "Cz", "Pz"]

    start_time = time.perf_counter()
    
    for subject in subjects:
        for session in sessions:
            for channel in channels:
                
                print(f"Starting {subject}-{session} channel {channel}")
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
                        verbose = False
                    )


    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds\n\n")
    
