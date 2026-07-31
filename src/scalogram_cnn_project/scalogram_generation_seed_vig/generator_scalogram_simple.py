import os
import cv2
import pywt
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from pathlib import Path
import scalogram_cnn_project.settings.config as config

import logging
logger = logging.getLogger(__name__)

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

def generate_scalogram(subject = 1,
                       channel="O1", 
                       epoch_index=10,
                       epoch_duration=8.0, # in seconds
                       wavelet_type = 'morl',
                       freq_min=3,
                       freq_max=30,
                       cmap="viridis",
                       ## Size of the first scalogram generated, according to A. Zayed (2025)
                       width_px = 662,  
                       height_px = 536,
                       dpi = 100,
                       show_bands = True,
                       ## Final sized of the scalogram, designed to be input of a CNN-2D
                       final_width_px = 256,
                       final_height_px = 256,
                       ):
    
    
    subject_file  = config.seed_vig_filenames[subject]

    from scalogram_cnn_project.utils.signal_loader import SignalLoader

    # 1. Load Data
    sample_data_folder = config.SEED_VIG_DIR
    sample_data_raw_file = sample_data_folder / subject_file
    
    logger.info(f"Loading: {sample_data_raw_file}")
    signal_data = SignalLoader.load_seed_vig(sample_data_raw_file)
    sfreq = signal_data.sfreq

    # 2. Preprocessing (Channel Selection & Filtering)
    try:
        raw_signal = signal_data.get_channel_signal(channel)
    except ValueError:
        logger.warning(f"Channel '{channel}' not found in SEED-VIG channel list. Defaulting to first channel.")
        raw_signal = signal_data.data[0]
    del signal_data

    # Apply bandpass filter to the selected channel
    filtered_signal = butter_bandpass_filter(raw_signal, freq_min, freq_max, sfreq, order=4)
    del raw_signal

    # 3. Epoching
    epoch_sample_duration = int(epoch_duration * sfreq)
    start_sample = int(epoch_index * epoch_duration * sfreq)
    end_sample = start_sample + epoch_sample_duration
    
    window = filtered_signal[start_sample:end_sample]
    del filtered_signal
    
    # 4. Continuous Wavelet Transform (CWT)
    freqs = np.linspace(freq_min, freq_max, 256)
    scales = pywt.frequency2scale(wavelet_type, freqs * (1/sfreq))
    
    coef, _ = pywt.cwt(window, scales, wavelet_type, sampling_period=1/sfreq)
    power = np.abs(coef)**2
     
    ## Then, after adjusting the color scale, we plot the scalogram
    vmin = np.percentile(power, 20)
    vmax = np.percentile(power, 99)
    
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    # Adjust axes to fill the figure without labels if desired, 
    # or use standard layouts to show axes as discussed previously.
    ax = fig.add_axes([0, 0, 1, 1]) 
    ax.axis('off') # Turn off axis for CNN input images
    
    time = np.linspace(0, epoch_duration, power.shape[1])
    ax.pcolormesh(time, freqs, power, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)


    # Visual Markers
    if show_bands:
        ax.axhline(4, color='white', linestyle='--', alpha=0.5)
        ax.axhline(8, color='white', linestyle='--', alpha=0.5)
        ax.axhline(13, color='white', linestyle='--', alpha=0.5)
        ax.text(1, 6, 'Theta', color='white', fontweight='bold')
        ax.text(1, 10, 'Alpha', color='white', fontweight='bold')
        ax.text(1, 20, 'Beta', color='white', fontweight='bold')


    # A corrected filename generation is needed for SEED-VIG data.
    fig_name = f'subject{subject}_channel{channel}_epoch{epoch_index}.png'
    save_path = config.OUTPUT_DIR / fig_name
    
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    
    # 6. Post-processing (Resize via OpenCV)
    image = cv2.imread(str(save_path))
    resized_image = cv2.resize(image, (final_width_px, final_height_px), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(save_path), resized_image)
    
    logger.info(f"Scalogram saved and resized at: {save_path}")



if __name__ == "__main__":
    
    generate_scalogram()