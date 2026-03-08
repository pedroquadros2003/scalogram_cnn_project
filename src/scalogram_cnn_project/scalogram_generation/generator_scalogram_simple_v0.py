import os
import cv2
import time
import mne
import pywt
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from pathlib import Path
import scalogram_cnn_project.settings.config as config


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

def generate_scalogram(subject=11,
                       session=3, 
                       channel="C3", 
                       epoch_index=10,
                       wavelet_type = 'morl',
                       epoch_duration = 30.0,
                       freq_min=3,
                       freq_max=30,
                       do_resampling = False, 
                       resample_freq = 128.0,
                       drowsiness_threshold=4,
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
    
    # 1. Load Data
    sample_data_folder = config.DROZY_DIR
    sample_data_raw_file = sample_data_folder / "psg" / f"{subject}-{session}.edf"
    
    print(f"Loading: {sample_data_raw_file}")
    raw = mne.io.read_raw_edf(sample_data_raw_file, preload=True)
    if do_resampling: raw.resample(resample_freq)

    raw_signal = raw.pick(picks=channel).get_data()
    sfreq = raw.info["sfreq"]
    del raw

    # 2. Preprocessing (Filtering)
    # Apply bandpass filter to the first channel (Cz)
    filtered_signal = butter_bandpass_filter(raw_signal[0, :], freq_min, freq_max, sfreq, order=4)
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
    power_db = 10 * np.log10(power + 1e-9)
    
    ## Then, after adjusting the color scale, we plot the scalogram
    vmin = power_db.min()
    vmax = power_db.max()
    
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    # Adjust axes to fill the figure without labels if desired, 
    # or use standard layouts to show axes as discussed previously.
    ax = fig.add_axes([0, 0, 1, 1]) 
    ax.axis('off') # Turn off axis for CNN input images
    
    time = np.linspace(0, epoch_duration, power_db.shape[1])
    ax.pcolormesh(time, freqs, power_db, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)


    # Visual Markers
    if show_bands:
        ax.axhline(4, color='white', linestyle='--', alpha=0.5)
        ax.axhline(8, color='white', linestyle='--', alpha=0.5)
        ax.axhline(13, color='white', linestyle='--', alpha=0.5)
        ax.text(1, 6, 'Theta', color='white', fontweight='bold')
        ax.text(1, 10, 'Alpha', color='white', fontweight='bold')
        ax.text(1, 20, 'Beta', color='white', fontweight='bold')

    drowsiness_level = 1 if config.drozy_kss_scale[subject][session]>=drowsiness_threshold else 0

    fig_name = f'drownsinessLevel{drowsiness_level}_subject{subject}_session{session}_channel{channel}_epoch{epoch_index}.png'
    save_path = config.OUTPUT_DIR / fig_name
    
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    
    # 6. Post-processing (Resize via OpenCV)
    image = cv2.imread(str(save_path))
    resized_image = cv2.resize(image, (final_width_px, final_height_px), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(save_path), resized_image)
    
    print(f"Scalogram saved and resized at: {save_path}")



if __name__ == "__main__":
    
    generate_scalogram()