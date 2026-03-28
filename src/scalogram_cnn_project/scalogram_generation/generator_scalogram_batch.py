## We begin by importing the necessary Python modules:

import numpy as np
import mne
import pywt
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

from pathlib import Path
import scalogram_cnn_project.settings.config as config

import logging
logger = logging.getLogger(__name__)


## Defining the Butterworth filter

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)



def generate_scalogram(
        subject = 1, session = 1, channel = "Fz",
        images_dir = config.OUTPUT_DIR / f'subject1_session1_channelFz',
        drowsiness_threshold=4,
        cmap="viridis",
        freq_min=3, freq_max=30,
        do_resampling = False,
        resample_freq = 128.0,
        epoch_duration = 30.0, 
        ## Determines the overlap between epochs
        overlap_ratio=0.733,
        wavelet_type = 'morl', 
        ## Size of the first scalogram generated, according to A. Zayed (2025)
        width_px = 662,  
        height_px = 536,
        dpi = 100,
        ## Final sized of the scalogram, designed to be input of a CNN-2D
        final_width_px = 64,
        final_height_px = 64,
):

    ## Importing the edf file
    sample_data_folder = config.DROZY_DIR
    sample_data_raw_file = (
        sample_data_folder / "psg" / f"{subject}-{session}.edf"
    )
    raw = mne.io.read_raw_edf(sample_data_raw_file)
    if do_resampling: raw.resample(resample_freq)

    raw_signal = raw.pick(picks = channel).get_data()
    sfreq = raw.info["sfreq"]
    tot_samples = raw.n_times
    del raw



    # Applying the filter to the data
    filtered_signal = butter_bandpass_filter(raw_signal[0,:],freq_min, freq_max, sfreq, order=4)
    del raw_signal

    ## Creating a directory for saving the images
    Path(images_dir).mkdir(parents=True, exist_ok=True)


    ## Converting epoch_duration and step_duration to number of samples
    epoch_sample_duration = int(epoch_duration * sfreq)
    step_sample_duration = int(epoch_duration * (1 - overlap_ratio) * sfreq)


    epoch_index = int(0)

    while  epoch_index*step_sample_duration + epoch_sample_duration < tot_samples :
        logger.info(f"{epoch_index*step_sample_duration + epoch_sample_duration} < {tot_samples}?")

        ## Now, we compute the CWT coefficients in dB
        freqs = np.linspace(freq_min, freq_max, 256)
        scales = pywt.frequency2scale(wavelet_type, freqs * 1/sfreq)
        window = filtered_signal[ epoch_index*step_sample_duration : epoch_index*step_sample_duration  + epoch_sample_duration]


        coef, _ = pywt.cwt(window, scales, wavelet_type, sampling_period=1/sfreq)
        power = np.abs(coef)**2
        power_db = 10 * np.log10(power + 1e-9) 


        ## Then, after adjusting the color scale, we plot the scalogram
        vmin = power_db.min()   
        vmax = power_db.max()


        fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])


        time = np.linspace(0, epoch_duration, power_db.shape[1])
        pcm = ax.pcolormesh(
            time,
            freqs,
            power_db,
            shading='auto',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )

        ## Next, we run a command that reduces the plot image to just the 
        ## bounded box, which is better for processing the image with a CNN-2D:

        extent = ax.get_window_extent().transformed(
            fig.dpi_scale_trans.inverted()
        )

        drowsiness_level = 1 if config.drozy_kss_scale[subject][session]>=drowsiness_threshold else 0

        fig_name = f'drownsinessLevel{drowsiness_level}_subject{subject}_session{session}_channel{channel}_epoch{epoch_index}.png'
        fig.savefig(
            images_dir / fig_name,
            bbox_inches=extent,
            dpi=dpi
        )

        plt.close()

        ## Finally, we resize the scalograms via cubic interpolation to size of the CNN-2D input

        image_path = images_dir / fig_name
        image = cv2.imread(image_path)

        resized_image = cv2.resize(src = image, dsize= (final_width_px, final_height_px), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(image_path, resized_image)


        ## Updating the epoch index

        epoch_index +=1



if __name__ == "__main__": 

    generate_scalogram()