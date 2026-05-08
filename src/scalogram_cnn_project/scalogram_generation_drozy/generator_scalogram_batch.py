## We begin by importing the necessary Python modules:

import numpy as np
import mne
import pywt
from scipy.signal import butter, filtfilt
import cv2
import json
from pathlib import Path
import scalogram_cnn_project.settings.config as config
from scalogram_cnn_project.utils.make_hash_id import make_hash_id

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
        sample_file_path = config.OUTPUT_DIR / f'subject1_session1_channelFz' / "samples.jsonl",
        drowsiness_threshold=4,
        cmap="gray",
        freq_min=3, freq_max=30,
        do_resampling = False,
        resample_freq = 128.0,
        epoch_duration = 30.0, 
        ## Determines the overlap between epochs
        overlap_ratio=0.733,
        wavelet_type = 'morl', 
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


        # Normalize power_db between [0, 255]
        vmin = np.percentile(power_db, 20)
        vmax = np.percentile(power_db, 99)
        img = np.clip(power_db, vmin, vmax)
        img = (img - vmin) / (vmax - vmin)
        img = (img * 255).astype(np.uint8)


        # Apply resize
        img_resized = cv2.resize(img, (final_width_px, final_height_px), interpolation=cv2.INTER_CUBIC)
        ## The first axis is for scales, which are in inverse proportion with frequencies. 
        ## Then, we flip first axis.
        img_resized = np.flipud(img_resized)  



        ## Prior to saving the image, we build a dict with all relevant information and create a hash_id from it
        drowsiness_level = 1 if config.drozy_kss_scale[subject][session]>=drowsiness_threshold else 0

        sample_entry = {
            "label": int(drowsiness_level),
            "subject": subject,
            "session": session,
            "epoch": epoch_index,
            "channel": channel
        }


        image_id = make_hash_id(sample_entry)
        sample_entry["image_id"] = image_id
        fig_name = f"{image_id}.png"


        ## Finally, we save the image as gray scale
        cv2.imwrite(str(images_dir / fig_name), img_resized)


        ## And save the in .jsonl file the relation between each hash_id and its metadata
        with open(sample_file_path, "a") as f:
            f.write(json.dumps(sample_entry) + "\n")


        ## Updating the epoch index
        epoch_index +=1



if __name__ == "__main__": 

    generate_scalogram()