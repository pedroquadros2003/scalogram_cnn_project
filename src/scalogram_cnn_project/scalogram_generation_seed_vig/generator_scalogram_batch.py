## We begin by importing the necessary Python modules:

import numpy as np
from scipy.io import loadmat
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
        subject = 1,
        channel = "O1",
        images_dir = config.OUTPUT_DIR / f'subject1_session1_channelFz',
        sample_file_path = config.OUTPUT_DIR / f'subject1_session1_channelFz' / "samples.jsonl",
        cmap="gray",
        freq_min=3, freq_max=30,
        epoch_duration = 8.0, 
        wavelet_type = 'morl', 
        ## Final sized of the scalogram, designed to be input of a CNN-2D
        final_width_px = 64,
        final_height_px = 64,
):
    
    subject_file  = config.seed_vig_filenames[subject]

    # 1. Load Data
    sample_data_raw_file = config.SEED_VIG_DIR / subject_file
    
    logger.info(f"Loading: {sample_data_raw_file}")

    mat = loadmat(sample_data_raw_file, squeeze_me=True, struct_as_record=False)
    EEG = mat["EEG"]
    del mat

    channel_list = list(EEG.chn)
    sfreq = EEG.sample_rate

    # 2. Preprocessing (Channel Selection & Filtering)

    try:
        channel_index = channel_list.index(channel)
    except ValueError:
        logger.warning(f"Channel '{channel}' not found in SEED-VIG channel list. Defaulting to first channel.")
        channel_index = 0

    raw_signal = EEG.data[:, channel_index]
    tot_samples=raw_signal.shape[0]
    del EEG

    # Apply bandpass filter to the selected channel
    filtered_signal = butter_bandpass_filter(raw_signal, freq_min, freq_max, sfreq, order=4)
    del raw_signal


    ## Creating a directory for saving the images
    Path(images_dir).mkdir(parents=True, exist_ok=True)


    ## Converting epoch_duration and step_duration to number of samples
    epoch_sample_duration = int(epoch_duration * sfreq)
    epoch_index = int(0)

    perclos_label_file = config.SEED_VIG_LABELS / subject_file
    mat = loadmat(perclos_label_file, squeeze_me=True, struct_as_record=False)
    perclos_label = mat["perclos"]
    del mat


    ## These are the frequencies and scales we are going to use for the CWT operation
    freqs = np.linspace(freq_min, freq_max, 256)
    scales = pywt.frequency2scale(wavelet_type, freqs * 1/sfreq)



    while  epoch_index*epoch_sample_duration + epoch_sample_duration < tot_samples :
        logger.info(f"{epoch_index*epoch_sample_duration + epoch_sample_duration} < {tot_samples}?")


        start_sample = epoch_index*epoch_sample_duration
        end_sample  = start_sample+ epoch_sample_duration

        window = filtered_signal[ start_sample : end_sample ]


        coef, _ = pywt.cwt(window, scales, wavelet_type, sampling_period=1/sfreq)
        power = np.abs(coef)**2


        # Normalize power_db between [0, 255]
        vmin = np.percentile(power, 20)
        vmax = np.percentile(power, 99)
        img = np.clip(power, vmin, vmax)
        img = (img - vmin) / (vmax - vmin)
        img = (img * 255).astype(np.uint8)


        # Apply resize
        img_resized = cv2.resize(img, (final_width_px, final_height_px), interpolation=cv2.INTER_CUBIC)
        ## The first axis is for scales, which are in inverse proportion with frequencies. 
        ## Then, we flip first axis.
        img_resized = np.flipud(img_resized)  



        sample_entry = {
            "label": round(perclos_label[epoch_index]),
            "subject_file": subject_file,
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