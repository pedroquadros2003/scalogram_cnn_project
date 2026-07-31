## We begin by importing the necessary Python modules:

import numpy as np
import pywt
from scipy.signal import butter, filtfilt, welch
import cv2
from scalogram_cnn_project.utils.make_hash_id import make_hash_id
import json
from matplotlib import colormaps
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

## Defining the function that calculates the band power

def bandpower(f, psd, fmin, fmax):
    mask = (f >= fmin) & (f <= fmax)
    return np.trapezoid(psd[mask], f[mask])


def generate_scalogram_and_biomarkers(
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

    from scalogram_cnn_project.utils.signal_loader import SignalLoader

    ## Importing the edf file
    sample_data_folder = config.DROZY_DIR
    sample_data_raw_file = (
        sample_data_folder / "psg" / f"{subject}-{session}.edf"
    )
    signal_data = SignalLoader.load_drozy(
        file_path=sample_data_raw_file,
        resample_freq=resample_freq if do_resampling else None
    )
    raw_signal = signal_data.get_channel_signal(channel)
    sfreq = signal_data.sfreq
    tot_samples = signal_data.data.shape[1]
    del signal_data

    # Applying the filter to the data
    filtered_signal = butter_bandpass_filter(raw_signal, freq_min, freq_max, sfreq, order=4)
    del raw_signal

    ## Creating a directory for saving the images
    Path(images_dir).mkdir(parents=True, exist_ok=True)


    ## Converting epoch_duration and step_duration to number of samples
    epoch_sample_duration = int(epoch_duration * sfreq)
    step_sample_duration = int(epoch_duration * (1 - overlap_ratio) * sfreq)

    epoch_index = int(0)

    ## These are the frequencies and scales we are going to use for the CWT operation
    freqs = np.linspace(freq_min, freq_max, 256)
    scales = pywt.frequency2scale(wavelet_type, freqs * 1/sfreq)


    ## Create a list of numpy arrays, each with three features extracted from each epoch
    feature_np_list = []

    while  epoch_index*step_sample_duration + epoch_sample_duration < tot_samples :
        logger.info(f"{epoch_index*step_sample_duration + epoch_sample_duration} < {tot_samples}?")

        ## Now, we compute the CWT coefficients in dB

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


        # ---------------------------------
        # APPLY COLORMAP
        # ---------------------------------

        if cmap == "gray":

            img_to_save = img_resized

        else:

            # normalize to [0,1]
            img_float = img_resized.astype(np.float32) / 255.0

            # apply matplotlib colormap
            cmap_fn = colormaps[cmap]

            # RGBA output in [0,1]
            colored = cmap_fn(img_float)

            # remove alpha channel
            colored = colored[..., :3]

            # convert to uint8
            colored = (255 * colored).astype(np.uint8)

            # matplotlib gives RGB, OpenCV expects BGR
            img_to_save = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)

        # save image
        cv2.imwrite(str(images_dir / fig_name), img_to_save)

        ## And save the in .jsonl file the relation between each hash_id and its metadata
        with open(sample_file_path, "a") as f:
            f.write(json.dumps(sample_entry) + "\n")


        ## Next, we comput the PSD and relative power of each important frequency band
        f, psd = welch(
            window,
            fs=resample_freq,
            nperseg=256,
            noverlap=128
        )

        theta = bandpower(f, psd, 4, 8)
        alpha = bandpower(f, psd, 8, 13)
        beta  = bandpower(f, psd, 13, 30)

        feature_np_list.append( np.array( [(alpha + theta)/beta , alpha/beta , theta/beta] ) )

        ## Updating the epoch index
        epoch_index +=1


    ## Creating the feature array and normalize one feature a time
    features = np.stack(feature_np_list)

    features_min = features.min(axis=0)
    features_max = features.max(axis=0)

    features_norm = (features - features_min) / (features_max - features_min + 1e-8)
    return features_norm


if __name__ == "__main__": 

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s:%(name)s:%(message)s"
    )

    logger.debug( generate_scalogram_and_biomarkers() )