## We begin by importing the necessary Python modules:

import scalogram_cnn_project.settings.config as config
import numpy as np
import mne
from scipy.signal import butter, filtfilt
from pathlib import Path

import logging
logger = logging.getLogger(__name__)



def generate_epoch_object(
        subject = 1, session = 1,
        freq_min=3, freq_max=30,
        epoch_duration = 30.0, 
        ## Determines the overlap between epochs
        overlap_ratio = 0,
        verbose = True,
):

    ## Importing the edf file
    sample_data_folder = config.DROZY_DIR
    sample_data_raw_file = (
        sample_data_folder / "psg" / f"{subject}-{session}.edf"
    )
    raw = mne.io.read_raw_edf(sample_data_raw_file)

    ch_names = [ "Fz", "Cz", "C3", "C4", "Pz"]
    ch_types = ['eeg'] * len(ch_names)
    sfreq = raw.info["sfreq"]
    raw_signal = raw.pick(picks = ch_names).get_data()
    tot_samples = raw_signal.shape[1]
    
    del raw


    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types=ch_types
    )


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



    # Applying the filter to the data
    filtered_signal = raw_signal

    for index_channel in range (len(ch_names)):
        filtered_signal[index_channel,:] = butter_bandpass_filter(raw_signal[index_channel,:],freq_min, freq_max, sfreq, order=4)


    ## Converting epoch_duration and step_duration to number of samples
    epoch_sample_duration = int(epoch_duration * sfreq)
    step_sample_duration = int(epoch_duration * (1 - overlap_ratio) * sfreq)


    epoch_index = int(0)
    epoch_list = []

    while  epoch_index*step_sample_duration + epoch_sample_duration < tot_samples :

        if verbose: print (f"{epoch_index*step_sample_duration + epoch_sample_duration} < {tot_samples}?")

        n_channels_n_times_list = []
        for index_channel in range (len(ch_names)):
            
            n_times = filtered_signal[index_channel, epoch_index*step_sample_duration : epoch_index*step_sample_duration  + epoch_sample_duration]

            n_channels_n_times_list.append(n_times)
        
        n_channels_n_times_np = np.stack(n_channels_n_times_list, axis=0)
        
        epoch_list.append(n_channels_n_times_np)

        ## Updating the epoch index
        epoch_index +=1

    data = np.stack(epoch_list)

    epochs = mne.EpochsArray(data, info)
    montage = mne.channels.make_standard_montage('standard_1020')
    epochs.set_montage(montage)

    return epochs





