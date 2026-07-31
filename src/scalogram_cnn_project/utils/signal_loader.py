from pathlib import Path
from scipy.io import loadmat
import mne
import numpy as np
from scalogram_cnn_project.utils.signal_data import SignalData

class SignalLoader:
    """
    Loader class with static methods to read physical signal files (e.g. .mat, .edf)
    and parse them into standardized SignalData instances.
    """
    
    @staticmethod
    def load_seed_vig(file_path: str) -> SignalData:
        """
        Load a SEED-VIG .mat file and parse it.
        
        Args:
            file_path (str): Path to the SEED-VIG .mat file
            
        Returns:
            SignalData: Standardized signal container
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"SEED-VIG file not found: {file_path}")
            
        # loadmat with struct_as_record=False to access fields as attributes
        mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
        if "EEG" not in mat:
            raise KeyError(f"Key 'EEG' not found in MAT file {file_path}")
            
        eeg_struct = mat["EEG"]
        channels = list(eeg_struct.chn)
        sfreq = float(eeg_struct.sample_rate)
        
        # EEG.data in SEED-VIG is stored as (num_samples, num_channels)
        # We transpose it to match SignalData shape: (num_channels, num_samples)
        data = np.asarray(eeg_struct.data, dtype=np.float32).T
        
        return SignalData(data=data, channels=channels, sfreq=sfreq)
        
    @staticmethod
    def load_drozy(file_path: str) -> SignalData:
        """
        Load a DROZY .edf file and parse it using MNE.
        
        Args:
            file_path (str): Path to the DROZY .edf file
            
        Returns:
            SignalData: Standardized signal container
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"DROZY file not found: {file_path}")
            
        raw = mne.io.read_raw_edf(str(path), preload=True, verbose="WARNING")
        channels = list(raw.ch_names)
        sfreq = float(raw.info["sfreq"])
        data = raw.get_data().astype(np.float32) # shape is (num_channels, num_times)
        
        return SignalData(data=data, channels=channels, sfreq=sfreq)
        
    @staticmethod
    def load_signal(file_path: str, dataset_type: str) -> SignalData:
        """
        Load a signal file using the specified dataset type static parser.
        
        Args:
            file_path (str): Path to the signal file
            dataset_type (str): Type of dataset, either 'seed_vig' or 'drozy'
            
        Returns:
            SignalData: The parsed signal data container
        """
        dtype = str(dataset_type).lower().strip()
        if dtype in ["seed_vig", "seedvig"]:
            return SignalLoader.load_seed_vig(file_path)
        elif dtype == "drozy":
            return SignalLoader.load_drozy(file_path)
        else:
            raise ValueError(
                f"Unsupported dataset type '{dataset_type}'. "
                f"Supported types: 'seed_vig', 'drozy'"
            )
