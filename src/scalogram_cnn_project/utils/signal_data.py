import numpy as np

class SignalData:
    """
    Standardized container for physiological/EEG signal data.
    All data is stored in memory as numpy arrays.
    """
    def __init__(self, data: np.ndarray, channels: list, sfreq: float):
        """
        Initialize the SignalData container.
        
        Args:
            data (np.ndarray): 2D array of shape (num_channels, num_samples)
            channels (list): List of channel names (strings) corresponding to each row in data
            sfreq (float): Sampling frequency of the signal in Hz
        """
        self.data = np.asarray(data, dtype=np.float32)
        self.channels = [str(ch).strip() for ch in channels]
        self.sfreq = float(sfreq)
        
        if self.data.ndim != 2:
            raise ValueError(f"Signal data must be a 2D array, but got shape {self.data.shape}")
        if self.data.shape[0] != len(self.channels):
            raise ValueError(
                f"Number of channels in data ({self.data.shape[0]}) "
                f"does not match number of channel names ({len(self.channels)})"
            )
            
    def get_channel_signal(self, channel_name: str) -> np.ndarray:
        """
        Retrieve the 1D signal array for a specific channel name.
        
        Args:
            channel_name (str): Name of the channel to retrieve
            
        Returns:
            np.ndarray: 1D array of the signal
        """
        target = str(channel_name).strip()
        try:
            idx = self.channels.index(target)
            return self.data[idx]
        except ValueError:
            raise ValueError(
                f"Channel '{target}' not found. "
                f"Available channels: {self.channels}"
            )
            
    def get_channel_window(self, channel_name: str, start_min: float, end_min: float) -> np.ndarray:
        """
        Get a window of the signal for a specific channel between start_min and end_min (in minutes).
        
        Args:
            channel_name (str): Name of the channel
            start_min (float): Start time in minutes
            end_min (float): End time in minutes
            
        Returns:
            np.ndarray: 1D array of the sliced signal
        """
        signal = self.get_channel_signal(channel_name)
        start_sample = int(start_min * 60.0 * self.sfreq)
        end_sample = int(end_min * 60.0 * self.sfreq)
        
        # Boundary checks
        start_sample = max(0, start_sample)
        end_sample = min(len(signal), end_sample)
        
        return signal[start_sample:end_sample]
