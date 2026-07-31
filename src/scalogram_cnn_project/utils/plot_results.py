import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def plot_signal_comparison(
    input_signal: np.ndarray,
    predicted_signal: np.ndarray,
    sfreq: float,
    start_min: float,
    end_min: float,
    predict_min: float,
    output_path: str,
    ground_truth_signal: np.ndarray = None
):
    """
    Plot the input signal and the forecasted/predicted signal, saving it to a file.
    Optionally, plot the ground truth signal for comparison.
    
    Args:
        input_signal (np.ndarray): 1D array of the input signal.
        predicted_signal (np.ndarray): 1D array of the predicted signal.
        sfreq (float): Sampling frequency in Hz.
        start_min (float): Start time of input in minutes.
        end_min (float): End time of input in minutes.
        predict_min (float): Duration of prediction in minutes.
        output_path (str): File path where the plot will be saved.
        ground_truth_signal (np.ndarray, optional): 1D array of actual future signal.
    """
    # Calculate time axes
    input_len = len(input_signal)
    pred_len = len(predicted_signal)
    
    input_time = np.linspace(start_min, end_min, input_len)
    pred_time = np.linspace(end_min, end_min + predict_min, pred_len)
    
    plt.figure(figsize=(12, 6))
    
    # Plot input signal
    plt.plot(input_time, input_signal, label="Input Signal (Past)", color="royalblue", alpha=0.85)
    
    # Plot predicted signal
    plt.plot(pred_time, predicted_signal, label="Predicted Signal (Future)", color="darkorange", linewidth=2)
    
    # Plot ground truth if available
    if ground_truth_signal is not None:
        gt_len = min(len(ground_truth_signal), pred_len)
        plt.plot(
            pred_time[:gt_len], 
            ground_truth_signal[:gt_len], 
            label="Actual Signal (Ground Truth)", 
            color="grey", 
            linestyle="--", 
            alpha=0.7
        )
        
    plt.xlabel("Time (minutes)")
    plt.ylabel("Amplitude")
    plt.title("Signal Forecasting Comparison")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)
    
    # Save plot
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_file), dpi=150, bbox_inches="tight")
    plt.close()
