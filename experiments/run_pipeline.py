import argparse
import logging
from pathlib import Path
import numpy as np
from scipy.io import savemat
import tensorflow as tf
from scalogram_cnn_project.utils.signal_loader import SignalLoader
from scalogram_cnn_project.utils.plot_results import plot_signal_comparison
import scalogram_cnn_project.settings.config as config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run forecasting pipeline for physiological signal.")
    parser.add_argument("--file", type=str, required=True, help="Path to input signal file (.mat or .edf)")
    parser.add_argument("--dataset-type", type=str, required=True, choices=["seed_vig", "drozy"], help="Type of dataset (seed_vig or drozy)")
    parser.add_argument("--channel", type=str, required=True, help="Channel name to process")
    parser.add_argument("--start-min", type=float, required=True, help="Start of input window in minutes")
    parser.add_argument("--end-min", type=float, required=True, help="End of input window in minutes")
    parser.add_argument("--predict-min", type=float, required=True, help="Duration of forecast in minutes")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained Keras/TensorFlow forecasting model (.h5)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save outputs")
    
    args = parser.parse_args()
    
    # Resolve output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = config.OUTPUT_DIR
        
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load signal file
    input_file = Path(args.file)
    if not input_file.exists():
        if args.dataset_type == "seed_vig":
            candidate = config.SEED_VIG_DIR / input_file.name
        else:
            candidate = config.DROZY_DIR / "psg" / input_file.name
            
        if candidate.exists():
            logger.info(f"Resolved file '{args.file}' to dataset path: {candidate}")
            input_file = candidate
        else:
            raise FileNotFoundError(
                f"Could not find file '{args.file}' directly or inside dataset directory. "
                f"Attempted search path: {candidate}"
            )
            
    logger.info(f"Loading signal from {input_file} using parser for {args.dataset_type}...")
    signal_data = SignalLoader.load_signal(str(input_file), args.dataset_type)
    sfreq = signal_data.sfreq
    
    # 2. Extract input signal window
    logger.info(f"Extracting input window from {args.start_min} to {args.end_min} minutes for channel {args.channel}...")
    input_signal = signal_data.get_channel_window(args.channel, args.start_min, args.end_min)
    
    # Extract ground truth future signal window if available in the file
    logger.info("Extracting ground truth future signal (if available in file)...")
    full_signal = signal_data.get_channel_signal(args.channel)
    max_duration_min = len(full_signal) / (60.0 * sfreq)
    
    ground_truth = None
    if args.end_min + args.predict_min <= max_duration_min:
        ground_truth = signal_data.get_channel_window(args.channel, args.end_min, args.end_min + args.predict_min)
        logger.info("Ground truth future signal extracted successfully.")
    else:
        logger.warning(
            f"Future window ends at {args.end_min + args.predict_min} minutes, "
            f"but file only has {max_duration_min:.2f} minutes of signal. "
            f"Ground truth will not be plotted."
        )
        
    # 3. Load model
    logger.info(f"Loading trained forecasting model from {args.model_path}...")
    model = tf.keras.models.load_model(args.model_path, compile=False)
    
    # Check expected input steps
    expected_input_steps = model.input_shape[1]
    
    actual_input_steps = len(input_signal)
    
    # Adjust input signal to match model's expected shape if necessary
    if actual_input_steps != expected_input_steps:
        logger.warning(
            f"Input signal length ({actual_input_steps}) does not match model expected input steps ({expected_input_steps})."
        )
        if actual_input_steps > expected_input_steps:
            logger.info(f"Truncating input to last {expected_input_steps} samples.")
            input_signal_model = input_signal[-expected_input_steps:]
        else:
            logger.info("Zero-padding input signal to match expected input steps.")
            input_signal_model = np.pad(input_signal, (0, expected_input_steps - actual_input_steps), 'constant')
    else:
        input_signal_model = input_signal
        
    # 4. Predict
    # Shape: (1, input_steps, 1)
    model_input = np.expand_dims(np.expand_dims(input_signal_model, axis=0), axis=-1)
    logger.info("Running forecasting inference...")
    model_output = model.predict(model_input)
    
    # Reshape predicted sequence: (output_steps,)
    predicted_signal = np.reshape(model_output, (-1,))
    
    # Scale or adjust forecast size if prediction time differs from model expected output steps
    actual_output_steps = int(args.predict_min * 60.0 * sfreq)
    if len(predicted_signal) != actual_output_steps:
        logger.warning(
            f"Model prediction steps ({len(predicted_signal)}) differs from requested prediction duration steps ({actual_output_steps})."
        )
        # Interpolate or slice to match requested prediction duration
        if len(predicted_signal) > actual_output_steps:
            predicted_signal = predicted_signal[:actual_output_steps]
        else:
            predicted_signal = np.pad(predicted_signal, (0, actual_output_steps - len(predicted_signal)), 'edge')
            
    # 5. Save output predicted signal to .mat
    file_basename = input_file.stem
    out_mat_path = out_dir / f"predicted_{file_basename}_{args.channel}.mat"
    
    logger.info(f"Saving predicted signal to MAT file: {out_mat_path}...")
    save_dict = {
        "predicted_signal": predicted_signal,
        "sfreq": sfreq,
        "channel": args.channel,
        "start_min": args.start_min,
        "end_min": args.end_min,
        "predict_min": args.predict_min
    }
    savemat(str(out_mat_path), save_dict)
    
    # 6. Save plot
    out_plot_path = out_dir / f"plot_{file_basename}_{args.channel}.png"
    logger.info(f"Saving comparison plot to: {out_plot_path}...")
    
    plot_signal_comparison(
        input_signal=input_signal,
        predicted_signal=predicted_signal,
        sfreq=sfreq,
        start_min=args.start_min,
        end_min=args.end_min,
        predict_min=args.predict_min,
        output_path=str(out_plot_path),
        ground_truth_signal=ground_truth
    )
    
    logger.info("Forecasting pipeline run completed successfully!")

if __name__ == "__main__":
    main()
