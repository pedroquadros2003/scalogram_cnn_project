import argparse
import logging
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from scalogram_cnn_project.utils.signal_loader import SignalLoader
from scalogram_cnn_project.models_for_prediction.model_predict_builder import create_prediction_model
import scalogram_cnn_project.settings.config as config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train RNN model for signal forecasting.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--dataset-type", type=str, choices=["seed_vig", "drozy"], help="Type of dataset (seed_vig or drozy)")
    parser.add_argument("--channel", type=str, help="Channel name to train on")
    parser.add_argument("--model-version", type=str, default="v0", help="Model version to train (e.g. v0 for LSTM, v1 for GRU)")
    parser.add_argument("--input-min", type=float, default=5.0, help="Input window duration in minutes")
    parser.add_argument("--predict-min", type=float, default=2.0, help="Output prediction duration in minutes")
    parser.add_argument("--stride-sec", type=float, default=30.0, help="Window sliding stride in seconds")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent dim for RNN layer")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output-model", type=str, default=None, help="Path to save the trained model")
    parser.add_argument("--subjects", type=int, nargs="+", default=None, help="Subject IDs to filter files for training (e.g. 1 2 5)")
    parser.add_argument("--train-split", type=float, default=0.8, help="Fraction of data used for training (chronological split)")
    
    args = parser.parse_args()
    
    if args.config:
        import yaml
        logger.info(f"Loading configuration from YAML file: {args.config}")
        with open(args.config, "r") as f:
            yaml_config = yaml.safe_load(f)
        for key, val in yaml_config.items():
            if hasattr(args, key):
                setattr(args, key, val)
                
    if not args.dataset_type:
        parser.error("--dataset-type is required (either via CLI or YAML config)")
    if not args.channel:
        parser.error("--channel is required (either via CLI or YAML config)")
    if not (0.0 <= args.train_split <= 1.0):
        parser.error("--train-split must be between 0.0 and 1.0")
    
    # Resolve input directory
    if args.dataset_type == "seed_vig":
        data_dir = config.SEED_VIG_DIR
        file_pattern = "*.mat"
    else:
        data_dir = config.DROZY_DIR / "psg"
        file_pattern = "*.edf"
        
    logger.info(f"Scanning for {file_pattern} files in {data_dir}...")
    files = list(data_dir.glob(file_pattern))
    if not files:
        raise ValueError(f"No signal files found in {data_dir}")
        
    # Filter files by subject if specified
    if args.subjects:
        filtered_files = []
        for f in files:
            try:
                if args.dataset_type == "seed_vig":
                    subj_id = int(f.stem.split("_")[0])
                else:
                    subj_id = int(f.stem.split("-")[0])
                
                if subj_id in args.subjects:
                    filtered_files.append(f)
            except (ValueError, IndexError):
                logger.warning(f"Could not parse subject ID from filename: {f.name}. Skipping.")
        files_to_process = filtered_files
        logger.info(f"Filtered to {len(files_to_process)} files matching subjects: {args.subjects}")
    else:
        # Default behavior: load all files
        files_to_process = files
        logger.info(f"No subject filter specified. Processing all {len(files_to_process)} files...")
        
    all_X_train = []
    all_y_train = []
    all_X_val = []
    all_y_val = []
    sfreq = None
    
    for f in files_to_process:
        try:
            logger.info(f"Loading {f.name}...")
            signal_data = SignalLoader.load_signal(str(f), args.dataset_type)
            if sfreq is None:
                sfreq = signal_data.sfreq
            elif sfreq != signal_data.sfreq:
                logger.warning(f"File {f.name} has different sfreq {signal_data.sfreq} vs baseline {sfreq}. Skipping.")
                continue
                
            input_len = int(args.input_min * 60.0 * sfreq)
            output_len = int(args.predict_min * 60.0 * sfreq)
            stride = int(args.stride_sec * sfreq)
            
            try:
                signal = signal_data.get_channel_signal(args.channel)
            except ValueError as e:
                logger.warning(f"Skipping {f.name}: {e}")
                continue
                
            num_samples = len(signal)
            
            # Slide window
            i = 0
            file_X = []
            file_y = []
            while i + input_len + output_len <= num_samples:
                file_X.append(signal[i : i + input_len])
                file_y.append(signal[i + input_len : i + input_len + output_len])
                i += stride
                
            if file_X:
                n_pairs = len(file_X)
                n_train = int(n_pairs * args.train_split)
                # Discard windows that overlap between train and val to avoid data leakage
                neglected = int(np.ceil((input_len + output_len) / stride))
                
                # Split training and validation chronologically
                all_X_train.extend(file_X[:n_train])
                all_y_train.extend(file_y[:n_train])
                
                val_start = n_train + neglected
                if val_start < n_pairs:
                    all_X_val.extend(file_X[val_start:])
                    all_y_val.extend(file_y[val_start:])
                else:
                    logger.warning(
                        f"File {f.name} does not have enough signal length for validation set after chronological split and neglect window."
                    )
                
        except Exception as e:
            logger.error(f"Error processing {f.name}: {e}")
            
    if not all_X_train:
        raise ValueError("No training samples were generated. Check channel name, data and train_split.")
        
    X_train = np.array(all_X_train, dtype=np.float32)
    y_train = np.array(all_y_train, dtype=np.float32)
    
    # Reshape to (samples, timesteps, features=1)
    X_train = np.expand_dims(X_train, axis=-1)
    y_train = np.expand_dims(y_train, axis=-1)
    
    if all_X_val:
        X_val = np.array(all_X_val, dtype=np.float32)
        y_val = np.array(all_y_val, dtype=np.float32)
        X_val = np.expand_dims(X_val, axis=-1)
        y_val = np.expand_dims(y_val, axis=-1)
        validation_data = (X_val, y_val)
        logger.info(f"Dataset generated. Train input shape: {X_train.shape}, Val input shape: {X_val.shape}")
    else:
        validation_data = None
        logger.info(f"Dataset generated. Train input shape: {X_train.shape} (No validation data)")
        
    # Build model
    parameters = {
        "input_steps": X_train.shape[1],
        "output_steps": y_train.shape[1],
        "latent_dim": args.latent_dim,
        "optimizer_name": "adam",
        "learning_rate": args.learning_rate
    }
    
    model = create_prediction_model(args.model_version, parameters)
    model.summary()
    
    # Train
    logger.info("Starting model training...")
    model.fit(
        X_train, y_train,
        validation_data=validation_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1
    )
    
    # Save model
    if args.output_model:
        out_path = Path(args.output_model)
    else:
        out_path = config.OUTPUT_DIR / "models" / f"rnn_predict_{args.model_version}_{args.dataset_type}_{args.channel}.h5"
        
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving model to {out_path}...")
    model.save(str(out_path))
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
