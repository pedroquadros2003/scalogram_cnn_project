import argparse
import logging
from pathlib import Path
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
import yaml

from scalogram_cnn_project.utils.signal_loader import SignalLoader
from scalogram_cnn_project.utils.classification_sequence import ClassificationSequence
import scalogram_cnn_project.settings.config as config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class BatchProgressCallback(tf.keras.callbacks.Callback):
    """
    Callback to output batch-level loss and accuracy metrics during classification training.
    """
    def __init__(self, total_samples, batch_size):
        super().__init__()
        self.total_samples = total_samples
        self.batch_size = batch_size
        self.processed = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.processed = 0
        logger.info(f"Epoch {epoch + 1} starting...")

    def on_train_batch_end(self, batch, logs=None):
        batch_size = logs.get('size', self.batch_size) if logs else self.batch_size
        self.processed = min(self.total_samples, self.processed + batch_size)
        loss = logs.get('loss', 0.0) if logs else 0.0
        acc = logs.get('accuracy', 0.0) if logs else 0.0
        print(f"Processed {self.processed}/{self.total_samples} sample pairs - loss: {loss:.6f} - accuracy: {acc:.4f}", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Train RNN-coupled MLP Classifier for drowsiness detection.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--dataset-type", type=str, choices=["seed_vig", "drozy"], help="Type of dataset (seed_vig or drozy)")
    parser.add_argument("--channel", type=str, help="Channel name to train on")
    parser.add_argument("--rnn-model-path", type=str, help="Path to the pretrained RNN forecaster model")
    parser.add_argument("--input-min", type=float, default=5.0, help="Input window duration in minutes")
    parser.add_argument("--predict-min", type=float, default=2.0, help="Output prediction duration in minutes")
    parser.add_argument("--stride-sec", type=float, default=30.0, help="Window sliding stride in seconds")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for classification MLP")
    parser.add_argument("--train-split", type=float, default=0.8, help="Fraction of data used for training")
    parser.add_argument("--subjects", type=int, nargs="+", default=None, help="Subject IDs to filter files for training")
    parser.add_argument("--drowsiness-threshold", type=int, default=4, help="Threshold for DROZY drowsiness labels (KSS >= threshold)")
    parser.add_argument("--resample-freq", type=float, default=None, help="Frequency to resample the signal to (Hz)")
    parser.add_argument("--force-cpu", action="store_true", help="Force training to run on CPU")
    parser.add_argument("--output-model", type=str, default=None, help="Path to save the trained coupled model")
    parser.add_argument("--metrics-json-path", type=str, default=None, help="Path to save final training and validation metrics as JSON")
    
    args = parser.parse_args()
    
    if args.config:
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
    if not args.rnn_model_path:
        parser.error("--rnn-model-path is required (either via CLI or YAML config)")
    if not (0.0 <= args.train_split <= 1.0):
        parser.error("--train-split must be between 0.0 and 1.0")
        
    if args.force_cpu:
        logger.info("Forcing CPU execution (disabling GPU devices)...")
        tf.config.set_visible_devices([], 'GPU')
        
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
        files_to_process = files
        logger.info(f"No subject filter specified. Processing all {len(files_to_process)} files...")
        
    loaded_signals = []
    loaded_labels_data = [] # Stores file-level label resources (KSS or PERCLOS array)
    sfreq = None
    
    for f in files_to_process:
        try:
            logger.info(f"Loading {f.name}...")
            signal_data = SignalLoader.load_signal(str(f), args.dataset_type, resample_freq=args.resample_freq)
            if sfreq is None:
                sfreq = signal_data.sfreq
            elif sfreq != signal_data.sfreq:
                logger.warning(f"File {f.name} has different sfreq {signal_data.sfreq} vs baseline {sfreq}. Skipping.")
                continue
                
            try:
                signal = signal_data.get_channel_signal(args.channel)
            except ValueError as e:
                logger.warning(f"Skipping {f.name}: {e}")
                continue
                
            # Load label resources for this file
            if args.dataset_type == "seed_vig":
                perclos_file = config.SEED_VIG_LABELS / f.name
                if not perclos_file.exists():
                    logger.warning(f"PERCLOS file {perclos_file.name} not found. Skipping file {f.name}.")
                    continue
                mat = loadmat(str(perclos_file), squeeze_me=True, struct_as_record=False)
                perclos_label = mat["perclos"]
                loaded_labels_data.append(perclos_label)
            else:
                # DROZY uses KSS score
                parts = f.stem.split("-")
                subj_id = int(parts[0])
                sess_id = int(parts[1])
                kss_val = config.drozy_kss_scale[subj_id][sess_id]
                drowsy_label = 1 if kss_val >= args.drowsiness_threshold else 0
                loaded_labels_data.append(drowsy_label)
                
            loaded_signals.append(signal)
            
        except Exception as e:
            logger.error(f"Error processing {f.name}: {e}")
            
    if not loaded_signals:
        raise ValueError("No training samples were generated. Check channel name, data and config.")
        
    input_len = int(args.input_min * 60.0 * sfreq)
    output_len = int(args.predict_min * 60.0 * sfreq)
    stride = int(args.stride_sec * sfreq)
    
    train_indices = []
    val_indices = []
    normalized_signals = []
    
    # Scale signals per file using ONLY training statistics to avoid data leakage
    for sig_idx, signal in enumerate(loaded_signals):
        num_samples = len(signal)
        i = 0
        file_indices = []
        while i + input_len + output_len <= num_samples:
            file_indices.append(i)
            i += stride
            
        if file_indices:
            n_pairs = len(file_indices)
            n_train = int(n_pairs * args.train_split)
            neglected = int(np.ceil((input_len + output_len) / stride))
            
            # Compute scaling statistics strictly on the training partition
            if n_train > 0:
                end_train_idx = (n_train - 1) * stride + input_len + output_len
                train_portion = signal[:end_train_idx]
                mean = np.mean(train_portion)
                std = np.std(train_portion)
            else:
                mean = np.mean(signal)
                std = np.std(signal)
                
            normalized_signal = (signal - mean) / std if std > 0 else (signal - mean)
            normalized_signals.append(normalized_signal)
            
            # Map labels and split chronologically
            if args.dataset_type == "seed_vig":
                perclos = loaded_labels_data[sig_idx]
            
            # Training partition
            for start_idx in file_indices[:n_train]:
                if args.dataset_type == "seed_vig":
                    start_sec = (start_idx + input_len) / sfreq
                    end_sec = (start_idx + input_len + output_len) / sfreq
                    start_epoch = int(start_sec / 8.0)
                    end_epoch = int(np.ceil(end_sec / 8.0))
                    start_epoch = min(len(perclos) - 1, max(0, start_epoch))
                    end_epoch = min(len(perclos), max(start_epoch + 1, end_epoch))
                    perclos_slice = perclos[start_epoch : end_epoch]
                    y = int(np.round(np.mean(perclos_slice)))
                else:
                    y = loaded_labels_data[sig_idx]
                train_indices.append((sig_idx, start_idx, y))
                
            # Validation partition (after overlap gap)
            val_start = n_train + neglected
            if val_start < n_pairs:
                for start_idx in file_indices[val_start:]:
                    if args.dataset_type == "seed_vig":
                        start_sec = (start_idx + input_len) / sfreq
                        end_sec = (start_idx + input_len + output_len) / sfreq
                        start_epoch = int(start_sec / 8.0)
                        end_epoch = int(np.ceil(end_sec / 8.0))
                        start_epoch = min(len(perclos) - 1, max(0, start_epoch))
                        end_epoch = min(len(perclos), max(start_epoch + 1, end_epoch))
                        perclos_slice = perclos[start_epoch : end_epoch]
                        y = int(np.round(np.mean(perclos_slice)))
                    else:
                        y = loaded_labels_data[sig_idx]
                    val_indices.append((sig_idx, start_idx, y))
            else:
                logger.warning(
                    f"Signal at index {sig_idx} does not have enough signal length for validation set after chronological split and neglect window."
                )
                
    if not train_indices:
        raise ValueError("No training samples were generated. Check dataset configuration.")
        
    train_seq = ClassificationSequence(
        loaded_signals=normalized_signals,
        indices_with_labels=train_indices,
        input_len=input_len,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    if val_indices:
        val_seq = ClassificationSequence(
            loaded_signals=normalized_signals,
            indices_with_labels=val_indices,
            input_len=input_len,
            batch_size=args.batch_size,
            shuffle=False
        )
        logger.info(f"Dataset generated. Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
    else:
        val_seq = None
        logger.info(f"Dataset generated. Train samples: {len(train_indices)} (No validation data)")
        
    # Build Coupled Model
    logger.info(f"Loading pretrained RNN forecaster from {args.rnn_model_path}...")
    rnn_model = tf.keras.models.load_model(args.rnn_model_path, compile=False)
    rnn_model.trainable = False # Freeze RNN forecasting layers
    
    combined_model = tf.keras.Sequential([
        rnn_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    combined_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    combined_model.summary()
    
    # Train
    logger.info("Starting coupled RNN-MLP model training...")
    progress_callback = BatchProgressCallback(total_samples=len(train_indices), batch_size=args.batch_size)
    
    history = combined_model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=args.epochs,
        verbose=0,
        callbacks=[progress_callback]
    )
    
    # Print final metrics
    logger.info("Training finished. Final classification metrics:")
    if history and history.history:
        for metric_name, values in history.history.items():
            if values:
                logger.info(f"  - Final {metric_name}: {values[-1]:.6f}")
                
    # Save model
    if args.output_model:
        out_path = Path(args.output_model)
    else:
        out_path = config.OUTPUT_DIR / "models" / f"combined_predict_classifier_{args.dataset_type}_{args.channel}.h5"
        
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving coupled model to {out_path}...")
    combined_model.save(str(out_path))
    
    # Save metrics JSON if requested
    if args.metrics_json_path and history and history.history:
        import json
        metrics_dict = {}
        for metric_name, values in history.history.items():
            if values:
                metrics_dict[metric_name] = float(values[-1])
        logger.info(f"Saving final metrics to {args.metrics_json_path}...")
        with open(args.metrics_json_path, "w") as f:
            json.dump(metrics_dict, f, indent=2)
            
    logger.info("Coupled classification model training completed successfully!")

if __name__ == "__main__":
    main()
