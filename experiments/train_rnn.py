import argparse
import logging
from pathlib import Path
import numpy as np
import tensorflow as tf
from scalogram_cnn_project.utils.signal_loader import SignalLoader
from scalogram_cnn_project.models_for_prediction.model_predict_builder import create_prediction_model
import scalogram_cnn_project.settings.config as config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SignalSequence(tf.keras.utils.Sequence):
    def __init__(self, loaded_signals, indices, input_len, output_len, batch_size, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.loaded_signals = loaded_signals
        self.indices = list(indices)
        self.input_len = input_len
        self.output_len = output_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))
        
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        X_batch = []
        y_batch = []
        for sig_idx, start_idx in batch_indices:
            signal = self.loaded_signals[sig_idx]
            X_batch.append(signal[start_idx : start_idx + self.input_len])
            y_batch.append(signal[start_idx + self.input_len : start_idx + self.input_len + self.output_len])
            
        X_batch = np.array(X_batch, dtype=np.float32)
        y_batch = np.array(y_batch, dtype=np.float32)
        
        # Reshape to (batch, timesteps, features=1)
        X_batch = np.expand_dims(X_batch, axis=-1)
        y_batch = np.expand_dims(y_batch, axis=-1)
        
        return X_batch, y_batch
        
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

class BatchProgressCallback(tf.keras.callbacks.Callback):
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
        print(f"Processed {self.processed}/{self.total_samples} sample pairs - loss: {loss:.6f}", flush=True)

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
    parser.add_argument("--resample-freq", type=float, default=None, help="Frequency to resample the signal to (Hz) to speed up training")
    parser.add_argument("--force-cpu", action="store_true", help="Force training to run on CPU to avoid GPU VRAM OOM crashes")
    
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
        # Default behavior: load all files
        files_to_process = files
        logger.info(f"No subject filter specified. Processing all {len(files_to_process)} files...")
        
    loaded_signals = []
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
                
            loaded_signals.append(signal)
            
        except Exception as e:
            logger.error(f"Error processing {f.name}: {e}")
            
    if not loaded_signals:
        raise ValueError("No training samples were generated. Check channel name, data and train_split.")
        
    input_len = int(args.input_min * 60.0 * sfreq)
    output_len = int(args.predict_min * 60.0 * sfreq)
    stride = int(args.stride_sec * sfreq)
    
    train_indices = []
    val_indices = []
    
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
            
            # Split training and validation chronologically
            for start_idx in file_indices[:n_train]:
                train_indices.append((sig_idx, start_idx))
                
            val_start = n_train + neglected
            if val_start < n_pairs:
                for start_idx in file_indices[val_start:]:
                    val_indices.append((sig_idx, start_idx))
            else:
                logger.warning(
                    f"Signal at index {sig_idx} does not have enough signal length for validation set after chronological split and neglect window."
                )
                
    if not train_indices:
        raise ValueError("No training samples were generated. Check channel name, data and train_split.")
        
    train_seq = SignalSequence(
        loaded_signals=loaded_signals,
        indices=train_indices,
        input_len=input_len,
        output_len=output_len,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    if val_indices:
        val_seq = SignalSequence(
            loaded_signals=loaded_signals,
            indices=val_indices,
            input_len=input_len,
            output_len=output_len,
            batch_size=args.batch_size,
            shuffle=False
        )
        logger.info(f"Dataset generated. Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
    else:
        val_seq = None
        logger.info(f"Dataset generated. Train samples: {len(train_indices)} (No validation data)")
        
    # Build model
    parameters = {
        "input_steps": input_len,
        "output_steps": output_len,
        "latent_dim": args.latent_dim,
        "optimizer_name": "adam",
        "learning_rate": args.learning_rate
    }
    
    model = create_prediction_model(args.model_version, parameters)
    model.summary()
    
    # Train
    logger.info("Starting model training...")
    progress_callback = BatchProgressCallback(total_samples=len(train_indices), batch_size=args.batch_size)
    
    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=args.epochs,
        verbose=0,
        callbacks=[progress_callback]
    )
    
    # Print final metrics
    logger.info("Training finished. Final metrics:")
    if history and history.history:
        for metric_name, values in history.history.items():
            if values:
                logger.info(f"  - Final {metric_name}: {values[-1]:.6f}")
    
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
