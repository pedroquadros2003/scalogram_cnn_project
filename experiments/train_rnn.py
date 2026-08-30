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
    parser.add_argument("--subject", type=int, default=None, help="Subject ID to filter files for training (only one subject at a time)")
    parser.add_argument("--output-plot", type=str, default=None, help="Path to save comparison plot of test/validation split predictions")
    parser.add_argument("--train-split", type=float, default=0.8, help="Fraction of data used for training (chronological split)")
    parser.add_argument("--resample-freq", type=float, default=None, help="Frequency to resample the signal to (Hz) to speed up training")
    parser.add_argument("--force-cpu", action="store_true", help="Force training to run on CPU to avoid GPU VRAM OOM crashes")
    parser.add_argument("--metrics-json-path", type=str, default=None, help="Path to save final training and validation metrics as JSON")
    
    args = parser.parse_args()
    
    if args.config:
        import yaml
        logger.info(f"Loading configuration from YAML file: {args.config}")
        with open(args.config, "r") as f:
            yaml_config = yaml.safe_load(f)
        for key, val in yaml_config.items():
            if key == "subjects" and hasattr(args, "subject"):
                if isinstance(val, list) and len(val) > 0:
                    setattr(args, "subject", val[0])
                else:
                    setattr(args, "subject", val)
            elif hasattr(args, key):
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
        
    # Filter files by subject if specified, else default to first valid subject
    if args.subject is not None:
        filtered_files = []
        for f in files:
            try:
                if args.dataset_type == "seed_vig":
                    subj_id = int(f.stem.split("_")[0])
                else:
                    subj_id = int(f.stem.split("-")[0])
                
                if subj_id == args.subject:
                    filtered_files.append(f)
            except (ValueError, IndexError):
                logger.warning(f"Could not parse subject ID from filename: {f.name}. Skipping.")
        files_to_process = filtered_files
        logger.info(f"Filtered to {len(files_to_process)} files matching subject: {args.subject}")
    else:
        # Enforce single subject by automatically choosing the first parsed subject
        first_subj_id = None
        for f in files:
            try:
                if args.dataset_type == "seed_vig":
                    first_subj_id = int(f.stem.split("_")[0])
                else:
                    first_subj_id = int(f.stem.split("-")[0])
                break
            except (ValueError, IndexError):
                continue
        
        if first_subj_id is not None:
            filtered_files = []
            for f in files:
                try:
                    if args.dataset_type == "seed_vig":
                        subj_id = int(f.stem.split("_")[0])
                    else:
                        subj_id = int(f.stem.split("-")[0])
                    if subj_id == first_subj_id:
                        filtered_files.append(f)
                except (ValueError, IndexError):
                    pass
            files_to_process = filtered_files
            logger.info(f"No subject specified. Automatically filtered to {len(files_to_process)} files matching first detected subject: {first_subj_id}")
        else:
            files_to_process = files[:1]
            logger.info(f"No subject specified and could not parse subject IDs. Processing only the first file: {[f.name for f in files_to_process]}")
        
    loaded_signals = []
    loaded_means = []
    loaded_stds = []
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
                
            # Calculate training portion boundaries to avoid data leakage during normalization
            num_samples = len(signal)
            file_sfreq = signal_data.sfreq
            file_input_len = int(args.input_min * 60.0 * file_sfreq)
            file_output_len = int(args.predict_min * 60.0 * file_sfreq)
            file_stride = int(args.stride_sec * file_sfreq)
            
            # Count windows
            n_pairs = 0
            idx = 0
            while idx + file_input_len + file_output_len <= num_samples:
                n_pairs += 1
                idx += file_stride
                
            n_train = int(n_pairs * args.train_split)
            if n_train > 0:
                end_train_idx = (n_train - 1) * file_stride + file_input_len + file_output_len
                train_portion = signal[:end_train_idx]
                mean = np.mean(train_portion)
                std = np.std(train_portion)
            else:
                mean = np.mean(signal)
                std = np.std(signal)
                
            if std > 0:
                normalized_signal = (signal - mean) / std
            else:
                normalized_signal = signal - mean
            loaded_signals.append(normalized_signal)
            loaded_means.append(mean)
            loaded_stds.append(std)
            
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
    
    # Generate predictions and comparison plot on validation/test split
    if val_indices and val_seq is not None:
        logger.info("Generating predictions on the validation/test split for plotting...")
        predictions = model.predict(val_seq)
        
        # Group val_indices by sig_idx
        from collections import defaultdict
        val_by_sig = defaultdict(list)
        for idx_in_val, (sig_idx, start_idx) in enumerate(val_indices):
            val_by_sig[sig_idx].append((idx_in_val, start_idx))
            
        for sig_idx, idxs in val_by_sig.items():
            # Sort by start_idx to ensure chronological order
            idxs = sorted(idxs, key=lambda x: x[1])
            
            start_idx_first = idxs[0][1]
            test_start_sample = start_idx_first + input_len
            
            start_idx_last = idxs[-1][1]
            test_end_sample = start_idx_last + input_len + output_len
            
            L = test_end_sample - test_start_sample
            if L <= 0:
                continue
                
            norm_signal = loaded_signals[sig_idx]
            gt_signal_norm = norm_signal[test_start_sample:test_end_sample]
            
            pred_sum = np.zeros(L)
            pred_count = np.zeros(L)
            
            for idx_in_val, start_idx in idxs:
                offset = start_idx + input_len - test_start_sample
                pred_window = np.squeeze(predictions[idx_in_val])
                if pred_window.ndim == 0:
                    pred_window = np.array([pred_window])
                pred_sum[offset : offset + len(pred_window)] += pred_window
                pred_count[offset : offset + len(pred_window)] += 1
                
            pred_signal_norm = pred_sum / np.maximum(pred_count, 1)
            
            # Denormalize
            mean = loaded_means[sig_idx]
            std = loaded_stds[sig_idx]
            gt_signal = gt_signal_norm * std + mean
            pred_signal = pred_signal_norm * std + mean
            
            # Matplotlib Plotting
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                
                plt.figure(figsize=(14, 6))
                time_axis = np.arange(test_start_sample, test_end_sample) / (sfreq * 60.0)  # Time in minutes
                
                plt.plot(time_axis, gt_signal, label="Original Signal (Test Split)", color="blue", alpha=0.7)
                plt.plot(time_axis, pred_signal, label="Predicted Signal (RNN)", color="red", alpha=0.85)
                
                plt.title(f"Comparison of Original vs Predicted Signal - Subject {args.subject or 'default'} (Channel {args.channel})")
                plt.xlabel("Time (minutes)")
                plt.ylabel("Amplitude")
                plt.legend(loc="best")
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                
                # Determine output plot filename
                if args.output_plot:
                    plot_path = Path(args.output_plot)
                else:
                    plot_path = out_path.with_suffix(".png")
                    
                if len(val_by_sig) > 1:
                    plot_file = plot_path.parent / f"{plot_path.stem}_file{sig_idx}.png"
                else:
                    plot_file = plot_path
                    
                plot_file.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(str(plot_file), dpi=150, bbox_inches="tight")
                plt.close()
                logger.info(f"Saved prediction comparison plot to {plot_file}")
            except Exception as plot_err:
                logger.error(f"Failed to generate prediction comparison plot: {plot_err}")
    else:
        logger.warning("No validation data available. Skipping comparison plot.")
    
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
            
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
