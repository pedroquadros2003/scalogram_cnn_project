import argparse
import logging
import os
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
import yaml

from scalogram_cnn_project.utils.signal_loader import SignalLoader
from scalogram_cnn_project.utils.classification_sequence import ClassificationSequence
from scalogram_cnn_project.models_for_prediction_classification import create_coupled_classifier_model
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

def resolve_model_path(model_path_str):
    """
    Resolves a model path whether provided as an absolute path,
    relative to current working directory, or relative to outputs/models/.
    """
    if not model_path_str:
        return None
    p = Path(model_path_str)
    if p.exists():
        return p
    # Try relative to workspace
    p_ws = config.PROJECT_DIR / model_path_str
    if p_ws.exists():
        return p_ws
    # Try in outputs/models/
    p_models = config.OUTPUT_DIR / "models" / p.name
    if p_models.exists():
        return p_models
def plot_training_history(history, output_plot_path, title_prefix="Coupled Classifier"):
    """
    Plots comparative training and validation curves (Loss, Accuracy, AUC) over epochs,
    matching the visual standards of the CNN+CWT and forecaster pipelines.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    hist = history.history if hasattr(history, "history") else history
    if not hist:
        return
        
    epochs_range = range(1, len(next(iter(hist.values()))) + 1)
    
    # Identify available metric pairs (metric, val_metric): Loss and Accuracy
    metric_pairs = []
    if "loss" in hist:
        metric_pairs.append(("loss", "val_loss", "Loss (Binary Cross-Entropy)"))
    if "accuracy" in hist:
        metric_pairs.append(("accuracy", "val_accuracy", "Accuracy"))
        
    if not metric_pairs:
        return
        
    n_metrics = len(metric_pairs)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6.5 * n_metrics, 5.0), dpi=150)
    if n_metrics == 1:
        axes = [axes]
        
    import matplotlib.ticker as ticker
    for ax, (train_key, val_key, label_name) in zip(axes, metric_pairs):
        if train_key in hist:
            ax.plot(epochs_range, hist[train_key], label="Train", color="#1f77b4", linewidth=2.2, marker="o", markersize=4.5)
        if val_key in hist:
            ax.plot(epochs_range, hist[val_key], label="Validation", color="#ff7f0e", linewidth=2.2, marker="s", markersize=4.5)
            
        ax.set_title(f"{label_name}", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Epoch", fontsize=11, fontweight="medium")
        ax.set_ylabel(label_name, fontsize=11, fontweight="medium")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", frameon=True, shadow=True)
        
        # Enforce strictly integer tick marks on the epoch x-axis
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        if len(epochs_range) <= 20:
            ax.set_xticks(list(epochs_range))
        
    plt.suptitle(f"{title_prefix} - Training vs Validation Curves", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    
    out_path = Path(output_plot_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved training history comparison plot to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Train Anticipatory/Predictive RNN-coupled Classifier for drowsiness detection.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--dataset-type", type=str, choices=["seed_vig", "drozy"], help="Type of dataset (seed_vig or drozy)")
    parser.add_argument("--channel", type=str, help="Channel name to train on (e.g. CP2, C3, Oz)")
    parser.add_argument("--model-version", type=str, default="v0", help="Architecture version for classifier (e.g. v0, v1)")
    parser.add_argument("--rnn-model-path", type=str, default="outputs/models/best_rnn_predictor_seedvig_CP2.h5", help="Path to the pretrained RNN forecaster model (.h5) or null/none/random to train from scratch")
    parser.add_argument("--latent-dim", type=int, default=32, help="Number of latent recurrent units when training from scratch")
    parser.add_argument("--rnn-type", type=str, default="lstm", choices=["lstm", "gru"], help="Recurrent backbone architecture when training from scratch (lstm or gru)")
    parser.add_argument("--lead-time-sec", type=float, default=0.0, help="Anticipation lead time X in seconds (target state at t + lead_time_sec)")
    parser.add_argument("--input-sec", type=float, default=None, help="Input signal window duration in seconds (if None, inferred from forecaster)")
    parser.add_argument("--input-min", type=float, default=None, help="[Deprecated] Input duration in minutes (converted to input-sec if provided)")
    parser.add_argument("--predict-min", type=float, default=None, help="[Deprecated] Lead duration in minutes (converted to lead-time-sec if provided)")
    parser.add_argument("--stride-sec", type=float, default=5.0, help="Window sliding stride in seconds")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for classification MLP")
    parser.add_argument("--hidden-units-1", type=int, default=None, help="Neurons in first dense layer")
    parser.add_argument("--hidden-units-2", type=int, default=None, help="Neurons in second dense layer")
    parser.add_argument("--dropout-1", type=float, default=None, help="Dropout rate for first dense layer")
    parser.add_argument("--dropout-2", type=float, default=None, help="Dropout rate for second dense layer")
    parser.add_argument("--optimizer-name", type=str, default="adam", help="Optimizer name (adam, rmsprop, sgd)")
    parser.add_argument("--class-weight-mode", type=str, default="none", help="Class weighting mode ('none', 'balanced', 'auto', 'custom')")
    parser.add_argument("--drowsy-weight", type=float, default=None, help="Explicit loss weight for drowsy class 1 (sets {0: 1.0, 1: drowsy_weight})")
    parser.add_argument("--class-weight", type=str, default=None, help="Custom class weights dictionary or JSON string")
    parser.add_argument("--train-split", type=float, default=0.8, help="Fraction of data used for training")
    parser.add_argument("--subjects", type=int, nargs="+", default=None, help="Subject IDs to filter files for training")
    parser.add_argument("--subject", type=int, default=None, help="Single Subject ID to filter files for training")
    parser.add_argument("--validation-subject", type=int, default=None, help="Subject ID left out for validation in LOSO cross-validation")
    parser.add_argument("--validation-subjects", type=int, nargs="+", default=None, help="Subject IDs left out for validation")
    parser.add_argument("--train-subjects", type=int, nargs="+", default=None, help="Explicit list of subject IDs for training")
    parser.add_argument("--use-full-signal", action="store_true", help="When enabled, uses 100% of signals without intra-file train/val split (default in LOSO)")
    parser.add_argument("--drowsiness-threshold", type=float, default=None, help="Threshold for drowsiness (PERCLOS threshold >= val for SEED-VIG, KSS >= val for DROZY)")
    parser.add_argument("--fine-tune-rnn", action="store_true", help="If set, unfreezes the LSTM/GRU backbone for end-to-end fine-tuning")
    parser.add_argument("--resample-freq", type=float, default=100.0, help="Frequency to resample the signal to (Hz)")
    parser.add_argument("--force-cpu", action="store_true", help="Force training to run on CPU")
    parser.add_argument("--output-model", type=str, default=None, help="Path to save the trained coupled model")
    parser.add_argument("--output-plot", type=str, default=None, help="Path to save comparative training vs validation curves plot")
    parser.add_argument("--save-plot", action=argparse.BooleanOptionalAction, default=True, help="Whether to generate and save training/validation comparison plots")
    parser.add_argument("--metrics-json-path", type=str, default=None, help="Path to save final training and validation metrics as JSON")
    
    args = parser.parse_args()
    
    if args.config:
        logger.info(f"Loading configuration from YAML file: {args.config}")
        with open(args.config, "r") as f:
            yaml_config = yaml.safe_load(f)
        for key, val in yaml_config.items():
            if key in ["validation_subject", "validation-subject"]:
                if isinstance(val, list) and len(val) > 0:
                    args.validation_subject = int(val[0])
                elif val is not None:
                    args.validation_subject = int(val)
            elif key in ["validation_subjects", "validation-subjects"]:
                if isinstance(val, int):
                    args.validation_subjects = [val]
                elif isinstance(val, list):
                    args.validation_subjects = [int(v) for v in val]
                elif isinstance(val, str):
                    args.validation_subjects = [int(v.strip()) for v in val.split(",") if v.strip()]
            elif key in ["train_subjects", "train-subjects"]:
                if isinstance(val, int):
                    args.train_subjects = [val]
                elif isinstance(val, list):
                    args.train_subjects = [int(v) for v in val]
                elif isinstance(val, str):
                    args.train_subjects = [int(v.strip()) for v in val.split(",") if v.strip()]
            elif key == "subject":
                if isinstance(val, list) and len(val) > 0:
                    args.subject = int(val[0])
                elif val is not None:
                    args.subject = int(val)
            elif key == "subjects":
                if isinstance(val, int):
                    args.subjects = [val]
                elif isinstance(val, list):
                    args.subjects = [int(v) for v in val]
                elif isinstance(val, str):
                    args.subjects = [int(v.strip()) for v in val.split(",") if v.strip()]
                else:
                    args.subjects = val
            elif hasattr(args, key):
                setattr(args, key, val)
                
    # Normalize subjects
    if args.subjects is None and args.subject is not None:
        args.subjects = [args.subject]
    elif isinstance(args.subjects, int):
        args.subjects = [args.subjects]
    elif isinstance(args.subjects, (list, tuple)):
        args.subjects = [int(s) for s in args.subjects]
        
    val_subj_list = []
    if args.validation_subject is not None:
        val_subj_list.append(args.validation_subject)
    if args.validation_subjects is not None:
        for s in args.validation_subjects:
            if s not in val_subj_list:
                val_subj_list.append(int(s))
                
    is_loso_mode = len(val_subj_list) > 0
    if is_loso_mode:
        args.use_full_signal = True
        
    if not args.dataset_type:
        parser.error("--dataset-type is required (either via CLI or YAML config)")
    if not args.channel:
        parser.error("--channel is required (either via CLI or YAML config)")
    if not (0.0 <= args.train_split <= 1.0):
        parser.error("--train-split must be between 0.0 and 1.0")

    # Check if we should initialize RNN backbone from scratch with random weights
    is_from_scratch = (
        args.rnn_model_path is None or
        str(args.rnn_model_path).strip().lower() in ["none", "null", "random", "scratch", ""]
    )
        
    # Handle backward compatibility for input_min and predict_min
    if args.input_min is not None and args.input_sec is None:
        args.input_sec = args.input_min * 60.0
    if args.predict_min is not None and args.lead_time_sec == 0.0:
        args.lead_time_sec = args.predict_min * 60.0
        
    # Set default drowsiness thresholds if not provided
    if args.drowsiness_threshold is None:
        if args.dataset_type == "seed_vig":
            args.drowsiness_threshold = 0.5  # Standard PERCLOS binarization threshold matching round(perclos)
        else:
            args.drowsiness_threshold = 4.0  # KSS threshold >= 4 for DROZY
            
    if args.force_cpu:
        logger.info("Forcing CPU execution (disabling GPU devices)...")
        tf.config.set_visible_devices([], 'GPU')

    if is_from_scratch:
        logger.info(f"Initializing RNN backbone FROM SCRATCH with random weights ({args.rnn_type.upper()}, {args.latent_dim} units)...")
        rnn_model = None
        resolved_model_path = None
        args.fine_tune_rnn = True  # Newly initialized backbone is trained end-to-end

        if args.input_sec is None:
            args.input_sec = 1.0  # 1.0s default
        input_len = int(args.input_sec * args.resample_freq)
    else:
        resolved_model_path = resolve_model_path(args.rnn_model_path)
        if not resolved_model_path or not resolved_model_path.exists():
            raise FileNotFoundError(f"Pretrained forecaster model not found at path: {args.rnn_model_path} (Resolved: {resolved_model_path})")
            
        logger.info(f"Loading pretrained RNN forecaster from {resolved_model_path}...")
        rnn_model = tf.keras.models.load_model(str(resolved_model_path), compile=False)
        
        # Inquire expected input shape from the forecaster
        model_expected_input_steps = rnn_model.input_shape[1] if hasattr(rnn_model, 'input_shape') and rnn_model.input_shape else None
        
        if args.input_sec is None:
            if model_expected_input_steps is not None:
                input_len = model_expected_input_steps
                args.input_sec = input_len / args.resample_freq
                logger.info(f"Automatically aligned input window to match forecaster: {input_len} steps ({args.input_sec:.2f}s at {args.resample_freq} Hz).")
            else:
                args.input_sec = 1.0  # 1.0s default
                input_len = int(args.input_sec * args.resample_freq)
        else:
            input_len = int(args.input_sec * args.resample_freq)
            if model_expected_input_steps is not None and input_len != model_expected_input_steps:
                logger.warning(
                    f"Configured input_len ({input_len} steps) differs from pretrained model expected steps ({model_expected_input_steps} steps). "
                    f"Forcing input_len to {model_expected_input_steps} steps to match model architecture."
                )
                input_len = model_expected_input_steps
                args.input_sec = input_len / args.resample_freq
            
    lead_time_samples = int(args.lead_time_sec * args.resample_freq)
    stride = int(args.stride_sec * args.resample_freq)
    
    logger.info(f"--- Anticipatory Classification Settings ---")
    logger.info(f"  - Dataset: {args.dataset_type.upper()}")
    logger.info(f"  - Channel: {args.channel}")
    if is_from_scratch:
        logger.info(f"  - Backbone: Random Initialization ({args.rnn_type.upper()} with {args.latent_dim} units)")
    else:
        logger.info(f"  - Pretrained Model: {resolved_model_path.name}")
    logger.info(f"  - Input Window: {input_len} samples ({args.input_sec:.2f} seconds)")
    logger.info(f"  - Lead Time (Anticipation X): {lead_time_samples} samples ({args.lead_time_sec:.2f} seconds ahead)")
    logger.info(f"  - Window Stride: {stride} samples ({args.stride_sec:.2f} seconds)")
    logger.info(f"  - Drowsiness Threshold: {args.drowsiness_threshold}")
    logger.info(f"  - Fine-Tuning Backbone: {args.fine_tune_rnn}")
    if is_loso_mode:
        logger.info(f"  - Validation Strategy: LOSO (Leave-One-Subject-Out)")
        logger.info(f"  - Validation Subject(s): {val_subj_list}")
    
    # Resolve input directory
    if args.dataset_type == "seed_vig":
        data_dir = config.SEED_VIG_DIR
        file_pattern = "*.mat"
    else:
        data_dir = config.DROZY_DIR / "psg"
        file_pattern = "*.edf"
        
    logger.info(f"Scanning for {file_pattern} files in {data_dir}...")
    files = sorted(list(data_dir.glob(file_pattern)))
    if not files:
        raise ValueError(f"No signal files found in {data_dir}")
        
    def parse_file_subject_id(file_path):
        stem = Path(file_path).stem
        if args.dataset_type == "seed_vig":
            return int(stem.split("_")[0])
        else:
            return int(stem.split("-")[0])

    train_indices = []
    val_indices = []
    normalized_signals = []
    total_window_span = input_len + lead_time_samples

    if is_loso_mode:
        # Separate files into training set (all other subjects) and validation set (left out subject)
        train_files = []
        val_files = []
        for f in files:
            try:
                sid = parse_file_subject_id(f)
                if sid in val_subj_list:
                    val_files.append(f)
                elif args.train_subjects is not None:
                    if sid in args.train_subjects:
                        train_files.append(f)
                else:
                    train_files.append(f)
            except (ValueError, IndexError):
                logger.warning(f"Could not parse subject ID from filename: {f.name}. Skipping.")
                
        logger.info(f"LOSO Partitioning: {len(train_files)} files for Training, {len(val_files)} files for Validation.")
        
        def process_loso_file_group(file_list, target_indices_list, partition_name):
            for f in file_list:
                try:
                    logger.info(f"Loading {partition_name} file: {f.name}...")
                    signal_data = SignalLoader.load_signal(str(f), args.dataset_type, resample_freq=args.resample_freq)
                    sfreq = signal_data.sfreq
                    signal = signal_data.get_channel_signal(args.channel)
                    
                    # Load label
                    if args.dataset_type == "seed_vig":
                        perclos_file = config.SEED_VIG_LABELS / f.name
                        if not perclos_file.exists():
                            logger.warning(f"PERCLOS file {perclos_file.name} not found. Skipping file {f.name}.")
                            continue
                        mat = loadmat(str(perclos_file), squeeze_me=True, struct_as_record=False)
                        perclos_label = mat["perclos"]
                    else:
                        parts = f.stem.split("-")
                        subj_id = int(parts[0])
                        sess_id = int(parts[1])
                        kss_val = config.drozy_kss_scale[subj_id][sess_id]
                        drowsy_label = 1 if kss_val >= args.drowsiness_threshold else 0
                        
                    # Standardize full signal per file
                    mean = np.mean(signal)
                    std = np.std(signal)
                    norm_sig = (signal - mean) / std if std > 0 else (signal - mean)
                    
                    sig_idx = len(normalized_signals)
                    normalized_signals.append(norm_sig)
                    
                    # Extract 100% of windows for this file
                    num_samples = len(norm_sig)
                    i = 0
                    while i + total_window_span <= num_samples:
                        if args.dataset_type == "seed_vig":
                            target_sec = (i + input_len + lead_time_samples) / sfreq
                            target_epoch = int(target_sec / 8.0)
                            target_epoch = min(len(perclos_label) - 1, max(0, target_epoch))
                            perclos_val = perclos_label[target_epoch]
                            y = 1 if perclos_val >= args.drowsiness_threshold else 0
                        else:
                            y = drowsy_label
                        target_indices_list.append((sig_idx, i, y))
                        i += stride
                except Exception as e:
                    logger.error(f"Error processing {partition_name} file {f.name}: {e}")

        process_loso_file_group(train_files, train_indices, "TRAIN")
        process_loso_file_group(val_files, val_indices, "VALIDATION (LOSO)")
        
    else:
        # Standard intra-session chronological 80/20 split
        if args.subjects:
            filtered_files = []
            for f in files:
                try:
                    sid = parse_file_subject_id(f)
                    if sid in args.subjects:
                        filtered_files.append(f)
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse subject ID from filename: {f.name}. Skipping.")
            files_to_process = filtered_files
            logger.info(f"Filtered to {len(files_to_process)} files matching subjects: {args.subjects}")
        else:
            files_to_process = files
            logger.info(f"No subject filter specified. Processing all {len(files_to_process)} files...")
            
        loaded_signals = []
        loaded_labels_data = []
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
            
        # Scale signals per file using ONLY training statistics to avoid data leakage
        for sig_idx, signal in enumerate(loaded_signals):
            num_samples = len(signal)
            i = 0
            file_indices = []
            while i + total_window_span <= num_samples:
                file_indices.append(i)
                i += stride
                
            if file_indices:
                n_pairs = len(file_indices)
                n_train = int(n_pairs * args.train_split)
                neglected = int(np.ceil(total_window_span / stride))
                
                # Compute scaling statistics strictly on the training partition
                if n_train > 0:
                    end_train_idx = (n_train - 1) * stride + total_window_span
                    train_portion = signal[:end_train_idx]
                    mean = np.mean(train_portion)
                    std = np.std(train_portion)
                else:
                    mean = np.mean(signal)
                    std = np.std(signal)
                    
                normalized_signal = (signal - mean) / std if std > 0 else (signal - mean)
                normalized_signals.append(normalized_signal)
                
                if args.dataset_type == "seed_vig":
                    perclos = loaded_labels_data[sig_idx]
                
                # Training partition
                for start_idx in file_indices[:n_train]:
                    if args.dataset_type == "seed_vig":
                        target_sec = (start_idx + input_len + lead_time_samples) / sfreq
                        target_epoch = int(target_sec / 8.0)
                        target_epoch = min(len(perclos) - 1, max(0, target_epoch))
                        perclos_val = perclos[target_epoch]
                        y = 1 if perclos_val >= args.drowsiness_threshold else 0
                    else:
                        y = loaded_labels_data[sig_idx]
                    train_indices.append((sig_idx, start_idx, y))
                    
                # Validation partition (after overlap gap to prevent data leakage)
                val_start = n_train + neglected
                if val_start < n_pairs:
                    for start_idx in file_indices[val_start:]:
                        if args.dataset_type == "seed_vig":
                            target_sec = (start_idx + input_len + lead_time_samples) / sfreq
                            target_epoch = int(target_sec / 8.0)
                            target_epoch = min(len(perclos) - 1, max(0, target_epoch))
                            perclos_val = perclos[target_epoch]
                            y = 1 if perclos_val >= args.drowsiness_threshold else 0
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
        
    # Build Coupled Model via models_for_prediction_classification package
    model_params = {
        "input_len": input_len,
        "learning_rate": args.learning_rate,
        "fine_tune_rnn": args.fine_tune_rnn,
        "optimizer_name": args.optimizer_name,
        "latent_dim": getattr(args, "latent_dim", 32),
        "rnn_type": getattr(args, "rnn_type", "lstm"),
    }
    if args.hidden_units_1 is not None:
        model_params["hidden_units_1"] = args.hidden_units_1
    if args.hidden_units_2 is not None:
        model_params["hidden_units_2"] = args.hidden_units_2
    if args.dropout_1 is not None:
        model_params["dropout_1"] = args.dropout_1
    if args.dropout_2 is not None:
        model_params["dropout_2"] = args.dropout_2
        
    combined_model = create_coupled_classifier_model(args.model_version, rnn_model, model_params)
    combined_model.summary()
    
    # Resolve class_weight for balanced or custom training
    class_weight = None
    if args.drowsy_weight is not None:
        class_weight = {0: 1.0, 1: float(args.drowsy_weight)}
        logger.info(f"Using explicit drowsy class weight multiplier: {class_weight}")
    elif args.class_weight is not None:
        if isinstance(args.class_weight, dict):
            class_weight = {int(k): float(v) for k, v in args.class_weight.items()}
        elif isinstance(args.class_weight, str):
            try:
                parsed = json.loads(args.class_weight)
                class_weight = {int(k): float(v) for k, v in parsed.items()}
            except Exception:
                logger.warning(f"Could not parse class_weight string '{args.class_weight}'. Defaulting to None.")
                class_weight = None
            logger.info(f"Using custom class weights: {class_weight}")
    elif str(args.class_weight_mode).lower() in ["balanced", "auto", "true"]:
        train_y = np.array([idx[2] for idx in train_indices])
        n_samples = len(train_y)
        n_0 = int(np.sum(train_y == 0))
        n_1 = int(np.sum(train_y == 1))
        if n_0 > 0 and n_1 > 0:
            w0 = float(n_samples / (2.0 * n_0))
            w1 = float(n_samples / (2.0 * n_1))
            class_weight = {0: w0, 1: w1}
            logger.info(f"Computed balanced class weights (N0={n_0}, N1={n_1}): {class_weight}")
        else:
            class_weight = None
    else:
        logger.info("Class weighting disabled (standard binary crossentropy).")
        
    # Train
    logger.info("Starting coupled anticipatory classifier training...")
    progress_callback = BatchProgressCallback(total_samples=len(train_indices), batch_size=args.batch_size)
    
    history = combined_model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=args.epochs,
        verbose=0,
        callbacks=[progress_callback],
        class_weight=class_weight
    )
    
    # Print final metrics
    logger.info("Training finished. Final classification metrics:")
    metrics_dict = {
        "n_train_samples": len(train_indices),
        "n_val_samples": len(val_indices),
        "validation_subject": val_subj_list[0] if val_subj_list else None,
        "is_loso": is_loso_mode,
        "from_scratch": is_from_scratch,
        "rnn_model_path": str(args.rnn_model_path) if not is_from_scratch else None
    }
    if history and history.history:
        metrics_dict["history"] = {k: [float(x) for x in v] for k, v in history.history.items()}
        for metric_name, values in history.history.items():
            if values:
                metrics_dict[metric_name] = float(values[-1])
                logger.info(f"  - Final {metric_name}: {values[-1]:.6f}")
                
    # Save model
    if args.output_model:
        out_path = Path(args.output_model)
    else:
        prefix = f"combined_scratch_{args.rnn_type}" if is_from_scratch else "combined_anticipatory"
        out_path = config.OUTPUT_DIR / "models" / f"{prefix}_classifier_{args.dataset_type}_{args.channel}_lead{int(args.lead_time_sec)}s.h5"
        
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving coupled model to {out_path}...")
    combined_model.save(str(out_path))
    
    # Save training and validation history curves plot (Loss, Accuracy, AUC over epochs)
    should_plot = args.save_plot and (args.output_plot is None or (isinstance(args.output_plot, str) and args.output_plot.lower() not in ["none", "false"]) or args.output_plot is True)
    if should_plot and history and history.history:
        if args.output_plot and isinstance(args.output_plot, str) and args.output_plot.lower() not in ["none", "false"]:
            plot_file = Path(args.output_plot)
        else:
            plot_file = out_path.with_suffix(".png")
        try:
            plot_title = f"{args.dataset_type.upper()} ({args.channel}) - Lead {int(args.lead_time_sec)}s - {args.model_version.upper()}"
            plot_training_history(history, plot_file, title_prefix=plot_title)
        except Exception as plot_err:
            logger.error(f"Failed to generate training history comparison plot: {plot_err}")
    elif not should_plot:
        logger.info("Plot generation is disabled (--no-save-plot / save_plot: false).")
    
    # Save metrics JSON if requested
    if args.metrics_json_path and metrics_dict:
        logger.info(f"Saving final metrics to {args.metrics_json_path}...")
        with open(args.metrics_json_path, "w") as f:
            json.dump(metrics_dict, f, indent=2)
            
    logger.info("Coupled classification model training completed successfully!")

if __name__ == "__main__":
    main()
