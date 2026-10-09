import argparse
import logging
import os
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
import yaml

from scalogram_cnn_project.utils.signal_loader import SignalLoader
from scalogram_cnn_project.utils.joint_sequence import JointSequence
from scalogram_cnn_project.models_for_joint_training import create_joint_model
import scalogram_cnn_project.settings.config as config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class JointBatchProgressCallback(tf.keras.callbacks.Callback):
    """
    Callback to log batch-level training metrics for multi-task joint models.
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
        clf_acc = logs.get('classification_output_accuracy', 0.0) if logs else 0.0
        f_mae = logs.get('forecast_output_mae', 0.0) if logs else 0.0
        print(
            f"Processed {self.processed}/{self.total_samples} samples | "
            f"Total Loss: {loss:.4f} | Clf Acc: {clf_acc*100:.1f}% | Forecast MAE: {f_mae:.4f}",
            flush=True
        )

def plot_joint_training_history(history, output_plot_path, title_prefix="Joint Model"):
    """
    Generates a 3-panel publication comparison figure showing:
      Panel 1: Loss Dynamics (Total Loss, Classification Loss, Forecast Loss)
      Panel 2: Classification Performance (Accuracy and AUC over epochs)
      Panel 3: Forecasting Fidelity (MAE and RMSE over epochs)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    
    hist = history.history if hasattr(history, "history") else history
    if not hist:
        return
        
    num_epochs = len(next(iter(hist.values())))
    epochs_range = range(1, num_epochs + 1)
    
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10.5,
        "axes.labelsize": 11.5,
        "axes.titlesize": 12.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=180)
    plt.subplots_adjust(wspace=0.28)
    
    # --- Panel 1: Loss Dynamics (Total, Classification, Forecast) ---
    ax_loss = axes[0]
    if "loss" in hist:
        ax_loss.plot(epochs_range, hist["loss"], label="Train Total Loss", color="#1f77b4", linewidth=2.2, marker="o", markersize=4)
    if "val_loss" in hist:
        ax_loss.plot(epochs_range, hist["val_loss"], label="Val Total Loss", color="#ff7f0e", linewidth=2.2, marker="s", markersize=4)
    if "classification_output_loss" in hist:
        ax_loss.plot(epochs_range, hist["classification_output_loss"], label="Train Clf Loss (BCE)", color="#2ca02c", linestyle="--", linewidth=1.5)
    if "val_classification_output_loss" in hist:
        ax_loss.plot(epochs_range, hist["val_classification_output_loss"], label="Val Clf Loss (BCE)", color="#d62728", linestyle="--", linewidth=1.5)
    if "forecast_output_loss" in hist:
        ax_loss.plot(epochs_range, hist["forecast_output_loss"], label="Train Forecast Loss (MSE)", color="#9467bd", linestyle=":", linewidth=1.5)
    if "val_forecast_output_loss" in hist:
        ax_loss.plot(epochs_range, hist["val_forecast_output_loss"], label="Val Forecast Loss (MSE)", color="#8c564b", linestyle=":", linewidth=1.5)
        
    ax_loss.set_title("A. Loss Trajectories (Total & Component Losses)", fontweight="bold", pad=8)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, linestyle="--", alpha=0.5)
    ax_loss.legend(loc="upper right", frameon=True, fontsize=8.5)
    ax_loss.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    # --- Panel 2: Classification Performance (Accuracy & AUC) ---
    ax_clf = axes[1]
    if "classification_output_accuracy" in hist:
        ax_clf.plot(epochs_range, hist["classification_output_accuracy"], label="Train Accuracy", color="#1f77b4", linewidth=2.2, marker="o", markersize=4)
    if "val_classification_output_accuracy" in hist:
        ax_clf.plot(epochs_range, hist["val_classification_output_accuracy"], label="Val Accuracy", color="#ff7f0e", linewidth=2.2, marker="s", markersize=4)
    if "classification_output_auc" in hist:
        ax_clf.plot(epochs_range, hist["classification_output_auc"], label="Train AUC", color="#2ca02c", linestyle="--", linewidth=1.8)
    if "val_classification_output_auc" in hist:
        ax_clf.plot(epochs_range, hist["val_classification_output_auc"], label="Val AUC", color="#d62728", linestyle="--", linewidth=1.8)
        
    ax_clf.set_title("B. Classification Dynamics (Accuracy & AUC)", fontweight="bold", pad=8)
    ax_clf.set_xlabel("Epoch")
    ax_clf.set_ylabel("Score / Metric")
    ax_clf.set_ylim(-0.02, 1.05)
    ax_clf.grid(True, linestyle="--", alpha=0.5)
    ax_clf.legend(loc="lower right", frameon=True)
    ax_clf.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    # --- Panel 3: Forecasting Error (MAE & RMSE) ---
    ax_f = axes[2]
    if "forecast_output_mae" in hist:
        ax_f.plot(epochs_range, hist["forecast_output_mae"], label="Train MAE", color="#1f77b4", linewidth=2.2, marker="o", markersize=4)
    if "val_forecast_output_mae" in hist:
        ax_f.plot(epochs_range, hist["val_forecast_output_mae"], label="Val MAE", color="#ff7f0e", linewidth=2.2, marker="s", markersize=4)
    if "forecast_output_rmse" in hist:
        ax_f.plot(epochs_range, hist["forecast_output_rmse"], label="Train RMSE", color="#9467bd", linestyle="--", linewidth=1.8)
    if "val_forecast_output_rmse" in hist:
        ax_f.plot(epochs_range, hist["val_forecast_output_rmse"], label="Val RMSE", color="#8c564b", linestyle="--", linewidth=1.8)
        
    ax_f.set_title("C. Signal Reconstruction Error (MAE / RMSE)", fontweight="bold", pad=8)
    ax_f.set_xlabel("Epoch")
    ax_f.set_ylabel("Error (Normalized)")
    ax_f.grid(True, linestyle="--", alpha=0.5)
    ax_f.legend(loc="upper right", frameon=True)
    ax_f.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    plt.suptitle(f"{title_prefix} — Joint Training & Validation Performance Curves", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    out_path = Path(output_plot_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved joint training comparison plot to {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Train Multi-Task Joint Model (Signal Forecasting + Drowsiness Classification) with single alpha loss balance."
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--dataset-type", type=str, choices=["seed_vig", "drozy"], help="Type of dataset (seed_vig or drozy)")
    parser.add_argument("--channel", type=str, help="Channel name to train on (e.g. CP2, C3, Oz)")
    parser.add_argument("--predict-version", type=str, default="v0", help="Forecasting backbone architecture (v0=LSTM, v1=GRU)")
    parser.add_argument("--classifier-version", type=str, default="v0", help="Classification head architecture (v0=standard MLP, v1=BN-MLP)")
    parser.add_argument("--loss-alpha", type=float, default=0.5, help="Single trade-off alpha parameter in [0.0, 1.0]: Loss = alpha * Loss_clf + (1 - alpha) * Loss_forecast")
    parser.add_argument("--lead-time-sec", type=float, default=0.0, help="Anticipation lead time X in seconds (target vigilance state at t + lead_time_sec)")
    parser.add_argument("--forecast-steps", type=int, default=1, help="Number of future voltage time steps to forecast (default: 1 step = 10 ms)")
    parser.add_argument("--input-sec", type=float, default=1.0, help="Input signal window duration in seconds (default: 1.0 s)")
    parser.add_argument("--stride-sec", type=float, default=5.0, help="Sliding window stride in seconds (default: 5.0 s)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--latent-dim", type=int, default=32, help="Number of recurrent latent units in shared backbone")
    parser.add_argument("--hidden-units-1", type=int, default=None, help="Neurons in first classifier dense layer")
    parser.add_argument("--hidden-units-2", type=int, default=None, help="Neurons in second classifier dense layer")
    parser.add_argument("--dropout-1", type=float, default=None, help="Dropout rate for first classifier dense layer")
    parser.add_argument("--dropout-2", type=float, default=None, help="Dropout rate for second classifier dense layer")
    parser.add_argument("--optimizer-name", type=str, default="adam", help="Optimizer name (adam, rmsprop, sgd)")
    parser.add_argument("--class-weight-mode", type=str, default="none", help="Class weighting mode ('none', 'balanced', 'auto', 'custom')")
    parser.add_argument("--drowsy-weight", type=float, default=None, help="Explicit weight multiplier for drowsy class 1")
    parser.add_argument("--train-split", type=float, default=0.8, help="Fraction of data used for training (in temporal split)")
    parser.add_argument("--subjects", type=int, nargs="+", default=None, help="Subject IDs to filter files for training")
    parser.add_argument("--subject", type=int, default=None, help="Single Subject ID to filter files for training")
    parser.add_argument("--validation-subject", type=int, default=None, help="Subject ID left out for validation in LOSO cross-validation")
    parser.add_argument("--validation-subjects", type=int, nargs="+", default=None, help="Subject IDs left out for validation")
    parser.add_argument("--train-subjects", type=int, nargs="+", default=None, help="Explicit list of subject IDs for training")
    parser.add_argument("--use-full-signal", action="store_true", help="When enabled, uses 100% of signals without intra-file train/val split (default in LOSO)")
    parser.add_argument("--drowsiness-threshold", type=float, default=None, help="Threshold for drowsiness (PERCLOS >= val for SEED-VIG, KSS >= val for DROZY)")
    parser.add_argument("--resample-freq", type=float, default=100.0, help="Frequency to resample the signal to (Hz)")
    parser.add_argument("--force-cpu", action="store_true", help="Force training to run on CPU")
    parser.add_argument("--output-model", type=str, default=None, help="Path to save the trained joint model")
    parser.add_argument("--output-plot", type=str, default=None, help="Path to save comparative training vs validation curves plot")
    parser.add_argument("--save-plot", action=argparse.BooleanOptionalAction, default=True, help="Whether to generate and save training/validation comparison plots")
    parser.add_argument("--metrics-json-path", type=str, default=None, help="Path to save final metrics as JSON")
    
    args = parser.parse_args()
    
    if args.config:
        logger.info(f"Loading configuration from YAML file: {args.config}")
        with open(args.config, "r") as f:
            yaml_config = yaml.safe_load(f)
        for key, val in yaml_config.items():
            norm_key = key.replace("-", "_")
            if norm_key in ["validation_subject"]:
                if isinstance(val, list) and len(val) > 0:
                    args.validation_subject = int(val[0])
                elif val is not None:
                    args.validation_subject = int(val)
            elif norm_key in ["validation_subjects"]:
                if isinstance(val, int):
                    args.validation_subjects = [val]
                elif isinstance(val, list):
                    args.validation_subjects = [int(v) for v in val]
                elif isinstance(val, str):
                    args.validation_subjects = [int(v.strip()) for v in val.split(",") if v.strip()]
            elif norm_key in ["train_subjects"]:
                if isinstance(val, int):
                    args.train_subjects = [val]
                elif isinstance(val, list):
                    args.train_subjects = [int(v) for v in val]
                elif isinstance(val, str):
                    args.train_subjects = [int(v.strip()) for v in val.split(",") if v.strip()]
            elif norm_key == "subject":
                if isinstance(val, list) and len(val) > 0:
                    args.subject = int(val[0])
                elif val is not None:
                    args.subject = int(val)
            elif norm_key == "subjects":
                if isinstance(val, int):
                    args.subjects = [val]
                elif isinstance(val, list):
                    args.subjects = [int(v) for v in val]
                elif isinstance(val, str):
                    args.subjects = [int(v.strip()) for v in val.split(",") if v.strip()]
                else:
                    args.subjects = val
            elif hasattr(args, norm_key):
                setattr(args, norm_key, val)
                
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
    if not (0.0 <= args.loss_alpha <= 1.0):
        parser.error("--loss-alpha must be a float between 0.0 and 1.0")
        
    # Default drowsiness thresholds
    if args.drowsiness_threshold is None:
        if args.dataset_type == "seed_vig":
            args.drowsiness_threshold = 0.5
        else:
            args.drowsiness_threshold = 4.0
            
    if args.force_cpu:
        logger.info("Forcing CPU execution (disabling GPU devices)...")
        tf.config.set_visible_devices([], 'GPU')
        
    input_len = int(args.input_sec * args.resample_freq)
    forecast_steps = int(args.forecast_steps)
    lead_time_samples = int(args.lead_time_sec * args.resample_freq)
    stride = int(args.stride_sec * args.resample_freq)
    total_window_span = input_len + max(forecast_steps, lead_time_samples)
    
    logger.info("=" * 70)
    logger.info("--- JOINT MODEL (MULTI-TASK TRAINING: PREDICTION + CLASSIFICATION) ---")
    logger.info(f"  - Dataset: {args.dataset_type.upper()} | Channel: {args.channel}")
    logger.info(f"  - Architecture: Predictor {args.predict_version.upper()} + Classifier {args.classifier_version.upper()}")
    logger.info(f"  - Loss Balance Alpha: {args.loss_alpha:.3f} (Clf weight: {args.loss_alpha:.3f}, Forecast weight: {1.0-args.loss_alpha:.3f})")
    logger.info(f"  - Input Window: {input_len} samples ({args.input_sec:.2f} s)")
    logger.info(f"  - Forecast Steps: {forecast_steps} sample(s) ({forecast_steps/args.resample_freq*1000.0:.1f} ms)")
    logger.info(f"  - Lead Time (Anticipation): {lead_time_samples} samples ({args.lead_time_sec:.2f} s ahead)")
    logger.info(f"  - Window Stride: {stride} samples ({args.stride_sec:.2f} s)")
    logger.info(f"  - Latent Recurrent Units: {args.latent_dim} (Random weights / Zero Data Leakage)")
    if is_loso_mode:
        logger.info(f"  - Evaluation Mode: LOSO (Validation Subject(s): {val_subj_list})")
    else:
        logger.info(f"  - Evaluation Mode: Temporal Split (Train: {args.train_split*100:.0f}%, Val: {(1-args.train_split)*100:.0f}%)")
    logger.info("=" * 70)
    
    # Resolve signal directory
    if args.dataset_type == "seed_vig":
        data_dir = config.SEED_VIG_DIR
        file_pattern = "*.mat"
    else:
        data_dir = config.DROZY_DIR / "psg"
        file_pattern = "*.edf"
        
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
    
    if is_loso_mode:
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
                logger.warning(f"Could not parse subject ID from {f.name}. Skipping.")
                
        logger.info(f"LOSO Partitioning: {len(train_files)} files for Training, {len(val_files)} files for Validation.")
        
        def process_loso_group(file_list, target_indices, partition_name):
            for f in file_list:
                try:
                    logger.info(f"Loading {partition_name} file: {f.name}...")
                    signal_data = SignalLoader.load_signal(str(f), args.dataset_type, resample_freq=args.resample_freq)
                    sfreq = signal_data.sfreq
                    signal = signal_data.get_channel_signal(args.channel)
                    
                    if args.dataset_type == "seed_vig":
                        perclos_file = config.SEED_VIG_LABELS / f.name
                        if not perclos_file.exists():
                            logger.warning(f"PERCLOS file {perclos_file.name} not found. Skipping {f.name}.")
                            continue
                        mat = loadmat(str(perclos_file), squeeze_me=True, struct_as_record=False)
                        perclos_label = mat["perclos"]
                    else:
                        parts = f.stem.split("-")
                        subj_id = int(parts[0])
                        sess_id = int(parts[1])
                        kss_val = config.drozy_kss_scale[subj_id][sess_id]
                        drowsy_label = 1 if kss_val >= args.drowsiness_threshold else 0
                        
                    mean = np.mean(signal)
                    std = np.std(signal)
                    norm_sig = (signal - mean) / std if std > 0 else (signal - mean)
                    
                    sig_idx = len(normalized_signals)
                    normalized_signals.append(norm_sig)
                    
                    num_samples = len(norm_sig)
                    i = 0
                    while i + total_window_span <= num_samples:
                        # Forecast target
                        y_forecast = norm_sig[i + input_len : i + input_len + forecast_steps]
                        if forecast_steps == 1:
                            y_forecast = y_forecast[0]
                            
                        # Classification target
                        if args.dataset_type == "seed_vig":
                            target_sec = (i + input_len + lead_time_samples) / sfreq
                            target_epoch = int(target_sec / 8.0)
                            target_epoch = min(len(perclos_label) - 1, max(0, target_epoch))
                            y_clf = 1 if perclos_label[target_epoch] >= args.drowsiness_threshold else 0
                        else:
                            y_clf = drowsy_label
                            
                        target_indices.append((sig_idx, i, y_forecast, y_clf))
                        i += stride
                except Exception as e:
                    logger.error(f"Error processing {partition_name} file {f.name}: {e}")
                    
        process_loso_group(train_files, train_indices, "TRAIN")
        process_loso_group(val_files, val_indices, "VALIDATION (LOSO)")
        
    else:
        # Standard intra-session chronological 80/20 split
        if args.subjects:
            files_to_process = [f for f in files if parse_file_subject_id(f) in args.subjects]
        else:
            files_to_process = files
            
        loaded_signals = []
        loaded_labels = []
        sfreq = None
        
        for f in files_to_process:
            try:
                logger.info(f"Loading {f.name}...")
                signal_data = SignalLoader.load_signal(str(f), args.dataset_type, resample_freq=args.resample_freq)
                if sfreq is None:
                    sfreq = signal_data.sfreq
                signal = signal_data.get_channel_signal(args.channel)
                
                if args.dataset_type == "seed_vig":
                    perclos_file = config.SEED_VIG_LABELS / f.name
                    if not perclos_file.exists():
                        continue
                    mat = loadmat(str(perclos_file), squeeze_me=True, struct_as_record=False)
                    loaded_labels.append(mat["perclos"])
                else:
                    parts = f.stem.split("-")
                    kss_val = config.drozy_kss_scale[int(parts[0])][int(parts[1])]
                    loaded_labels.append(1 if kss_val >= args.drowsiness_threshold else 0)
                    
                loaded_signals.append(signal)
            except Exception as e:
                logger.error(f"Error processing {f.name}: {e}")
                
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
                
                if n_train > 0:
                    train_portion = signal[:(n_train - 1) * stride + total_window_span]
                    mean = np.mean(train_portion)
                    std = np.std(train_portion)
                else:
                    mean = np.mean(signal)
                    std = np.std(signal)
                    
                norm_sig = (signal - mean) / std if std > 0 else (signal - mean)
                normalized_signals.append(norm_sig)
                
                # Train partition
                for start_idx in file_indices[:n_train]:
                    y_f = norm_sig[start_idx + input_len : start_idx + input_len + forecast_steps]
                    if forecast_steps == 1:
                        y_f = y_f[0]
                    if args.dataset_type == "seed_vig":
                        t_sec = (start_idx + input_len + lead_time_samples) / sfreq
                        t_ep = min(len(loaded_labels[sig_idx]) - 1, max(0, int(t_sec / 8.0)))
                        y_c = 1 if loaded_labels[sig_idx][t_ep] >= args.drowsiness_threshold else 0
                    else:
                        y_c = loaded_labels[sig_idx]
                    train_indices.append((sig_idx, start_idx, y_f, y_c))
                    
                # Validation partition
                val_start = n_train + neglected
                if val_start < n_pairs:
                    for start_idx in file_indices[val_start:]:
                        y_f = norm_sig[start_idx + input_len : start_idx + input_len + forecast_steps]
                        if forecast_steps == 1:
                            y_f = y_f[0]
                        if args.dataset_type == "seed_vig":
                            t_sec = (start_idx + input_len + lead_time_samples) / sfreq
                            t_ep = min(len(loaded_labels[sig_idx]) - 1, max(0, int(t_sec / 8.0)))
                            y_c = 1 if loaded_labels[sig_idx][t_ep] >= args.drowsiness_threshold else 0
                        else:
                            y_c = loaded_labels[sig_idx]
                        val_indices.append((sig_idx, start_idx, y_f, y_c))
                        
    if not train_indices:
        raise ValueError("No training samples generated. Check dataset configuration.")
        
    train_seq = JointSequence(
        loaded_signals=normalized_signals,
        indices_with_dual_labels=train_indices,
        input_len=input_len,
        forecast_steps=forecast_steps,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    if val_indices:
        val_seq = JointSequence(
            loaded_signals=normalized_signals,
            indices_with_dual_labels=val_indices,
            input_len=input_len,
            forecast_steps=forecast_steps,
            batch_size=args.batch_size,
            shuffle=False
        )
        logger.info(f"Datasets generated: Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
    else:
        val_seq = None
        logger.info(f"Datasets generated: Train samples: {len(train_indices)} (No validation data)")
        
    # Build Joint Model using unified flexible builder
    joint_params = {
        "input_len": input_len,
        "forecast_steps": forecast_steps,
        "latent_dim": args.latent_dim,
        "loss_alpha": args.loss_alpha,
        "learning_rate": args.learning_rate,
        "optimizer_name": args.optimizer_name,
    }
    if args.hidden_units_1 is not None:
        joint_params["hidden_units_1"] = args.hidden_units_1
    if args.hidden_units_2 is not None:
        joint_params["hidden_units_2"] = args.hidden_units_2
    if args.dropout_1 is not None:
        joint_params["dropout_1"] = args.dropout_1
    if args.dropout_2 is not None:
        joint_params["dropout_2"] = args.dropout_2
        
    joint_model = create_joint_model(
        predict_version=args.predict_version,
        classifier_version=args.classifier_version,
        parameters=joint_params
    )
    joint_model.summary()
    
    # Train
    logger.info("Starting multi-task joint model training...")
    progress_callback = JointBatchProgressCallback(total_samples=len(train_indices), batch_size=args.batch_size)
    
    history = joint_model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=args.epochs,
        verbose=0,
        callbacks=[progress_callback]
    )
    
    logger.info("Joint training completed. Final summary metrics:")
    metrics_dict = {
        "n_train_samples": len(train_indices),
        "n_val_samples": len(val_indices),
        "validation_subject": val_subj_list[0] if val_subj_list else None,
        "is_loso": is_loso_mode,
        "loss_alpha": args.loss_alpha,
        "predict_version": args.predict_version,
        "classifier_version": args.classifier_version,
        "forecast_steps": args.forecast_steps,
        "lead_time_sec": args.lead_time_sec
    }
    if history and history.history:
        metrics_dict["history"] = {k: [float(x) for x in v] for k, v in history.history.items()}
        for k, v in history.history.items():
            if v:
                metrics_dict[k] = float(v[-1])
                logger.info(f"  - Final {k}: {v[-1]:.6f}")
                
    # Save Model
    if args.output_model:
        out_path = Path(args.output_model)
    else:
        out_path = config.OUTPUT_DIR / "models" / f"joint_model_{args.predict_version}_{args.classifier_version}_{args.dataset_type}_{args.channel}_alpha{int(args.loss_alpha*100)}.h5"
        
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving joint model to {out_path}...")
    joint_model.save(str(out_path))
    
    # Save Diagnostic Plot
    should_plot = args.save_plot and (args.output_plot is None or (isinstance(args.output_plot, str) and args.output_plot.lower() not in ["none", "false"]) or args.output_plot is True)
    if should_plot and history and history.history:
        if args.output_plot and isinstance(args.output_plot, str) and args.output_plot.lower() not in ["none", "false"]:
            plot_file = Path(args.output_plot)
        else:
            plot_file = out_path.with_suffix(".png")
        try:
            plot_title = f"Joint Model ({args.predict_version.upper()}+{args.classifier_version.upper()}) | α={args.loss_alpha:.2f} | {args.dataset_type.upper()} ({args.channel})"
            plot_joint_training_history(history, plot_file, title_prefix=plot_title)
        except Exception as pe:
            logger.error(f"Failed to generate joint training plot: {pe}")
            
    # Save metrics JSON
    if args.metrics_json_path and metrics_dict:
        logger.info(f"Saving metrics JSON to {args.metrics_json_path}...")
        with open(args.metrics_json_path, "w") as f:
            json.dump(metrics_dict, f, indent=2)
            
    logger.info("Joint training pipeline finished successfully!")

if __name__ == "__main__":
    main()
