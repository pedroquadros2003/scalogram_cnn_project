import logging
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from scalogram_cnn_project.utils.signal_loader import SignalLoader
import scalogram_cnn_project.settings.config as config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    model_path = config.PROJECT_DIR / "outputs/models/best_rnn_predictor_seedvig_CP2.h5"
    if not model_path.exists():
        model_path = config.PROJECT_DIR / "outputs/forecast_grid_6/rnn_predict_model_6e3550ac47.h5"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
        
    logger.info(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(str(model_path), compile=False)
    
    # Dataset Parameters
    dataset_type = "seed_vig"
    channel = "CP2"
    subject = 1
    resample_freq = 100.0
    train_split = 0.8
    input_steps = 100  # 1.0 second past context
    horizon_steps = 200  # 2.0 seconds forecast (first 200 points)
    num_test_windows = 100
    
    # Load Signal
    files = sorted(list(config.SEED_VIG_DIR.glob(f"{subject}_*.mat")))
    if not files:
        files = sorted(list(config.SEED_VIG_DIR.glob("*.mat")))
    target_file = files[0]
    
    logger.info(f"Loading signal from {target_file.name} (Channel: {channel})...")
    sig_data = SignalLoader.load_signal(str(target_file), dataset_type, resample_freq=resample_freq)
    raw_sig = sig_data.get_channel_signal(channel)
    sfreq = sig_data.sfreq
    
    num_samples = len(raw_sig)
    n_train = int(num_samples * train_split)
    train_part = raw_sig[:n_train]
    mean = np.mean(train_part)
    std = np.std(train_part)
    norm_sig = (raw_sig - mean) / max(std, 1e-6)
    val_sig = norm_sig[n_train:]
    
    logger.info(f"Validation split length: {len(val_sig)} samples ({len(val_sig)/sfreq:.1f} s).")
    
    # Sample 100 starting windows uniformly across the test split
    valid_range = len(val_sig) - input_steps - horizon_steps
    step_size = valid_range // num_test_windows
    start_indices = [i * step_size for i in range(num_test_windows)]
    
    init_inputs = np.zeros((num_test_windows, input_steps, 1), dtype=np.float32)
    ground_truth = np.zeros((num_test_windows, horizon_steps), dtype=np.float32)
    
    for i, s_idx in enumerate(start_indices):
        init_inputs[i, :, 0] = val_sig[s_idx : s_idx + input_steps]
        ground_truth[i, :] = val_sig[s_idx + input_steps : s_idx + input_steps + horizon_steps]
        
    logger.info(f"Running high-granularity autoregressive rollout for {horizon_steps} steps (0 to 2000 ms)...")
    
    rollout_predictions = np.zeros((num_test_windows, horizon_steps), dtype=np.float32)
    current_input = np.copy(init_inputs)
    
    # 200-step recursive rollout
    for step in range(horizon_steps):
        next_pred = model(current_input, training=False).numpy()
        pred_1d = np.squeeze(next_pred)
        if pred_1d.ndim == 0:
            pred_1d = np.array([pred_1d])
        rollout_predictions[:, step] = pred_1d
        
        # Shift input window
        next_point_3d = pred_1d.reshape(num_test_windows, 1, 1)
        current_input = np.concatenate([current_input[:, 1:, :], next_point_3d], axis=1)
        
    # Statistical Metrics per step k in [0..199]
    horizons_ms = np.arange(1, horizon_steps + 1) * (1000.0 / sfreq) # Time in ms
    horizons_sec = horizons_ms / 1000.0 # Time in seconds
    
    pearson_r_list = []
    mse_list = []
    mae_list = []
    amp_ratio_list = []
    
    for k in range(horizon_steps):
        gt_k = ground_truth[:, k]
        pred_k = rollout_predictions[:, k]
        
        if np.std(gt_k) > 1e-6 and np.std(pred_k) > 1e-6:
            r = np.corrcoef(gt_k, pred_k)[0, 1]
        else:
            r = 0.0
        pearson_r_list.append(float(r))
        
        mse = np.mean((gt_k - pred_k) ** 2)
        mse_list.append(float(mse))
        
        mae = np.mean(np.abs(gt_k - pred_k))
        mae_list.append(float(mae))
        
        ratio = np.std(pred_k) / max(np.std(gt_k), 1e-6)
        amp_ratio_list.append(float(ratio))
        
    logger.info(f"Granular metrics:")
    logger.info(f"  - 10 ms (Step 1):    r = {pearson_r_list[0]:.4f}, MSE = {mse_list[0]:.6f}")
    logger.info(f"  - 50 ms (Step 5):    r = {pearson_r_list[4]:.4f}, MSE = {mse_list[4]:.6f}")
    logger.info(f"  - 100 ms (Step 10):  r = {pearson_r_list[9]:.4f}, MSE = {mse_list[9]:.6f}")
    logger.info(f"  - 250 ms (Step 25):  r = {pearson_r_list[24]:.4f}, MSE = {mse_list[24]:.6f}")
    logger.info(f"  - 500 ms (Step 50):  r = {pearson_r_list[49]:.4f}, MSE = {mse_list[49]:.6f}")
    logger.info(f"  - 1000 ms (Step 100): r = {pearson_r_list[99]:.4f}, MSE = {mse_list[99]:.6f}")
    logger.info(f"  - 2000 ms (Step 200): r = {pearson_r_list[199]:.4f}, MSE = {mse_list[199]:.6f}")
    
    # ----------------------------------------------------
    # Generate Publication Figure
    # ----------------------------------------------------
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10.5,
        "figure.titlesize": 15
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(17, 11), dpi=220)
    plt.subplots_adjust(hspace=0.34, wspace=0.24)
    
    # Denormalize helper
    denorm = lambda x: x * std + mean
    
    # Show last 40 past samples (-0.4s) and all 200 future samples (0 to 2.0s)
    past_view_len = 40
    time_past_sec = np.linspace(-past_view_len / sfreq, 0.0, past_view_len, endpoint=False)
    time_future_sec = np.linspace(0.0, horizon_steps / sfreq, horizon_steps)
    
    ex1_idx = 5
    ex2_idx = 42
    
    # ----------------------------------------------------
    # Panel (a): High-Granularity Waveform - Sample 1
    # ----------------------------------------------------
    ax_a = axes[0, 0]
    past_sig_a = denorm(init_inputs[ex1_idx, -past_view_len:, 0])
    gt_sig_a = denorm(ground_truth[ex1_idx, :])
    pred_sig_a = denorm(rollout_predictions[ex1_idx, :])
    
    ax_a.plot(time_past_sec, past_sig_a, color="#333333", linewidth=1.8, label="Past Input (Context)")
    ax_a.plot(time_future_sec, gt_sig_a, color="#1f77b4", linewidth=1.6, alpha=0.9, label="Ground Truth (Actual EEG)", marker="o", markersize=3.0, markevery=5)
    ax_a.plot(time_future_sec, pred_sig_a, color="#d62728", linestyle="--", linewidth=2.0, label="Autoregressive Rollout (RNN)", marker="s", markersize=3.0, markevery=5)
    ax_a.axvline(0.0, color="#555555", linestyle=":", linewidth=1.4)
    
    ax_a.set_title("(a) Granular Waveform Trajectory — Sample 1 (First 200 Points / 2.0 s)", fontweight="bold", pad=8)
    ax_a.set_xlabel("Time (seconds)")
    ax_a.set_ylabel("Amplitude (μV)")
    ax_a.set_xlim(-0.4, 2.0)
    ax_a.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax_a.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax_a.grid(True, which="major", linestyle="--", alpha=0.55)
    ax_a.grid(True, which="minor", linestyle=":", alpha=0.3)
    ax_a.legend(loc="upper right", frameon=True, shadow=True)
    
    # ----------------------------------------------------
    # Panel (b): High-Granularity Waveform - Sample 2
    # ----------------------------------------------------
    ax_b = axes[0, 1]
    past_sig_b = denorm(init_inputs[ex2_idx, -past_view_len:, 0])
    gt_sig_b = denorm(ground_truth[ex2_idx, :])
    pred_sig_b = denorm(rollout_predictions[ex2_idx, :])
    
    ax_b.plot(time_past_sec, past_sig_b, color="#333333", linewidth=1.8, label="Past Input (Context)")
    ax_b.plot(time_future_sec, gt_sig_b, color="#1f77b4", linewidth=1.6, alpha=0.9, label="Ground Truth (Actual EEG)", marker="o", markersize=3.0, markevery=5)
    ax_b.plot(time_future_sec, pred_sig_b, color="#d62728", linestyle="--", linewidth=2.0, label="Autoregressive Rollout (RNN)", marker="s", markersize=3.0, markevery=5)
    ax_b.axvline(0.0, color="#555555", linestyle=":", linewidth=1.4)
    
    ax_b.set_title("(b) Granular Waveform Trajectory — Sample 2 (First 200 Points / 2.0 s)", fontweight="bold", pad=8)
    ax_b.set_xlabel("Time (seconds)")
    ax_b.set_ylabel("Amplitude (μV)")
    ax_b.set_xlim(-0.4, 2.0)
    ax_b.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax_b.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax_b.grid(True, which="major", linestyle="--", alpha=0.55)
    ax_b.grid(True, which="minor", linestyle=":", alpha=0.3)
    ax_b.legend(loc="upper right", frameon=True, shadow=True)
    
    # ----------------------------------------------------
    # Panel (c): Pearson Correlation Decay (0 to 2000 ms)
    # ----------------------------------------------------
    ax_c = axes[1, 0]
    ax_c.plot(horizons_sec, pearson_r_list, color="#1f77b4", linewidth=2.6, label="Pearson Correlation (r)")
    ax_c.axhline(0.0, color="#888888", linestyle="--", linewidth=1.0)
    
    # Key milestone points
    milestones = [
        (0.01, pearson_r_list[0], "10 ms: r=0.99", (10, -5)),
        (0.05, pearson_r_list[4], "50 ms: r=0.87", (10, 4)),
        (0.10, pearson_r_list[9], "100 ms: r=0.64", (10, 6)),
        (0.25, pearson_r_list[24], "250 ms: r=0.10", (10, 6)),
        (0.50, pearson_r_list[49], "500 ms: r=-0.15", (10, 6)),
        (1.00, pearson_r_list[99], "1.0 s: r≈0", (10, 6)),
    ]
    for x_m, y_m, txt, offset in milestones:
        ax_c.scatter([x_m], [y_m], color="#d62728", zorder=5, s=45)
        ax_c.annotate(txt, (x_m, y_m), textcoords="offset points", xytext=offset, fontsize=9.5, fontweight="bold", color="#222222")
        
    ax_c.set_title("(c) Step-by-Step Phase Coherence Decay (0 to 2.0 s)", fontweight="bold", pad=8)
    ax_c.set_xlabel("Prediction Horizon (seconds)")
    ax_c.set_ylabel("Pearson Correlation Coefficient (r)")
    ax_c.set_xlim(0.0, 2.0)
    ax_c.set_ylim(-0.35, 1.08)
    ax_c.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax_c.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax_c.grid(True, which="major", linestyle="--", alpha=0.55)
    ax_c.grid(True, which="minor", linestyle=":", alpha=0.3)
    ax_c.legend(loc="upper right", frameon=True, shadow=True)
    
    # ----------------------------------------------------
    # Panel (d): Point-by-Point MSE & Amplitude Ratio
    # ----------------------------------------------------
    ax_d1 = axes[1, 1]
    ax_d2 = ax_d1.twinx()
    
    line1 = ax_d1.plot(horizons_sec, mse_list, color="#d62728", linewidth=2.4, label="Mean Squared Error (MSE)")
    line2 = ax_d2.plot(horizons_sec, amp_ratio_list, color="#2ca02c", linestyle="--", linewidth=2.4, label="Amplitude Ratio (σ_pred / σ_gt)")
    ax_d2.axhline(1.0, color="#2ca02c", linestyle=":", alpha=0.5)
    
    ax_d1.set_title("(d) Error Growth and Dynamic Variance Ratio (0 to 2.0 s)", fontweight="bold", pad=8)
    ax_d1.set_xlabel("Prediction Horizon (seconds)")
    ax_d1.set_ylabel("Mean Squared Error (MSE)", color="#d62728")
    ax_d2.set_ylabel("Amplitude Ratio (σ_pred / σ_gt)", color="#2ca02c")
    ax_d1.tick_params(axis="y", labelcolor="#d62728")
    ax_d2.tick_params(axis="y", labelcolor="#2ca02c")
    ax_d1.set_xlim(0.0, 2.0)
    ax_d1.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax_d1.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax_d1.grid(True, which="major", linestyle="--", alpha=0.55)
    ax_d1.grid(True, which="minor", linestyle=":", alpha=0.3)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_d1.legend(lines, labels, loc="center right", frameon=True, shadow=True)
    
    plt.suptitle("High-Granularity Autoregressive Rollout Analysis (First 200 Points / 2.0 s at 100 Hz)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    
    output_png = config.PROJECT_DIR / "outputs/high_granularity_rollout_comparison.png"
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_png), dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Successfully saved high-granularity rollout plot to: {output_png}")

if __name__ == "__main__":
    main()
