import json
import logging
from pathlib import Path
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

from scalogram_cnn_project.utils.signal_loader import SignalLoader
import scalogram_cnn_project.settings.config as config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_cp2_5intervals_prediction_plot():
    cand_dir = config.PROJECT_DIR / "outputs/joint_gridsearch_loso_multichannel_pz/cand_00004_joint_4f1e62dc5f"
    channels_list = ["CP2", "PZ", "FT7", "P2"]
    resample_freq = 100.0
    input_len = 100
    baseline_sec = 120.0
    test_subj = 18

    subj_files = sorted(list(config.SEED_VIG_DIR.glob(f"{test_subj}_*.mat")))
    f = subj_files[0]
    logger.info(f"Loading test subject S{test_subj:02d} file: {f.name}...")
    sig_data = SignalLoader.load_signal(str(f), "seed_vig", resample_freq=resample_freq)
    sfreq = sig_data.sfreq
    sig = sig_data.get_channels_signal(channels_list)

    baseline_samples = int(baseline_sec * sfreq)
    mean = np.mean(sig[:baseline_samples], axis=0, keepdims=True)
    std = np.std(sig[:baseline_samples], axis=0, keepdims=True)
    std_full = np.std(sig, axis=0, keepdims=True)
    std = np.where(std <= 1e-6, np.where(std_full <= 1e-6, 1.0, std_full), std)
    norm_sig = (sig - mean) / std

    mean_cp2 = float(mean[0, 0])
    std_cp2 = float(std[0, 0])

    model_path = config.PROJECT_DIR / f"documentation/assets/model_fold_subj_{test_subj:02d}.h5"
    if not model_path.exists():
        model_path = cand_dir / f"fold_subj_{test_subj:02d}" / f"model_fold_subj_{test_subj:02d}.h5"
    logger.info(f"Loading model for fold S{test_subj:02d}: {model_path}...")
    model_s18 = tf.keras.models.load_model(str(model_path), compile=False)

    total_samples = len(norm_sig)  # 708,000 samples = 7080 s = 118.0 min (~2 hours)
    horizon_steps = 200            # 2.0 seconds at 100 Hz
    past_context_view = 40         # 0.4 seconds past context

    # 5 uniformly spaced intervals at 10%, 30%, 50%, 70%, 90% of the ~2-hour session
    interval_pcts = [0.10, 0.30, 0.50, 0.70, 0.90]
    interval_results = []

    for idx_int, pct in enumerate(interval_pcts):
        s_idx = int(total_samples * pct)
        t_start_sec = (s_idx + input_len) / sfreq
        t_start_min = t_start_sec / 60.0

        past_cp2_norm = norm_sig[s_idx + input_len - past_context_view : s_idx + input_len, 0]
        gt_cp2_norm = norm_sig[s_idx + input_len : s_idx + input_len + horizon_steps, 0]
        past_cp2_uv = past_cp2_norm * std_cp2 + mean_cp2
        gt_cp2_uv = gt_cp2_norm * std_cp2 + mean_cp2

        # Step-by-Step 10ms Continuous Signal Prediction over the 2.0s window
        X_open = np.zeros((horizon_steps, input_len, 4), dtype=np.float32)
        for k in range(horizon_steps):
            X_open[k] = norm_sig[s_idx + k : s_idx + k + input_len, :]
        out_open = model_s18(X_open, training=False)
        pred_open_cp2_norm = out_open["forecast_output"].numpy()[:, 0]
        pred_open_cp2_uv = pred_open_cp2_norm * std_cp2 + mean_cp2

        r_open = float(np.corrcoef(gt_cp2_norm, pred_open_cp2_norm)[0, 1])
        mae_open_norm = float(np.mean(np.abs(gt_cp2_norm - pred_open_cp2_norm)))
        mae_open_uv = float(np.mean(np.abs(gt_cp2_uv - pred_open_cp2_uv)))
        rmse_open_norm = float(np.sqrt(np.mean((gt_cp2_norm - pred_open_cp2_norm) ** 2)))
        rmse_open_uv = float(np.sqrt(np.mean((gt_cp2_uv - pred_open_cp2_uv) ** 2)))

        interval_results.append({
            "interval_num": idx_int + 1,
            "pct": int(pct * 100),
            "s_idx": s_idx,
            "t_start_sec": t_start_sec,
            "t_start_min": t_start_min,
            "past_cp2_uv": past_cp2_uv,
            "gt_cp2_uv": gt_cp2_uv,
            "pred_open_cp2_uv": pred_open_cp2_uv,
            "r_open": r_open,
            "mae_open_norm": mae_open_norm,
            "mae_open_uv": mae_open_uv,
            "rmse_open_norm": rmse_open_norm,
            "rmse_open_uv": rmse_open_uv,
        })

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10.5,
        "axes.labelsize": 11.0,
        "axes.titlesize": 11.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
    })

    fig2 = plt.figure(figsize=(18, 14.5), dpi=220)
    gs = GridSpec(5, 1, figure=fig2, hspace=0.48)

    time_past_sec = np.linspace(-past_context_view / sfreq, 0.0, past_context_view, endpoint=False)
    time_future_sec = np.linspace(0.0, horizon_steps / sfreq, horizon_steps)
    panel_letters = ["(a)", "(b)", "(c)", "(d)", "(e)"]

    for idx_p, res in enumerate(interval_results):
        ax_p = fig2.add_subplot(gs[idx_p, 0])

        all_uv = np.concatenate([res["past_cp2_uv"], res["gt_cp2_uv"], res["pred_open_cp2_uv"]])
        y_min_uv = float(np.min(all_uv))
        y_max_uv = float(np.max(all_uv))
        span_uv = max(y_max_uv - y_min_uv, 5.0)

        ax_p.plot(
            time_past_sec,
            res["past_cp2_uv"],
            color="#2b2b2b",
            linewidth=1.8,
            label="Contexto Passado (-0.4s a 0.0s)"
        )
        ax_p.plot(
            time_future_sec,
            res["gt_cp2_uv"],
            color="#1f77b4",
            linewidth=1.9,
            alpha=0.92,
            label="Sinal Real EEG CP2 (Ground Truth)",
            marker="o",
            markersize=3.0,
            markevery=5
        )
        ax_p.plot(
            time_future_sec,
            res["pred_open_cp2_uv"],
            color="#d62728",
            linestyle="--",
            linewidth=2.0,
            label="Predição do Modelo no Canal CP2 (1-Passo / 10 ms)",
            marker="s",
            markersize=3.0,
            markevery=5
        )
        ax_p.axvline(0.0, color="#555555", linestyle=":", linewidth=1.5, label="Início da Janela de Predição (t = 0.0s)")

        ax_p.set_ylim(y_min_uv - 0.15 * span_uv, y_max_uv + 0.22 * span_uv)

        ax_p.set_title(
            f"{panel_letters[idx_p]} Intervalo {res['interval_num']} ({res['pct']}% das 2h | t = {res['t_start_min']:.1f} min / {res['t_start_sec']:.0f} s, Δ = 2.0 s) — "
            f"Correlação r = {res['r_open']:.4f} | MAE = {res['mae_open_uv']:.2f} μV ({res['mae_open_norm']:.3f} σ) | RMSE = {res['rmse_open_uv']:.2f} μV",
            fontweight="bold",
            fontsize=11.0,
            pad=7
        )
        if idx_p == 4:
            ax_p.set_xlabel("Tempo Relativo no Intervalo de 2 Segundos (s)", fontsize=11.0)
        ax_p.set_ylabel("Amplitude CP2 (μV)", fontsize=10.5)
        ax_p.set_xlim(-0.4, 2.0)
        ax_p.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax_p.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
        ax_p.grid(True, which="major", linestyle="--", alpha=0.55)
        ax_p.grid(True, which="minor", linestyle=":", alpha=0.3)

        if idx_p == 0:
            ax_p.legend(loc="upper right", frameon=True, shadow=True, ncol=4, fontsize=9.2)

    plt.suptitle(
        "Evolução da Predição de Sinal do Modelo Conjunto (Preditor+Classificador) no Canal CP2\n"
        "5 Intervalos de 2 Segundos Uniformemente Espaçados ao Longo das 2 Horas (candidate_0004, Indivíduo de Teste S18)",
        fontsize=13.5,
        fontweight="bold",
        y=0.995
    )
    out_dir = config.PROJECT_DIR / "documentation/assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fig2 = out_dir / "cand_00004_cp2_5intervals_evolution.png"
    plt.savefig(str(out_fig2), dpi=240, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved updated Figure 2 (signal predictions only) to {out_fig2}")

    # Also save a copy to outputs if outputs directory exists
    outputs_fig2 = config.PROJECT_DIR / "outputs/cand_00004_cp2_5intervals_evolution.png"
    if outputs_fig2.parent.exists():
        import shutil
        shutil.copy2(out_fig2, outputs_fig2)


if __name__ == "__main__":
    generate_cp2_5intervals_prediction_plot()
