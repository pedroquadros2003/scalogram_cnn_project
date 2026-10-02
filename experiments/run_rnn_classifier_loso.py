import argparse
import itertools
import json
import logging
import os
import sys
import subprocess
from pathlib import Path
import numpy as np
import yaml

import scalogram_cnn_project.settings.config as config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def discover_dataset_subjects(dataset_type):
    """
    Discover all available subject IDs from the dataset directory.
    """
    if dataset_type == "seed_vig":
        data_dir = config.SEED_VIG_DIR
        pattern = "*.mat"
        files = list(data_dir.glob(pattern))
        subjects = sorted(list(set(int(f.stem.split("_")[0]) for f in files if "_" in f.stem)))
    else:
        data_dir = config.DROZY_DIR / "psg"
        pattern = "*.edf"
        files = list(data_dir.glob(pattern))
        subjects = sorted(list(set(int(f.stem.split("-")[0]) for f in files if "-" in f.stem)))
    return subjects


def plot_loso_evolution_overview(fold_metrics_dict, output_plot_path, title_prefix="LOSO Overview"):
    """
    Generates a 2-part multi-panel figure for Leave-One-Subject-Out cross-validation:
      Panel A: Overlapping Learning Curves (Loss and Accuracy across all folds with mean curve)
      Panel B: Subject-by-Subject Validation Accuracy Bar Chart (with Global Mean and Std band)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.gridspec import GridSpec

    subject_ids = sorted(list(fold_metrics_dict.keys()))
    if not subject_ids:
        logger.warning("No fold metrics available to plot.")
        return

    # Extract histories
    histories = [fold_metrics_dict[sid].get("history", {}) for sid in subject_ids]
    valid_histories = [h for h in histories if h and "loss" in h and "val_loss" in h]

    fig = plt.figure(figsize=(16, 11), dpi=180)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.1, 1.0], hspace=0.32, wspace=0.22)

    ax_loss = fig.add_subplot(gs[0, 0])
    ax_acc = fig.add_subplot(gs[0, 1])
    ax_bars = fig.add_subplot(gs[1, :])

    if valid_histories:
        num_epochs = max(len(h.get("loss", [])) for h in valid_histories)
        epochs_range = range(1, num_epochs + 1)
        train_losses = []
        val_losses = []

        # Panel A1: Multi-Fold Loss Dynamics
        for sid in subject_ids:
            h = fold_metrics_dict[sid].get("history", {})
            if "loss" in h and "val_loss" in h:
                t_loss = h["loss"]
                v_loss = h["val_loss"]
                e_r = range(1, len(t_loss) + 1)
                ax_loss.plot(e_r, t_loss, color="#1f77b4", alpha=0.25, linewidth=1.2)
                ax_loss.plot(e_r, v_loss, color="#ff7f0e", alpha=0.25, linewidth=1.2)
                train_losses.append(t_loss)
                val_losses.append(v_loss)

        if train_losses:
            # Pad and compute mean trajectories
            max_len = max(len(l) for l in train_losses)
            padded_train = [l + [l[-1]] * (max_len - len(l)) for l in train_losses]
            padded_val = [l + [l[-1]] * (max_len - len(l)) for l in val_losses]

            mean_train_loss = np.mean(padded_train, axis=0)
            mean_val_loss = np.mean(padded_val, axis=0)

            ax_loss.plot([], [], color="#1f77b4", alpha=0.35, linewidth=1.2, label="Train (Folds 1..N)")
            ax_loss.plot([], [], color="#ff7f0e", alpha=0.35, linewidth=1.2, label="Val (Folds 1..N)")
            ax_loss.plot(epochs_range, mean_train_loss, color="#1f77b4", linewidth=3.0, marker="o", markersize=5, label=f"Train Mean $\\mu$ ({mean_train_loss[-1]:.3f})")
            ax_loss.plot(epochs_range, mean_val_loss, color="#ff7f0e", linewidth=3.0, marker="s", markersize=5, label=f"Val Mean $\\mu$ ({mean_val_loss[-1]:.3f})")

        ax_loss.set_title("A1. Multi-Fold Loss Dynamics (BCE)", fontsize=13, fontweight="bold", pad=10)
        ax_loss.set_xlabel("Epoch", fontsize=11, fontweight="medium")
        ax_loss.set_ylabel("Binary Cross-Entropy Loss", fontsize=11, fontweight="medium")
        ax_loss.grid(True, linestyle="--", alpha=0.5)
        ax_loss.legend(loc="upper right", frameon=True, shadow=True, fontsize=9.5)
        ax_loss.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # -------------------------------------------------------------------------
    # Panel A2: Multi-Fold Accuracy Dynamics
    # -------------------------------------------------------------------------
    if valid_histories:
        train_accs = []
        val_accs = []

        for sid in subject_ids:
            h = fold_metrics_dict[sid].get("history", {})
            if "accuracy" in h and "val_accuracy" in h:
                # Scale to percentage 0-100%
                t_acc = [x * 100.0 if x <= 1.0 else x for x in h["accuracy"]]
                v_acc = [x * 100.0 if x <= 1.0 else x for x in h["val_accuracy"]]
                e_r = range(1, len(t_acc) + 1)
                ax_acc.plot(e_r, t_acc, color="#1f77b4", alpha=0.25, linewidth=1.2)
                ax_acc.plot(e_r, v_acc, color="#ff7f0e", alpha=0.25, linewidth=1.2)
                train_accs.append(t_acc)
                val_accs.append(v_acc)

        if train_accs:
            max_len = max(len(a) for a in train_accs)
            padded_train_acc = [a + [a[-1]] * (max_len - len(a)) for a in train_accs]
            padded_val_acc = [a + [a[-1]] * (max_len - len(a)) for a in val_accs]

            mean_train_acc = np.mean(padded_train_acc, axis=0)
            mean_val_acc = np.mean(padded_val_acc, axis=0)

            ax_acc.plot([], [], color="#1f77b4", alpha=0.35, linewidth=1.2, label="Train (Folds 1..N)")
            ax_acc.plot([], [], color="#ff7f0e", alpha=0.35, linewidth=1.2, label="Val (Folds 1..N)")
            ax_acc.plot(epochs_range, mean_train_acc, color="#1f77b4", linewidth=3.0, marker="o", markersize=5, label=f"Train Mean $\\mu$ ({mean_train_acc[-1]:.1f}%)")
            ax_acc.plot(epochs_range, mean_val_acc, color="#ff7f0e", linewidth=3.0, marker="s", markersize=5, label=f"Val Mean $\\mu$ ({mean_val_acc[-1]:.1f}%)")

        ax_acc.set_title("A2. Multi-Fold Accuracy Dynamics", fontsize=13, fontweight="bold", pad=10)
        ax_acc.set_xlabel("Epoch", fontsize=11, fontweight="medium")
        ax_acc.set_ylabel("Accuracy (%)", fontsize=11, fontweight="medium")
        ax_acc.set_ylim(0, 105)
        ax_acc.grid(True, linestyle="--", alpha=0.5)
        ax_acc.legend(loc="lower right", frameon=True, shadow=True, fontsize=9.5)
        ax_acc.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # -------------------------------------------------------------------------
    # Panel B: Subject-by-Subject Validation Accuracy (Bar Chart)
    # -------------------------------------------------------------------------
    val_accuracies = [fold_metrics_dict[sid].get("val_accuracy", fold_metrics_dict[sid].get("accuracy", 0.0)) * 100.0 for sid in subject_ids]
    x_positions = np.arange(len(subject_ids))
    
    mean_val_acc = float(np.mean(val_accuracies))
    std_val_acc = float(np.std(val_accuracies))

    # Color bars with subtle gradient or highlight
    colors = ["#2b5c8f" if acc >= mean_val_acc else "#5c84b0" for acc in val_accuracies]
    bars = ax_bars.bar(x_positions, val_accuracies, color=colors, width=0.62, edgecolor="#1c3d61", linewidth=1.2, zorder=3)

    # Reference lines for Mean and Std Dev band
    ax_bars.axhline(mean_val_acc, color="#d62728", linestyle="--", linewidth=2.2, label=f"Global Mean ($\\mu = {mean_val_acc:.1f}\\%$)", zorder=4)
    ax_bars.axhspan(max(0.0, mean_val_acc - std_val_acc), min(100.0, mean_val_acc + std_val_acc), color="#d62728", alpha=0.12, label=f"Std Dev ($\\pm \\sigma = {std_val_acc:.1f}\\%$)", zorder=2)

    # Annotate value atop each bar
    for bar, val in zip(bars, val_accuracies):
        height = bar.get_height()
        ax_bars.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1.2,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#1c3d61"
        )

    ax_bars.set_title("B. Individual Validation Accuracy per Subject (LOSO Cross-Validation)", fontsize=13, fontweight="bold", pad=12)
    ax_bars.set_xlabel("Validation Subject ID", fontsize=11, fontweight="medium")
    ax_bars.set_ylabel("Validation Accuracy (%)", fontsize=11, fontweight="medium")
    ax_bars.set_xticks(x_positions)
    ax_bars.set_xticklabels([f"S{sid}" for sid in subject_ids], fontsize=10)
    ax_bars.set_ylim(0, max(105.0, max(val_accuracies) + 8.0))
    ax_bars.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax_bars.legend(loc="upper right", frameon=True, shadow=True, fontsize=10.5)

    plt.suptitle(f"{title_prefix} - LOSO Cross-Validation Results", fontsize=15, fontweight="bold", y=0.98)
    
    out_p = Path(output_plot_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_p), dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved LOSO evolution overview multi-panel plot to: {out_p}")


def main():
    parser = argparse.ArgumentParser(description="Run Leave-One-Subject-Out (LOSO) Cross-Validation for RNN-coupled Classifier")
    parser.add_argument("--output_folder", "--output-folder", type=str, default="rnn_classifier_loso", help="Output folder inside OUTPUT_DIR")
    parser.add_argument("--params_file", "--params-file", "--config", type=str, default="configs/loso_rnn_classifier/seedvig_loso_v0.yaml", help="Path to YAML configuration file")
    parser.add_argument("--subjects", type=int, nargs="+", default=None, help="Explicit list of subject IDs to evaluate in LOSO (defaults to all discovered)")
    parser.add_argument("--dataset-type", type=str, default=None, help="Dataset override (seed_vig or drozy)")
    parser.add_argument("--channel", type=str, default=None, help="Channel name override")
    parser.add_argument("--lead-time-sec", type=float, default=None, help="Anticipation lead time override in seconds")
    parser.add_argument("--force-cpu", action="store_true", help="Force candidate executions to run on CPU")
    parser.add_argument("--save-plot", action=argparse.BooleanOptionalAction, default=True, help="Whether to generate per-fold and overview plots")
    parser.add_argument("--output-overview-plot", type=str, default=None, help="Custom path for overview plot")

    args = parser.parse_args()

    params_file_path = Path(args.params_file)
    if not params_file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {params_file_path}")

    with open(params_file_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Apply CLI overrides if present
    if args.dataset_type:
        base_config["dataset_type"] = args.dataset_type
    if args.channel:
        base_config["channel"] = args.channel
    if args.lead_time_sec is not None:
        base_config["lead_time_sec"] = args.lead_time_sec

    dataset_type = base_config.get("dataset_type", "seed_vig")
    channel = base_config.get("channel", "CP2")
    lead_time_sec = float(base_config.get("lead_time_sec", 0.0))

    # Discover subjects
    if args.subjects:
        loso_subjects = sorted(list(set(args.subjects)))
    elif "subjects" in base_config and base_config["subjects"]:
        if isinstance(base_config["subjects"], list):
            loso_subjects = sorted(base_config["subjects"])
        else:
            loso_subjects = [int(base_config["subjects"])]
    else:
        loso_subjects = discover_dataset_subjects(dataset_type)

    if not loso_subjects:
        raise ValueError(f"No subjects discovered for dataset '{dataset_type}'.")

    output_dir = config.OUTPUT_DIR / args.output_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = output_dir / "log.txt"
    file_handler = logging.FileHandler(log_file_path, mode="a")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    logger.info("=" * 70)
    logger.info("STARTING LEAVE-ONE-SUBJECT-OUT (LOSO) CROSS-VALIDATION")
    logger.info(f"  - Dataset: {dataset_type.upper()}")
    logger.info(f"  - Channel: {channel}")
    logger.info(f"  - Lead Time: {lead_time_sec:.1f}s")
    logger.info(f"  - Total LOSO Folds: {len(loso_subjects)} subjects ({loso_subjects})")
    logger.info(f"  - Output Folder: {output_dir}")
    logger.info("=" * 70)

    results_jsonl_path = output_dir / "loso_results.jsonl"
    summary_json_path = output_dir / "loso_summary.json"

    fold_metrics = {}

    for fold_idx, val_subj in enumerate(loso_subjects):
        logger.info("-" * 60)
        logger.info(f"=== FOLD {fold_idx + 1}/{len(loso_subjects)}: Validation Subject = S{val_subj} ===")
        logger.info("-" * 60)

        fold_dir = output_dir / f"fold_subj_{val_subj:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        fold_config = dict(base_config)
        fold_config["validation_subject"] = val_subj
        fold_config["save_plot"] = args.save_plot

        temp_config_path = fold_dir / "config.yaml"
        with open(temp_config_path, "w") as f:
            yaml.dump(fold_config, f, default_flow_style=False)

        metrics_json_path = fold_dir / "metrics.json"
        plot_path = fold_dir / "history.png"
        model_path = fold_dir / f"model_fold_subj_{val_subj:02d}.h5"

        cmd = [
            sys.executable, "experiments/train_rnn_classifier.py",
            "--config", str(temp_config_path),
            "--metrics-json-path", str(metrics_json_path),
            "--output-model", str(model_path),
            "--output-plot", str(plot_path),
            "--validation-subject", str(val_subj)
        ]
        if not args.save_plot:
            cmd.append("--no-save-plot")
        if args.force_cpu:
            cmd.append("--force-cpu")

        env = os.environ.copy()
        workspace_root = Path(__file__).parent.parent
        env["PYTHONPATH"] = str(workspace_root / "src") + os.pathsep + env.get("PYTHONPATH", "")

        try:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            with open(log_file_path, "a") as log_file:
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log_file.write(line)
                    log_file.flush()

            return_code = process.wait()

            if return_code == 0 and metrics_json_path.exists():
                with open(metrics_json_path, "r") as f:
                    metrics = json.load(f)

                fold_metrics[val_subj] = metrics
                val_acc = metrics.get("val_accuracy", metrics.get("accuracy", 0.0))
                val_loss = metrics.get("val_loss", metrics.get("loss", 0.0))
                logger.info(f"Fold S{val_subj} completed successfully. Val Accuracy: {val_acc*100:.2f}%, Val Loss: {val_loss:.4f}")

                # Save line to results.jsonl
                jsonl_entry = {
                    "fold": fold_idx + 1,
                    "validation_subject": val_subj,
                    "val_accuracy": val_acc,
                    "val_loss": val_loss,
                    "model_path": str(model_path),
                    "plot_path": str(plot_path) if args.save_plot else None,
                    "metrics": metrics
                }
                with open(results_jsonl_path, "a") as jf:
                    jf.write(json.dumps(jsonl_entry) + "\n")
            else:
                logger.error(f"Fold S{val_subj} failed with exit code {return_code}.")
                fold_metrics[val_subj] = {"val_accuracy": 0.0, "val_loss": 99.0, "status": "FAILED"}

        except Exception as e:
            logger.error(f"Exception during fold S{val_subj} execution: {e}")
            fold_metrics[val_subj] = {"val_accuracy": 0.0, "val_loss": 99.0, "status": "ERROR"}

    # Compute Global Summary Statistics
    logger.info("=" * 70)
    logger.info("LEAVE-ONE-SUBJECT-OUT (LOSO) VALIDATION FINISHED - CONSOLIDATED SUMMARY")
    logger.info("=" * 70)

    val_acc_list = [m.get("val_accuracy", m.get("accuracy", 0.0)) for m in fold_metrics.values() if "val_accuracy" in m or "accuracy" in m]
    val_loss_list = [m.get("val_loss", m.get("loss", 0.0)) for m in fold_metrics.values() if "val_loss" in m or "loss" in m]
    train_acc_list = [m.get("accuracy", 0.0) for m in fold_metrics.values() if "accuracy" in m]
    train_loss_list = [m.get("loss", 0.0) for m in fold_metrics.values() if "loss" in m]

    mean_val_acc = float(np.mean(val_acc_list)) if val_acc_list else 0.0
    std_val_acc = float(np.std(val_acc_list)) if val_acc_list else 0.0
    mean_val_loss = float(np.mean(val_loss_list)) if val_loss_list else 0.0
    std_val_loss = float(np.std(val_loss_list)) if val_loss_list else 0.0

    mean_train_acc = float(np.mean(train_acc_list)) if train_acc_list else 0.0
    mean_train_loss = float(np.mean(train_loss_list)) if train_loss_list else 0.0

    logger.info(f"Global Validation Accuracy: {mean_val_acc * 100.0:.2f}% ± {std_val_acc * 100.0:.2f}%")
    logger.info(f"Global Validation Loss:     {mean_val_loss:.4f} ± {std_val_loss:.4f}")
    logger.info(f"Global Training Accuracy:   {mean_train_acc * 100.0:.2f}%")
    logger.info(f"Global Training Loss:       {mean_train_loss:.4f}")

    summary_data = {
        "dataset_type": dataset_type,
        "channel": channel,
        "lead_time_sec": lead_time_sec,
        "total_folds": len(loso_subjects),
        "evaluated_subjects": loso_subjects,
        "mean_val_accuracy": mean_val_acc,
        "std_val_accuracy": std_val_acc,
        "mean_val_loss": mean_val_loss,
        "std_val_loss": std_val_loss,
        "mean_train_accuracy": mean_train_acc,
        "mean_train_loss": mean_train_loss,
        "per_subject_results": {sid: fold_metrics[sid] for sid in loso_subjects}
    }

    with open(summary_json_path, "w") as sf:
        json.dump(summary_data, sf, indent=2)
    logger.info(f"Saved LOSO summary JSON to: {summary_json_path}")

    # Generate Multi-Panel Overview Plot
    if args.save_plot:
        overview_plot_path = args.output_overview_plot or (output_dir / "loso_evolution_overview.png")
        plot_title = f"LOSO Anticipatory Classifier - {dataset_type.upper()} ({channel}) Lead {int(lead_time_sec)}s"
        try:
            plot_loso_evolution_overview(fold_metrics, overview_plot_path, title_prefix=plot_title)
        except Exception as pe:
            logger.error(f"Failed to generate overview plot: {pe}")

    logger.info("LOSO cross-validation workflow completed successfully.")


if __name__ == "__main__":
    main()
