import logging
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from scipy.io import loadmat

import scalogram_cnn_project.settings.config as config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_balanced_cohort_plot():
    balanced_subjects = [1, 4, 12, 14, 18]
    prev_subjects = [2, 4, 5, 14, 18]

    all_files = sorted(config.SEED_VIG_LABELS.glob("*.mat"))
    subj_data = {}
    for f in all_files:
        sid = int(f.stem.split("_")[0])
        mat = loadmat(str(f), squeeze_me=True, struct_as_record=False)
        p = np.array(mat["perclos"], dtype=np.float32)
        if sid not in subj_data:
            subj_data[sid] = []
        subj_data[sid].append(p)

    subj_perclos = {sid: np.concatenate(p_list) for sid, p_list in subj_data.items()}

    balanced_stats = []
    for sid in balanced_subjects:
        p = subj_perclos[sid]
        n_total = len(p)
        n_drowsy = int(np.sum(p >= 0.5))
        n_awake = n_total - n_drowsy
        pct_awake = 100.0 * n_awake / n_total
        pct_drowsy = 100.0 * n_drowsy / n_total
        balanced_stats.append({
            "sid": f"S{sid:02d}",
            "n_total": n_total,
            "n_awake": n_awake,
            "n_drowsy": n_drowsy,
            "pct_awake": pct_awake,
            "pct_drowsy": pct_drowsy,
            "p_vals": p,
        })

    tot_bal_awake = sum(s["n_awake"] for s in balanced_stats)
    tot_bal_drowsy = sum(s["n_drowsy"] for s in balanced_stats)
    tot_bal = tot_bal_awake + tot_bal_drowsy
    pct_bal_awake = 100.0 * tot_bal_awake / tot_bal
    pct_bal_drowsy = 100.0 * tot_bal_drowsy / tot_bal

    tot_prev_awake = sum(int(np.sum(subj_perclos[s] < 0.5)) for s in prev_subjects)
    tot_prev_drowsy = sum(int(np.sum(subj_perclos[s] >= 0.5)) for s in prev_subjects)
    tot_prev = tot_prev_awake + tot_prev_drowsy
    pct_prev_awake = 100.0 * tot_prev_awake / tot_prev
    pct_prev_drowsy = 100.0 * tot_prev_drowsy / tot_prev

    p_s02 = subj_perclos[2]
    n_s02_awake = int(np.sum(p_s02 < 0.5))
    n_s02_drowsy = int(np.sum(p_s02 >= 0.5))
    pct_s02_awake = 100.0 * n_s02_awake / len(p_s02)
    pct_s02_drowsy = 100.0 * n_s02_drowsy / len(p_s02)

    all_p = np.concatenate([p for p in subj_perclos.values()])
    tot_all_awake = int(np.sum(all_p < 0.5))
    tot_all_drowsy = int(np.sum(all_p >= 0.5))
    pct_all_awake = 100.0 * tot_all_awake / len(all_p)
    pct_all_drowsy = 100.0 * tot_all_drowsy / len(all_p)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10.0,
        "axes.labelsize": 11.0,
        "axes.titlesize": 11.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.0,
    })

    fig = plt.figure(figsize=(19.5, 6.4), dpi=240)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.25, 1.15, 1.10], wspace=0.32)

    color_awake = "#1f77b4"     # Steel blue
    color_drowsy = "#d62728"    # Crimson
    color_thresh = "#2ca02c"    # Green line

    # -------------------------------------------------------------
    # PANEL A: Per-subject Class Distribution
    # -------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])

    labels_a = [s["sid"] for s in balanced_stats] + ["Total Coorte\nBalanceada"]
    pct_awake_a = [s["pct_awake"] for s in balanced_stats] + [pct_bal_awake]
    pct_drowsy_a = [s["pct_drowsy"] for s in balanced_stats] + [pct_bal_drowsy]
    counts_a = [(s["n_awake"], s["n_drowsy"]) for s in balanced_stats] + [(tot_bal_awake, tot_bal_drowsy)]

    x = np.arange(len(labels_a))
    width = 0.58

    bars_awake = ax_a.bar(x, pct_awake_a, width, label="Awake (PERCLOS < 0.5)", color=color_awake, edgecolor="white", linewidth=1.2, alpha=0.92)
    bars_drowsy = ax_a.bar(x, pct_drowsy_a, width, bottom=pct_awake_a, label="Drowsy (PERCLOS ≥ 0.5)", color=color_drowsy, edgecolor="white", linewidth=1.2, alpha=0.92)

    ax_a.axhline(50.0, color="#111111", linestyle="--", linewidth=1.8, label="Equilíbrio Perfeito (50% / 50%)")

    for i in range(len(labels_a)):
        awk_pct = pct_awake_a[i]
        drw_pct = pct_drowsy_a[i]
        n_awk, n_drw = counts_a[i]

        ax_a.text(
            x[i],
            awk_pct / 2.0,
            f"{awk_pct:.1f}%\n({n_awk})",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=8.8,
        )
        ax_a.text(
            x[i],
            awk_pct + drw_pct / 2.0,
            f"{drw_pct:.1f}%\n({n_drw})",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=8.8,
        )

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels_a, fontweight="bold")
    ax_a.set_ylabel("Proporção de Rótulos (%)", fontweight="bold")
    ax_a.set_ylim(0, 106)
    ax_a.set_title("A. Proporção de Rótulos no Novo Gridsearch (Limiar = 0.50)", fontweight="bold", pad=9)
    ax_a.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=8.5)
    ax_a.grid(axis="y", linestyle=":", alpha=0.5)

    bars_awake[-1].set_edgecolor("#111111")
    bars_awake[-1].set_linewidth(2.2)
    bars_drowsy[-1].set_edgecolor("#111111")
    bars_drowsy[-1].set_linewidth(2.2)

    # -------------------------------------------------------------
    # PANEL B: Continuous PERCLOS Distribution (Violin + Boxplot)
    # -------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])

    violin_data = [s["p_vals"] for s in balanced_stats] + [np.concatenate([s["p_vals"] for s in balanced_stats])]
    x_b = np.arange(1, len(violin_data) + 1)

    parts = ax_b.violinplot(
        violin_data,
        positions=x_b,
        showmeans=True,
        showmedians=False,
        showextrema=True,
        widths=0.72
    )

    for pc in parts["bodies"]:
        pc.set_facecolor("#9467bd")
        pc.set_edgecolor("#4a148c")
        pc.set_alpha(0.60)
        pc.set_linewidth(1.2)
    parts["cmeans"].set_color("#d62728")
    parts["cmeans"].set_linewidth(2.0)
    parts["cmins"].set_color("#333333")
    parts["cmaxes"].set_color("#333333")
    parts["cbars"].set_color("#333333")

    ax_b.axhline(0.5, color=color_thresh, linestyle="--", linewidth=2.0, label="Limiar de Decisão (τ = 0.50)")
    ax_b.axhspan(0.0, 0.5, color=color_awake, alpha=0.08, label="Região Awake (PERCLOS < 0.50)")
    ax_b.axhspan(0.5, 1.0, color=color_drowsy, alpha=0.08, label="Região Drowsy (PERCLOS ≥ 0.50)")

    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(labels_a, fontweight="bold")
    ax_b.set_ylabel("Valor Contínuo PERCLOS [0.0, 1.0]", fontweight="bold")
    ax_b.set_ylim(-0.02, 1.05)
    ax_b.set_title("B. Densidade Contínua de PERCLOS e Limiar de Decisão", fontweight="bold", pad=9)
    ax_b.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=8.5)
    ax_b.grid(axis="y", linestyle=":", alpha=0.5)

    # -------------------------------------------------------------
    # PANEL C: Direct Balance Comparison
    # -------------------------------------------------------------
    ax_c = fig.add_subplot(gs[0, 2])

    cohort_names = [
        "Nova Coorte\nBalanceada\n[1, 4, 12, 14, 18]",
        "Coorte Anterior\n(Grid 4)\n[2, 4, 5, 14, 18]",
        "Sujeito 02\n(Outlier\nRemovido)",
        "Dataset Total\nSEED-VIG\n(21 Sujeitos)",
    ]

    awake_cohorts = [pct_bal_awake, pct_prev_awake, pct_s02_awake, pct_all_awake]
    drowsy_cohorts = [pct_bal_drowsy, pct_prev_drowsy, pct_s02_drowsy, pct_all_drowsy]
    dist_50 = [abs(50.0 - p) for p in drowsy_cohorts]

    x_c = np.arange(len(cohort_names))
    width_c = 0.54

    bars_c_awk = ax_c.bar(x_c, awake_cohorts, width_c, label="Awake", color=color_awake, edgecolor="white", linewidth=1.2, alpha=0.92)
    bars_c_drw = ax_c.bar(x_c, drowsy_cohorts, width_c, bottom=awake_cohorts, label="Drowsy", color=color_drowsy, edgecolor="white", linewidth=1.2, alpha=0.92)

    ax_c.axhline(50.0, color="#111111", linestyle="--", linewidth=1.8, label="50% / 50%")

    for i in range(len(cohort_names)):
        awk_pct = awake_cohorts[i]
        drw_pct = drowsy_cohorts[i]
        d = dist_50[i]

        ax_c.text(
            x_c[i],
            awk_pct / 2.0,
            f"{awk_pct:.1f}%",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=9.2,
        )
        ax_c.text(
            x_c[i],
            awk_pct + drw_pct / 2.0,
            f"{drw_pct:.1f}%",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=9.2,
        )
        color_tag = "#2ca02c" if d < 6.0 else ("#e65100" if d < 20.0 else "#c62828")
        ax_c.text(
            x_c[i],
            102.0,
            f"|Δ| = {d:.1f}%",
            ha="center",
            va="bottom",
            color=color_tag,
            fontweight="bold",
            fontsize=9.2,
        )

    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(cohort_names, fontsize=8.4, fontweight="bold")
    ax_c.set_ylabel("Proporção de Rótulos (%)", fontweight="bold")
    ax_c.set_ylim(0, 118)
    ax_c.set_title("C. Desbalanceamento Relativo (|Δ da Paridade|)", fontweight="bold", pad=9)
    ax_c.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=8.5)
    ax_c.grid(axis="y", linestyle=":", alpha=0.5)

    bars_c_awk[0].set_edgecolor("#2ca02c")
    bars_c_awk[0].set_linewidth(2.4)
    bars_c_drw[0].set_edgecolor("#2ca02c")
    bars_c_drw[0].set_linewidth(2.4)

    plt.suptitle(
        "Demonstração Empírica do Equilíbrio de Rótulos (PERCLOS ≥ 0.50) no Novo Gridsearch LOSO\n"
        "Eliminação do Viés da Classe Majoritária e Transição de 65.8% / 34.2% para 54.5% / 45.5%",
        fontsize=13.0,
        fontweight="bold",
        y=1.02,
    )

    out_dir = config.PROJECT_DIR / "documentation/assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "balanced_cohort_label_distribution.png"
    plt.savefig(str(out_path), dpi=240, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved balanced cohort label distribution plot to {out_path}")

    out_grid_dir = config.PROJECT_DIR / "outputs/joint_gridsearch_loso_balanced_stride8"
    if out_grid_dir.exists():
        import shutil
        shutil.copy2(out_path, out_grid_dir / "balanced_cohort_label_distribution.png")
        logger.info(f"Copied plot to {out_grid_dir / 'balanced_cohort_label_distribution.png'}")


if __name__ == "__main__":
    generate_balanced_cohort_plot()
