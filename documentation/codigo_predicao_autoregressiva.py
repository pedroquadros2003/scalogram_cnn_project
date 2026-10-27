"""
codigo_predicao_autoregressiva.py
==================================

Pipeline Executável de Demonstração e Rollout Autoregressivo (Closed-Loop)
para Previsão de Sinais EEG Contínuos com Redes Neurais Recorrentes (LSTM/GRU).

Compartilhado com: Sarah
Autor: Pedro Ulisses Quadros
Repositório: scalogram_cnn_project

Como Executar:
--------------
    python3 documentation/codigo_predicao_autoregressiva.py

Argumentos Opcionais Disponíveis:
---------------------------------
    python3 documentation/codigo_predicao_autoregressiva.py \
        --channel CP2 \
        --subject 1 \
        --input-steps 100 \
        --max-horizon 500 \
        --num-windows 100 \
        --output-plot documentation/resultado_rollout_sarah.png
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Garantir que o root do projeto e a pasta src estejam no sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scalogram_cnn_project.utils.signal_loader import SignalLoader
import scalogram_cnn_project.settings.config as config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("AutoregressivePrediction")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Executa predição autoregressiva (rollout closed-loop) de sinal EEG bruto."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Caminho para o modelo treinado .h5 (opcional; seleciona o melhor modelo pré-treinado por padrão)."
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="seed_vig",
        help="Tipo do dataset ('seed_vig' ou 'drozy'). Padrão: seed_vig."
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="CP2",
        help="Canal EEG analisado (ex: CP2, Pz, O1). Padrão: CP2."
    )
    parser.add_argument(
        "--subject",
        type=int,
        default=1,
        help="ID do sujeito experimental (SEED-VIG: 1 a 23). Padrão: 1."
    )
    parser.add_argument(
        "--resample-freq",
        type=float,
        default=100.0,
        help="Frequência de amostragem em Hz. Padrão: 100.0 Hz."
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fração cronológica de treino (0.8 = 80%% treino, 20%% teste). Padrão: 0.8."
    )
    parser.add_argument(
        "--input-steps",
        type=int,
        default=100,
        help="Número de passos de contexto passado (100 passos a 100 Hz = 1.0 segundo). Padrão: 100."
    )
    parser.add_argument(
        "--max-horizon",
        type=int,
        default=500,
        help="Número de passos futuros no rollout (500 passos a 100 Hz = 5.0 segundos). Padrão: 500."
    )
    parser.add_argument(
        "--num-windows",
        type=int,
        default=100,
        help="Número de janelas amostradas uniformemente na partição de validação para robustez estatística. Padrão: 100."
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default=str(PROJECT_ROOT / "documentation/resultado_rollout_sarah.png"),
        help="Caminho para salvar o gráfico diagnóstico gerado em alta resolução."
    )
    return parser.parse_args()


def localizar_modelo(caminho_informado):
    """Localiza o melhor modelo treinado disponível no repositório."""
    if caminho_informado and Path(caminho_informado).exists():
        return Path(caminho_informado)

    candidatos = [
        PROJECT_ROOT / "outputs/models/best_rnn_predictor_seedvig_CP2.h5",
        PROJECT_ROOT / "outputs/forecast_grid_6/rnn_predict_model_6e3550ac47.h5",
        PROJECT_ROOT / "outputs/forecast_grid_6/rnn_predict_model_3b863c33.h5",
    ]
    for cand in candidatos:
        if cand.exists():
            return cand

    # Procura o primeiro .h5 em forecast_grid_6
    grid6_models = list((PROJECT_ROOT / "outputs/forecast_grid_6").glob("*.h5"))
    if grid6_models:
        return grid6_models[0]

    raise FileNotFoundError(
        "Nenhum modelo pré-treinado .h5 foi encontrado em outputs/models ou outputs/forecast_grid_6/. "
        "Certifique-se de executar um treinamento ou baixar os pesos antes do rollout."
    )


def executar_rollout_autoregressivo(model, val_sig, input_steps, max_horizon, num_windows, sfreq):
    """
    Executa o rollout autoregressivo iterativo em malha fechada (closed-loop).
    
    A cada passo:
      1. Prevê o próximo valor x_hat(t+1) com base no buffer atual de 'input_steps'.
      2. Desloca o buffer, descartando o ponto mais antigo e inserindo x_hat(t+1).
      3. Repete o processo recursivamente até 'max_horizon' passos.
    """
    valid_range = len(val_sig) - input_steps - max_horizon
    if valid_range <= 0:
        raise ValueError("O sinal de validação é curto demais para o número de passos solicitado.")

    step_size = valid_range // num_windows
    start_indices = [i * step_size for i in range(num_windows)]

    # Matrizes para batching: (num_windows, input_steps, 1) e (num_windows, max_horizon)
    init_inputs = np.zeros((num_windows, input_steps, 1), dtype=np.float32)
    ground_truth = np.zeros((num_windows, max_horizon), dtype=np.float32)

    for i, s_idx in enumerate(start_indices):
        init_inputs[i, :, 0] = val_sig[s_idx : s_idx + input_steps]
        ground_truth[i, :] = val_sig[s_idx + input_steps : s_idx + input_steps + max_horizon]

    logger.info(f"Iniciando rollout autoregressivo para {max_horizon} passos ({max_horizon/sfreq:.1f} s)...")

    rollout_predictions = np.zeros((num_windows, max_horizon), dtype=np.float32)
    current_input = np.copy(init_inputs)

    for step in range(max_horizon):
        next_pred = model(current_input, training=False).numpy()
        pred_1d = np.squeeze(next_pred)
        if pred_1d.ndim == 0:
            pred_1d = np.array([pred_1d])

        rollout_predictions[:, step] = pred_1d

        # Realimentação em malha fechada: anexa nova predição ao fim do buffer
        next_point_3d = pred_1d.reshape(num_windows, 1, 1)
        current_input = np.concatenate([current_input[:, 1:, :], next_point_3d], axis=1)

        if (step + 1) % 100 == 0 or step == max_horizon - 1:
            logger.info(f"  -> Concluído passo {step + 1}/{max_horizon} ({(step + 1)/sfreq:.2f} s à frente)...")

    return init_inputs, ground_truth, rollout_predictions, start_indices


def calcular_metricas_por_horizonte(ground_truth, rollout_predictions, max_horizon, sfreq):
    """Calcula Correlação de Pearson, MSE, MAE e Razão de Amplitude ao longo do horizonte."""
    horizons_sec = np.arange(1, max_horizon + 1) / sfreq
    pearson_r = []
    mse_list = []
    mae_list = []
    amp_ratio = []

    for k in range(max_horizon):
        gt_k = ground_truth[:, k]
        pred_k = rollout_predictions[:, k]

        # Pearson Correlation
        if np.std(gt_k) > 1e-6 and np.std(pred_k) > 1e-6:
            r = np.corrcoef(gt_k, pred_k)[0, 1]
        else:
            r = 0.0
        pearson_r.append(float(r))

        # Erros
        mse_list.append(float(np.mean((gt_k - pred_k) ** 2)))
        mae_list.append(float(np.mean(np.abs(gt_k - pred_k))))

        # Razão de amplitude: desvio padrão predito / desvio padrão real
        ratio = np.std(pred_k) / max(np.std(gt_k), 1e-6)
        amp_ratio.append(float(ratio))

    return horizons_sec, pearson_r, mse_list, mae_list, amp_ratio


def gerar_grafico_diagnostico(
    horizons_sec,
    init_inputs,
    ground_truth,
    rollout_predictions,
    pearson_r,
    mse_list,
    amp_ratio,
    mean_trn,
    std_trn,
    input_steps,
    max_horizon,
    sfreq,
    output_path
):
    """Gera figura de publicação em 4 painéis ilustrando o comportamento do rollout."""
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

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=250)
    plt.subplots_adjust(hspace=0.34, wspace=0.24)

    time_past = np.linspace(-input_steps / sfreq, 0.0, input_steps)
    time_future = np.linspace(0.0, max_horizon / sfreq, max_horizon)

    denorm = lambda x: x * std_trn + mean_trn

    # Seleciona dois exemplos com comportamento demonstrativo
    ex1_idx = 5
    ex2_idx = min(42, ground_truth.shape[0] - 1)

    # Painel (a): Exemplo 1
    ax_a = axes[0, 0]
    ax_a.plot(time_past, denorm(init_inputs[ex1_idx, :, 0]), color="#1e293b", linewidth=1.6, label="Contexto Passado Real (1.0s)")
    ax_a.plot(time_future, denorm(ground_truth[ex1_idx, :]), color="#0284c7", linewidth=1.4, alpha=0.85, label="Sinal EEG Real (Ground Truth)")
    ax_a.plot(time_future, denorm(rollout_predictions[ex1_idx, :]), color="#dc2626", linestyle="--", linewidth=1.8, label="Rollout Autoregressivo (RNN)")
    ax_a.axvline(0.0, color="#64748b", linestyle=":", linewidth=1.2)
    ax_a.set_title("(a) Trajetória de Rollout Autoregressivo — Amostra 1", fontweight="bold", pad=8)
    ax_a.set_xlabel("Tempo (segundos)")
    ax_a.set_ylabel("Amplitude (μV)")
    ax_a.grid(True, linestyle="--", alpha=0.5)
    ax_a.legend(loc="upper right", frameon=True, shadow=True)

    # Painel (b): Exemplo 2
    ax_b = axes[0, 1]
    ax_b.plot(time_past, denorm(init_inputs[ex2_idx, :, 0]), color="#1e293b", linewidth=1.6, label="Contexto Passado Real (1.0s)")
    ax_b.plot(time_future, denorm(ground_truth[ex2_idx, :]), color="#0284c7", linewidth=1.4, alpha=0.85, label="Sinal EEG Real (Ground Truth)")
    ax_b.plot(time_future, denorm(rollout_predictions[ex2_idx, :]), color="#dc2626", linestyle="--", linewidth=1.8, label="Rollout Autoregressivo (RNN)")
    ax_b.axvline(0.0, color="#64748b", linestyle=":", linewidth=1.2)
    ax_b.set_title("(b) Trajetória de Rollout Autoregressivo — Amostra 2", fontweight="bold", pad=8)
    ax_b.set_xlabel("Tempo (segundos)")
    ax_b.set_ylabel("Amplitude (μV)")
    ax_b.grid(True, linestyle="--", alpha=0.5)
    ax_b.legend(loc="upper right", frameon=True, shadow=True)

    # Painel (c): Decaimento da Correlação de Pearson
    ax_c = axes[1, 0]
    ax_c.plot(horizons_sec, pearson_r, color="#0284c7", linewidth=2.5, label="Correlação Média de Pearson (r)")
    ax_c.axhline(0.0, color="#94a3b8", linestyle="--", linewidth=1.0)

    # Marcos temporais destacados
    milestones = [1, 10, 25, 50, 100]  # passos (10ms, 100ms, 250ms, 500ms, 1.0s)
    for m in milestones:
        if m - 1 < len(pearson_r):
            t_m = m / sfreq
            r_m = pearson_r[m - 1]
            ax_c.scatter([t_m], [r_m], color="#dc2626", zorder=5, s=35)
            ax_c.annotate(f"{int(t_m*1000)}ms: r={r_m:.2f}", (t_m, r_m), textcoords="offset points", xytext=(8, 4), fontsize=9)

    ax_c.set_title("(c) Perda de Coerência de Fase vs Horizonte Temporal", fontweight="bold", pad=8)
    ax_c.set_xlabel("Horizonte de Previsão (segundos)")
    ax_c.set_ylabel("Coeficiente de Pearson (r)")
    ax_c.set_ylim(-0.3, 1.08)
    ax_c.grid(True, linestyle="--", alpha=0.5)
    ax_c.legend(loc="upper right", frameon=True, shadow=True)

    # Painel (d): MSE e Razão de Amplitude
    ax_d1 = axes[1, 1]
    ax_d2 = ax_d1.twinx()

    l1 = ax_d1.plot(horizons_sec, mse_list, color="#dc2626", linewidth=2.4, label="Erro Quadrático Médio (MSE)")
    l2 = ax_d2.plot(horizons_sec, amp_ratio, color="#16a34a", linestyle="--", linewidth=2.4, label="Razão de Amplitude (σ_pred / σ_gt)")

    ax_d1.set_title("(d) Acumulação do Erro e Amortecimento da Amplitude", fontweight="bold", pad=8)
    ax_d1.set_xlabel("Horizonte de Previsão (segundos)")
    ax_d1.set_ylabel("MSE Normalizado", color="#dc2626")
    ax_d2.set_ylabel("Razão de Amplitude (σ_pred / σ_gt)", color="#16a34a")

    ax_d1.tick_params(axis="y", labelcolor="#dc2626")
    ax_d2.tick_params(axis="y", labelcolor="#16a34a")
    ax_d1.grid(True, linestyle="--", alpha=0.5)

    linhas = l1 + l2
    rotulos = [l.get_label() for l in linhas]
    ax_d1.legend(linhas, rotulos, loc="center right", frameon=True, shadow=True)

    plt.suptitle("Diagnóstico Experimental de Rollout Autoregressivo em Malha Fechada (100 Hz Raw EEG)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_file), dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Gráfico de publicação salvo com sucesso em: {out_file}")


def main():
    args = parse_arguments()

    logger.info("=================================================================")
    logger.info("   INICIANDO EXPERIMENTO DE PREDIÇÃO AUTOREGRESSIVA (ROLLOUT)   ")
    logger.info("=================================================================")

    # 1. Localizar e carregar modelo
    caminho_modelo = localizar_modelo(args.model_path)
    logger.info(f"Carregando modelo treinado de: {caminho_modelo}")
    model = tf.keras.models.load_model(str(caminho_modelo), compile=False)

    # 2. Localizar sinal SEED-VIG
    arquivos_sujeito = sorted(list(config.SEED_VIG_DIR.glob(f"{args.subject}_*.mat")))
    if not arquivos_sujeito:
        arquivos_sujeito = sorted(list(config.SEED_VIG_DIR.glob("*.mat")))
    if not arquivos_sujeito:
        raise FileNotFoundError(f"Nenhum arquivo .mat encontrado no diretório: {config.SEED_VIG_DIR}")

    arquivo_alvo = arquivos_sujeito[0]
    logger.info(f"Carregando sinal EEG: {arquivo_alvo.name} | Canal: {args.channel} | Fs: {args.resample_freq} Hz...")

    sig_data = SignalLoader.load_signal(str(arquivo_alvo), args.dataset_type, resample_freq=args.resample_freq)
    raw_sig = sig_data.get_channel_signal(args.channel)
    sfreq = sig_data.sfreq

    # 3. Particionamento estrito cronológico e normalização livre de vazamento
    num_amostras = len(raw_sig)
    n_train = int(num_amostras * args.train_split)
    train_part = raw_sig[:n_train]
    mean_trn = float(np.mean(train_part))
    std_trn = float(np.std(train_part))

    norm_sig = (raw_sig - mean_trn) / max(std_trn, 1e-6)
    val_sig = norm_sig[n_train:]
    logger.info(f"Sinal total: {num_amostras} amostras. Validação: {len(val_sig)} amostras ({len(val_sig)/sfreq:.1f} s).")

    # 4. Executar o rollout autoregressivo
    init_inputs, ground_truth, rollout_predictions, _ = executar_rollout_autoregressivo(
        model=model,
        val_sig=val_sig,
        input_steps=args.input_steps,
        max_horizon=args.max_horizon,
        num_windows=args.num_windows,
        sfreq=sfreq
    )

    # 5. Calcular métricas estatísticas
    horizons_sec, pearson_r, mse_list, mae_list, amp_ratio = calcular_metricas_por_horizonte(
        ground_truth=ground_truth,
        rollout_predictions=rollout_predictions,
        max_horizon=args.max_horizon,
        sfreq=sfreq
    )

    logger.info("-----------------------------------------------------------------")
    logger.info("RESUMO ESTATÍSTICO DO ROLLOUT AUTOREGRESSIVO:")
    logger.info(f"  • Passo 1   ( 10 ms): r = {pearson_r[0]:.4f} | MSE = {mse_list[0]:.6f} | Razão Amp = {amp_ratio[0]*100:.1f}%")
    logger.info(f"  • Passo 10  (100 ms): r = {pearson_r[9]:.4f} | MSE = {mse_list[9]:.6f} | Razão Amp = {amp_ratio[9]*100:.1f}%")
    logger.info(f"  • Passo 50  (500 ms): r = {pearson_r[49]:.4f} | MSE = {mse_list[49]:.6f} | Razão Amp = {amp_ratio[49]*100:.1f}%")
    logger.info(f"  • Passo 100 (1.0  s): r = {pearson_r[99]:.4f} | MSE = {mse_list[99]:.6f} | Razão Amp = {amp_ratio[99]*100:.1f}%")
    logger.info(f"  • Passo 500 (5.0  s): r = {pearson_r[-1]:.4f} | MSE = {mse_list[-1]:.6f} | Razão Amp = {amp_ratio[-1]*100:.1f}%")
    logger.info("-----------------------------------------------------------------")

    # 6. Gerar e salvar gráfico de publicação
    gerar_grafico_diagnostico(
        horizons_sec=horizons_sec,
        init_inputs=init_inputs,
        ground_truth=ground_truth,
        rollout_predictions=rollout_predictions,
        pearson_r=pearson_r,
        mse_list=mse_list,
        amp_ratio=amp_ratio,
        mean_trn=mean_trn,
        std_trn=std_trn,
        input_steps=args.input_steps,
        max_horizon=args.max_horizon,
        sfreq=sfreq,
        output_path=args.output_plot
    )

    logger.info("Experimento concluído com sucesso!")


if __name__ == "__main__":
    main()
