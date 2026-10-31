# Predição e Rollout Autoregressivo em Sinais EEG Contínuos

Este documento descreve a formulação matemática, a metodologia e a implementação computacional do experimento de **predição e rollout autoregressivo em malha fechada (*closed-loop*)** sobre séries temporais contínuas de eletroencefalograma (EEG).

---

## 1. Visão Geral da Metodologia

Para avaliar a degradação da predição autoregressiva em malha fechada, foram amostradas uniformemente 100 janelas de avaliação ao longo da partição de validação não observada (os últimos 20% do sinal contínuo reamostrado a 100 Hz).

Para cada janela, a rede neural recorrente treinada recebe 1,0 segundo (100 amostras) de contexto passado padronizado e prevê recursivamente os 200 passos temporais subsequentes (até 2,0 segundos à frente), anexando cada nova predição à janela deslizante de entrada e descartando a amostra mais antiga.

A cada passo discreto de horizonte $k$ (de 10 ms a 2000 ms):
1. Os valores reais (*ground truth*) e as predições do modelo nas 100 janelas são compilados em vetores 100-dimensionais correspondentes:
   $$\mathbf{y}_k = [y_k^{(1)}, y_k^{(2)}, \dots, y_k^{(100)}]^T, \quad \mathbf{\hat{y}}_k = [\hat{y}_k^{(1)}, \hat{y}_k^{(2)}, \dots, \hat{y}_k^{(100)}]^T$$
2. O coeficiente de correlação de Pearson entre janelas ($r$) é computado para quantificar o decaimento da coerência de fase;
3. O Erro Quadrático Médio (MSE) é calculado como a média das diferenças quadráticas entre as 100 janelas para rastrear o crescimento do erro;
4. A razão de amplitude ($\sigma_{\text{pred}} / \sigma_{\text{gt}}$) é avaliada para medir o amortecimento e achatamento da oscilação ao longo do tempo.

---

## 2. Arquitetura do Modelo Preditivo

O modelo utilizado é uma rede recorrente baseada em **LSTM Direto** (`v0`) ou **GRU Direto** (`v1`) definida no pacote `scalogram_cnn_project.models_for_prediction`. O modelo comprime a sequência de entrada para um vetor de estado latente e o projeta diretamente no horizonte desejado via camada densa:

```python
# Trecho de src/scalogram_cnn_project/models_for_prediction/model_predict_v0.py
model = Sequential([
    # Entrada: (batch_size, input_steps=100, features=1)
    Input(shape=(input_steps, features)),
    
    # Camada Recorrente LSTM (return_sequences=False extrai o estado final h_t)
    LSTM(latent_dim, activation='tanh', return_sequences=False),
    
    # Projeção densa linear para o número de passos de saída
    Dense(output_steps * features),
    
    # Reestruturação para o formato sequencial de saída (batch_size, output_steps, 1)
    Reshape((output_steps, features))
])

model.compile(optimizer=optimizer, loss='mse')
```

No treinamento de horizonte 1-passo (`input_steps = 100`, `output_steps = 1`), o modelo aprende a mapear o histórico imediato no próximo instante $x_{t+1}$.

---

## 3. Carregamento e Padronização Anti-Vazamento (*Leak-Free*)

Para impedir contaminação de dados (*data leakage*), o sinal contínuo de EEG (canal `CP2`, sujeito 1 do dataset SEED-VIG a $f_s = 100.0\text{ Hz}$) é dividido estritamente por ordem cronológica. As estatísticas de média ($\mu_{\text{train}}$) e desvio padrão ($\sigma_{\text{train}}$) são calculadas **exclusivamente sobre os primeiros 80% do sinal**, sendo aplicadas para padronizar os 20% restantes (partição de validação):

```python
# Trecho de experiments/generate_high_granularity_rollout_plot.py
sig_data = SignalLoader.load_signal(str(target_file), dataset_type, resample_freq=resample_freq)
raw_sig = sig_data.get_channel_signal(channel)
sfreq = sig_data.sfreq

num_samples = len(raw_sig)
n_train = int(num_samples * train_split)  # train_split = 0.8
train_part = raw_sig[:n_train]

# Estatísticas estritamente calculadas no treino
mean = np.mean(train_part)
std = np.std(train_part)

# Padronização z-score sem vazamento temporal
norm_sig = (raw_sig - mean) / max(std, 1e-6)
val_sig = norm_sig[n_train:]  # Partição de validação isolada (últimos 20%)
```

---

## 4. Amostragem Uniforme das 100 Janelas de Avaliação

Para garantir significância estatística sobre a partição de validação, são extraídas 100 janelas distribuídas uniformemente ao longo do intervalo disponível:

```python
# Trecho de experiments/generate_high_granularity_rollout_plot.py
valid_range = len(val_sig) - input_steps - horizon_steps
step_size = valid_range // num_test_windows  # num_test_windows = 100
start_indices = [i * step_size for i in range(num_test_windows)]

# Matrizes de entrada inicial e gabarito real (ground truth)
init_inputs = np.zeros((num_test_windows, input_steps, 1), dtype=np.float32)
ground_truth = np.zeros((num_test_windows, horizon_steps), dtype=np.float32)

for i, s_idx in enumerate(start_indices):
    init_inputs[i, :, 0] = val_sig[s_idx : s_idx + input_steps]
    ground_truth[i, :] = val_sig[s_idx + input_steps : s_idx + input_steps + horizon_steps]
```

---

## 5. Rollout Autoregressivo em Malha Fechada (*Closed-Loop*)

No rollout em malha fechada, o modelo gera previsões de forma recursiva: a cada passo temporal, a nova saída $\hat{x}_{t+1}$ é concatenada à janela de contexto, deslocando o buffer temporal e descartando a observação mais antiga:

$$\mathbf{X}_{t+1} = [x_{t-K+2}, x_{t-K+3}, \dots, x_t, \hat{x}_{t+1}]$$

```python
# Trecho de experiments/generate_high_granularity_rollout_plot.py
rollout_predictions = np.zeros((num_test_windows, horizon_steps), dtype=np.float32)
current_input = np.copy(init_inputs)

# Rollout recursivo de 200 passos (0 a 2000 ms a 100 Hz)
for step in range(horizon_steps):
    # Inferência com o modelo recorrente (retorna o próximo ponto previsto)
    next_pred = model(current_input, training=False).numpy()
    pred_1d = np.squeeze(next_pred)
    if pred_1d.ndim == 0:
        pred_1d = np.array([pred_1d])
    rollout_predictions[:, step] = pred_1d

    # Atualização da janela deslizante: remove o mais antigo e anexa a predição
    next_point_3d = pred_1d.reshape(num_test_windows, 1, 1)
    current_input = np.concatenate([current_input[:, 1:, :], next_point_3d], axis=1)
```

---

## 6. Avaliação Vetorial das Métricas por Horizonte $k$

Para cada horizonte temporal discreto $k \in \{1, 2, \dots, 200\}$ ($10\text{ ms}$ a $2000\text{ ms}$), o desempenho é avaliado através dos vetores de 100 amostras:

```python
# Trecho de experiments/generate_high_granularity_rollout_plot.py
pearson_r_list = []
mse_list = []
amp_ratio_list = []

for k in range(horizon_steps):
    gt_k = ground_truth[:, k]              # Vetor real de 100 elementos no horizonte k
    pred_k = rollout_predictions[:, k]     # Vetor predito de 100 elementos no horizonte k

    # 1. Correlação de Pearson entre janelas (rastreamento de fase)
    if np.std(gt_k) > 1e-6 and np.std(pred_k) > 1e-6:
        r = np.corrcoef(gt_k, pred_k)[0, 1]
    else:
        r = 0.0
    pearson_r_list.append(float(r))

    # 2. Erro Quadrático Médio no passo k
    mse = np.mean((gt_k - pred_k) ** 2)
    mse_list.append(float(mse))

    # 3. Razão de Amplitude (sigma_pred / sigma_gt)
    ratio = np.std(pred_k) / max(np.std(gt_k), 1e-6)
    amp_ratio_list.append(float(ratio))
```

---

## 7. Análise dos Resultados e Degradação Temporal

A avaliação revelou três regimes dinâmicos bem delimitados:


   - $10\text{ ms}$ ($1\text{ passo}$): $r \approx 0.99$, com retenção total de amplitude ($\approx 99\%$).

   - $50\text{ ms}$ ($5\text{ passos}$): $r \approx 0.87$.
   
   - $100\text{ ms}$ ($10\text{ passos}$): $r \approx 0.64$.


   - Aos $250\text{ ms}$ ($25\text{ passos}$), a correlação decai para $r \approx 0.10$. Pequenos erros acumulados deslocam a fase predita em relação à fase biológica real, ocasionando interferência destrutiva.

   - A correlação atinge $r \approx 0.00$ a partir de $1.0\text{ s}$, e o MSE estabiliza em patamares elevados.

