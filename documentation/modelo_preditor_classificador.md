# Arquitetura e Dinâmica do Modelo Conjunto: Preditor + Classificador de EEG

Este documento descreve a fundamentação matemática, a motivação neurofisiológica e a implementação computacional do **modelo multitarefa conjunto (preditor de sinal temporal + classificador de estado cognitivo/sonolência)** sobre séries temporais contínuas de eletroencefalograma (EEG).

---

## 1. Visão Geral e Motivação Multitarefa

A arquitetura proposta unifica em um único grafo computacional funcional do Keras duas tarefas complementares:
1. **Predição do Sinal Temporal (*Forecasting Head*):** Prediz as voltagens dos passos temporais imediatamente subsequentes do EEG ($x_{t+1}, \dots, x_{t+K}$). Essa tarefa atua como um **viés indutivo (*inductive bias*) e regularizador temporal**, forçando o espaço latente recorrente a preservar as oscilações biológicas fundamentais (ritmos $\theta, \alpha, \beta$);
2. **Classificação de Sonolência (*Classification Head*):** Mapeia o mesmo estado latente para a probabilidade contínua de sonolência do motorista ($\hat{y}_{\text{clf}} \in [0.0, 1.0]$), correspondente ao índice PERCLOS (*Percentage of Eye Closure*).

```
                                  [ENTRADA DO MODELO]
                       ┌─────────────────────────────────────────┐
                       │       Janela de Entrada do Sinal        │
                       │            X ∈ ℝ^(B × T × C)            │
                       │     (T = 100 passos, C = 1 ou 4 ch)     │
                       └────────────────────┬────────────────────┘
                                            │
                                            ▼
                       ┌─────────────────────────────────────────┐
                       │    Backbone Recorrente Compartilhado    │
                       │      LSTM ou GRU (latent_dim = 32)      │
                       │        return_sequences = False         │
                       └────────────────────┬────────────────────┘
                                            │
                                            ▼
                       ┌─────────────────────────────────────────┐
                       │      Vetor de Estado Latente (h_T)      │
                       │             h_T ∈ ℝ^(B × 32)            │
                       └───────────┬─────────────────┬───────────┘
                                   │                 │
                ┌──────────────────┘                 └──────────────────┐
                │                                                       │
                ▼ (Ramo 1: Preditor)                                    ▼ (Ramo 2: Classificador)
  ┌───────────────────────────┐                           ┌───────────────────────────┐
  │   Projeção Densa Linear   │                           │Batch Normalization (Head v1)│
  │ Dense(K × C, act='linear')│                           │   (Estabilização Lote)    │
  └─────────────┬─────────────┘                           └─────────────┬─────────────┘
                │                                                       │
                ▼                                                       ▼
  ┌───────────────────────────┐                           ┌───────────────────────────┐
  │   Sinal Futuro Predito    │                           │ 1ª Camada Densa + Dropout │
  │ ŷ_forecast ∈ ℝ^(B × K × C)│                           │Dense(h1, relu) + Drop(0.3)│
  └─────────────┬─────────────┘                           └─────────────┬─────────────┘
                │                                                       │
                ▼                                                       ▼
  ┌───────────────────────────┐                           ┌───────────────────────────┐
  │     Perda de Predição     │                           │ 2ª Camada Densa + Dropout │
  │  ℒ_forecast = MSE(y, ŷ)   │                           │Dense(h2, relu) + Drop(0.2)│
  └─────────────┬─────────────┘                           └─────────────┬─────────────┘
                │                                                       │
                │                                                       ▼
                │                                         ┌───────────────────────────┐
                │                                         │  Camada de Saída Sigmoide │
                │                                         │   Dense(1, act='sigmoid') │
                │                                         └─────────────┬─────────────┘
                │                                                       │
                │                                                       ▼
                │                                         ┌───────────────────────────┐
                │                                         │ Probabilidade Sonolência  │
                │                                         │    ŷ_clf ∈ [0.0, 1.0]     │
                │                                         └─────────────┬─────────────┘
                │                                                       │
                │                                                       ▼
                │                                         ┌───────────────────────────┐
                │                                         │  Perda de Classificação   │
                │                                         │   ℒ_clf = BCE(y_true, ŷ)  │
                │                                         └─────────────┬─────────────┘
                │                                                       │
                │                                                       ▼
                │                                         ┌───────────────────────────┐
                │                                         │   Equalização de Escala   │
                │                                         │   k · ℒ_clf  (k = 10.0)   │
                │                                         └─────────────┬─────────────┘
                │                                                       │
                │ Ponderação: (1 - α)                       Ponderação: α│
                └────────────────────┐             ┌────────────────────┘
                                     ▼             ▼
                       ┌─────────────────────────────────────────┐
                       │          Função de Perda Total          │
                       │    ℒ_total = α · (k · ℒ_clf) +          │
                       │              (1 - α) · ℒ_forecast       │
                       └────────────────────┬────────────────────┘
                                            │
                                            ▼ (Gradientes Conjuntos ∇_W)
                       ┌─────────────────────────────────────────┐
                       │       Retropropagação no Backbone       │
                       │        (Atualização do LSTM/GRU)        │
                       └─────────────────────────────────────────┘
```

---

## 2. Decomposição Camada por Camada da Arquitetura

O modelo é instanciado na função `create_joint_model` do pacote `scalogram_cnn_project.models_for_joint_training`. A seguir, detalha-se rigorosamente a operação matemática e conceitual de cada camada.

### 2.1. Camada de Entrada (`Input Layer`)

A camada de entrada recebe lotes de matrizes bidimensionais temporal-espaciais no formato $(B, T, C)$:
- $B$: Tamanho do lote (*batch size*, tipicamente 32 ou 64);
- $T$: Janela temporal de contexto passado (ex.: $T = 100$ amostras, correspondendo a $1.0\text{ s}$ de sinal reamostrado a $100\text{ Hz}$);
- $C$: Quantidade de canais de EEG (ex.: $C = 1$ para eletrodo único centro-parietal `CP2`, ou $C = 4$ na montagem espacial `['CP2', 'PZ', 'FT7', 'P2']`).

```python
# Trecho de src/scalogram_cnn_project/models_for_joint_training/model_joint.py (Linhas 79-80)
# 1. Shared Input Layer (Multi-Channel Spatial-Temporal Window)
inp = tf.keras.layers.Input(shape=(input_len, num_channels), name="signal_input")
```

---

### 2.2. Backbone Recorrente Compartilhado (`LSTM` ou `GRU`)

O sinal de entrada $\mathbf{X} \in \mathbb{R}^{B \times T \times C}$ é processado sequencialmente pela camada recorrente. Em cada passo temporal $t \in [1, T]$, a célula computa portas de esquecimento (*forget gates*), entrada (*input gates*) e atualização de estado oculto.

Configuração fundamental: `return_sequences=False`.
- Em vez de retornar toda a sequência de estados $[h_1, h_2, \dots, h_T]$, a camada descarta os estados intermediários e projeta estritamente o **vetor de estado latente final** $\mathbf{h}_T \in \mathbb{R}^{B \times \text{latent\_dim}}$ (com $\text{latent\_dim} = 32$ neurônios).
- $\mathbf{h}_T$ atua como o **resumo informacional compactado** de toda a dinâmica oscilatória observada no último segundo de sinal.
- Todos os pesos são inicializados do zero com distribuição uniforme de Glorot (`kernel_initializer="glorot_uniform"`), eliminando qualquer risco de vazamento de dados (*leak-free*).

```python
# Trecho de src/scalogram_cnn_project/models_for_joint_training/model_joint.py (Linhas 82-100)
# 2. Backbone Recorrente Compartilhado (Inicializado do zero com Glorot Uniform)

if p_ver == "v1" or "gru" in p_ver:
    # Camada Recorrente GRU: portas de reset e atualização compactas
    # return_sequences=False extrai estritamente o estado oculto final h_T
    recurrent_layer = tf.keras.layers.GRU(
        latent_dim,
        activation="tanh",
        return_sequences=False,
        kernel_initializer="glorot_uniform",
        name="shared_gru_backbone"
    )
else:
    # Camada Recorrente LSTM: portas de esquecimento, entrada e saída com célula de memória interna
    # return_sequences=False descarta passos intermediários e retém o resumo final h_T
    recurrent_layer = tf.keras.layers.LSTM(
        latent_dim,
        activation="tanh",
        return_sequences=False,
        kernel_initializer="glorot_uniform",
        name="shared_lstm_backbone"
    )

# Processa sequencialmente a janela temporal (batch, input_len, num_channels) -> h_T: (batch, latent_dim)    
latent_state = recurrent_layer(inp)
```

---

### 2.3. Ramo 1: Cabeça de Predição de Sinal (*Forecasting Head*)

A partir do vetor de estado latente compartilhado $\mathbf{h}_T$, o ramo preditivo reconstrói linearmente os valores de tensão futura do EEG:

$$\mathbf{\hat{y}}_{\text{forecast}} = \mathbf{W}_{\text{forecast}} \mathbf{h}_T + \mathbf{b}_{\text{forecast}}$$

- **Ativação Linear:** Como o sinal de EEG é uma grandeza bioelétrica contínua (padronizada via $z$-score com média nula e variância unitária), não há funções de ativação restritivas como Sigmoide ou ReLU na saída de previsão.
- Para horizonte de 1 passo ($K = 1$), a projeção produz diretamente o vetor de tensão do próximo instante temporal (10 ms à frente).
- Para horizontes múltiplos ($K > 1$), uma camada adicional `Reshape((forecast_steps, num_channels))` reconstitui a forma do bloco temporal futuro predito.

```python
# Trecho de src/scalogram_cnn_project/models_for_joint_training/model_joint.py (Linhas 102-119)
# 3. Ramo 1: Cabeça de Predição de Sinal (Projeção Linear de Voltagem)

if forecast_steps == 1:
    # Projeção densa direta para horizonte imediato de 1 passo (t + 1)
    # Entrada: latent_state h_T (batch, latent_dim) -> Saída: (batch, num_channels)
    # Ativação linear: preserva a escala bioelétrica contínua (z-score com média 0 e std 1)
    forecast_output = tf.keras.layers.Dense(
        num_channels,
        activation="linear",
        kernel_initializer="glorot_uniform",
        name="forecast_output"
    )(latent_state)
else:
    # Projeção densa linear para horizonte multi-passos (K passos x C canais)
    # Saída intermediária achatada: (batch, forecast_steps * num_channels)
    forecast_dense = tf.keras.layers.Dense(
        forecast_steps * num_channels,
        activation="linear",
        kernel_initializer="glorot_uniform",
        name="forecast_dense"
    )(latent_state)
    
    # Camada de Reestruturação (Reshape):
    # Converte o vetor 1D achatado de volta para o bloco temporal 3D: (batch, forecast_steps, num_channels)
    target_shape = (forecast_steps, num_channels) if num_channels > 1 else (forecast_steps,)
    forecast_output = tf.keras.layers.Reshape(target_shape, name="forecast_output")(forecast_dense)
```

---

### 2.4. Ramo 2: Cabeça de Classificação de Sonolência (*Classification Head*)

Concomitantemente à predição temporal, o mesmo estado latente $\mathbf{h}_T$ é alimentado na cabeça de classificação (*MLP Head*). 

> [NOTA]
> **Desacoplamento Independente de Versões (`predict_version` vs. `classifier_version`):**
> O modelo possui **dois eixos de versão completamente independentes**:
> - `predict_version`: define exclusivamente o **backbone recorrente** (`"v0"` para **LSTM**, `"v1"` para **GRU**);
> - `classifier_version`: define exclusivamente a **cabeça de classificação** (`"v0"` para **MLP padrão**, `"v1"` para **MLP com Batch Normalization** e capacidade ampliada).
>
> Portanto, **o uso de Batch Normalization NÃO é exclusivo do GRU**:
> - `predict_version="v0"` + `classifier_version="v1"`: Backbone **LSTM** com Batch Normalization na cabeça classificadora;
> - `predict_version="v1"` + `classifier_version="v1"`: Backbone **GRU** com Batch Normalization na cabeça classificadora;
> - `predict_version="v0"` + `classifier_version="v0"`: Backbone **LSTM** sem Batch Normalization;
> - `predict_version="v1"` + `classifier_version="v0"`: Backbone **GRU** sem Batch Normalization.

As camadas da cabeça de classificação operam da seguinte forma:

1. **Normalização em Lote (`BatchNormalization`, ativada quando `classifier_version="v1"`):** 
   Estabiliza a distribuição das ativações do estado latente $\mathbf{h}_T$ ao longo dos minilotes antes de entrar nas camadas densas. Isso acelera a convergência e reduz a sensibilidade a variações de amplitude térmica e biológica entre diferentes janelas de EEG. Na versão padrão (`classifier_version="v0"`), o estado $\mathbf{h}_T$ segue diretamente para a primeira camada densa sem essa normalização;
2. **Primeira Camada Densa Não-Linear (`Dense(h1, activation='relu')`):** 
   Projeta o espaço latente de 32 dimensões para $h_1$ neurônios ($h_1 = 64$ em `v0`; $h_1 = 128$ em `v1`), permitindo que a rede capture combinações não-lineares complexas de potências espectrais associadas à transição para sonolência (aumento de teta e desaceleração de alfa);
3. **Primeira Camada de Dropout (`Dropout(d1)` com $p = 0.3$):** 
   Desativa aleatoriamente 30% dos neurônios durante o treino para prevenir co-adaptação e evitar sobreajuste em ruídos de eletrodo;
4. **Segunda Camada Densa com Dropout (`Dense(h2, activation='relu') + Dropout(d2)` com $p = 0.2$):** 
   Refina os atributos para um espaço de decisão compacto ($h_2 = 32$ em `v0`; $h_2 = 64$ em `v1`);
5. **Neurônio de Saída com Ativação Sigmoide (`Dense(1, activation='sigmoid')`):** 
   Converte o produto escalar final em uma probabilidade calibrada no intervalo $[0.0, 1.0]$:
   $$\hat{y}_{\text{clf}} = \sigma(z) = \frac{1}{1 + e^{-z}}$$
   onde $\hat{y}_{\text{clf}} \ge 0.5$ indica estado de fadiga/sonolência (índice PERCLOS).

```python
# Trecho de src/scalogram_cnn_project/models_for_joint_training/model_joint.py (Linhas 121-147)
# 4. Ramo 2: Cabeça de Classificação de Sonolência (MLP Head)

clf_x = latent_state

if c_ver == "v1":
    # Batch Normalization: Estabiliza a distribuição do estado latente entre minilotes
    # Entrada/Saída: (batch, latent_dim)
    clf_x = tf.keras.layers.BatchNormalization(name="classifier_bn_1")(clf_x)
    
# 1ª Camada Densa: Extrai combinações não-lineares dos ritmos espectrais
# Entrada: (batch, latent_dim) -> Saída: (batch, h1) [h1=64 em v0; 128 em v1]
clf_x = tf.keras.layers.Dense(
    h1,
    activation="relu",
    kernel_initializer="glorot_uniform",
    name="classifier_dense_1"
)(clf_x)

# 1º Dropout: Previne co-adaptação desligando 30% dos neurônios estocasticamente
clf_x = tf.keras.layers.Dropout(d1, name="classifier_dropout_1")(clf_x)

# 2ª Camada Densa: Refinamento para o espaço de decisão de sonolência
# Entrada: (batch, h1) -> Saída: (batch, h2) [h2=32 em v0; 64 em v1]
clf_x = tf.keras.layers.Dense(
    h2,
    activation="relu",
    kernel_initializer="glorot_uniform",
    name="classifier_dense_2"
)(clf_x)

# 2º Dropout: Regularização estocástica adicional com taxa de 20%
clf_x = tf.keras.layers.Dropout(d2, name="classifier_dropout_2")(clf_x)

# Camada de Saída: Neurônio único com ativação sigmoide calibrando a probabilidade de sonolência
# Entrada: (batch, h2) -> Saída: (batch, 1) com valores no intervalo contínuo [0.0, 1.0]
classification_output = tf.keras.layers.Dense(
    1,
    activation="sigmoid",
    kernel_initializer="glorot_uniform",
    name="classification_output"
)(clf_x)
```

---

## 3. Formulação da Função de Perda Conjunta e Equalização de Magnitude

### 3.1. O Problema da Disparidade de Ordens de Grandeza

O treinamento de redes multitarefa com objetivos heterogêneos sofre naturalmente com o problema de escala de grandezas numéricas entre perdas:
- A perda de classificação por **Entropia Cruzada Binária (BCE)** opera em escala logarítmica:
  $$\mathcal{L}_{\text{clf}} = -\frac{1}{B} \sum_{i=1}^B \left[ y_i \log \hat{y}_i + (1 - y_i) \log (1 - \hat{y}_i) \right] \approx 0.60 \text{ a } 0.70$$
- A perda de predição temporal por **Erro Quadrático Médio (MSE)** sobre as diferenças de voltagem padronizadas:
  $$\mathcal{L}_{\text{forecast}} = \frac{1}{B \cdot K \cdot C} \sum_{i=1}^B \sum_{k=1}^K \sum_{c=1}^C \left( y_{k,c}^{(i)} - \hat{y}_{k,c}^{(i)} \right)^2 \approx 6.0 \text{ a } 7.0$$

Sem balanceamento prévio, a tarefa de predição é **uma ordem de magnitude ($10\times$) maior** do que a de classificação. Como consequência, a retropropagação nos pesos do backbone recorrente é inteiramente monopolizada pela predição do sinal, resultando em desempenho subótimo na classificação da sonolência.

### 3.2. Fator Multiplicador de Equalização $k$ e Parâmetro $\alpha$

Para solucionar esse desequilíbrio e garantir que as derivadas parciais de ambas as cabeças atuem com intensidades proporcionais sobre o estado latente $\mathbf{h}_T$, introduz-se o fator multiplicador $k = 10.0$ e o coeficiente de alocação de gradiente $\alpha \in [0.0, 1.0]$:

$$\mathcal{L}_{\text{total}} = \alpha \cdot \left(k \cdot \mathcal{L}_{\text{clf}}\right) + (1 - \alpha) \cdot \mathcal{L}_{\text{forecast}}$$

Multiplicando $\mathcal{L}_{\text{clf}}$ por $k = 10.0$, seu valor atinge $10 \times 0.65 \approx 6.5$, equiparando-se perfeitamente aos $\approx 6.5$ da perda de MSE de previsão. Com as perdas em pé de igualdade numérica, $\alpha$ opera como um seletor exato de prioridade:
- $\alpha = 0.5$: Equilíbrio simétrico exato (50% do gradiente para classificação e 50% para predição);
- $\alpha = 0.7$: Modo focado em classificação (70% do gradiente para acurácia de sonolência, mantendo 30% de regularização indutiva pelo sinal);
- $\alpha = 1.0$: Treinamento puro de classificação (0% influência do preditor);
- $\alpha = 0.0$: Treinamento puro de predição temporal (0% influência do classificador).

```python
# Trecho de src/scalogram_cnn_project/models_for_joint_training/model_joint.py (Linhas 160-190)
# Loss_total = alpha * (loss_k * Loss_clf) + (1 - alpha) * Loss_forecast
loss_weights = {
    "classification_output": alpha * loss_k,   # e.g. 0.5 * 10.0 = 5.0
    "forecast_output": 1.0 - alpha             # e.g. 1.0 - 0.5 = 0.5
}

losses = {
    "classification_output": "binary_crossentropy",
    "forecast_output": "mse"
}

metrics = {
    "classification_output": [
        "accuracy",
        tf.keras.metrics.AUC(name="auc")
    ],
    "forecast_output": [
        "mae",
        tf.keras.metrics.RootMeanSquaredError(name="rmse")
    ]
}

model.compile(
    optimizer=optimizer,
    loss=losses,
    loss_weights=loss_weights,
    metrics=metrics
)
```

---

## 4. Alimentação dos Alvos Duplos via `JointSequence`

Para treinar o modelo multitarefa sem redundância e com embaralhamento eficiente a cada época, foi desenvolvido o gerador `JointSequence` (subclasse de `tf.keras.utils.Sequence`) em `src/scalogram_cnn_project/utils/joint_sequence.py`.

Em modelos multitarefa com múltiplas saídas nomeadas (`"forecast_output"` e `"classification_output"`), o Keras exige que o gerador de dados retorne, a cada passo, uma tupla composta por:
1. **Entrada $\mathbf{X}$:** Um tensor tridimensional com o lote de janelas temporais de EEG;
2. **Alvos $\mathbf{Y}$:** Um **dicionário de tensores**, onde cada chave corresponde com exatidão ao nome de uma das camadas de saída (`"forecast_output"` para a voltagem real futura e `"classification_output"` para o rótulo binário de sonolência PERCLOS).

O método `__getitem__` é o responsável por construir esse par entrada/alvos sob demanda para cada minilote:

```python
# Trecho de src/scalogram_cnn_project/utils/joint_sequence.py (Linhas 53-93)
def __getitem__(self, index):
    """
    Constrói e retorna um minilote completo para o treinamento multitarefa.
    Chamado automaticamente pelo Keras em cada passo do loop de treinamento (model.fit).
    
    Parâmetros:
        index (int): Índice do lote atual (varia de 0 até len(sequence) - 1).
    """
    # 1. Fatiamento dos índices deste minilote:
    # Utiliza a lista 'self.indexes' (que é reembaralhada aleatoriamente a cada época via on_epoch_end)
    batch_indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
    b_size = len(batch_indexes)
    
    # 2. Pré-alocação estática de matrizes NumPy preenchidas com zeros:
    # A pré-alocação evita a sobrecarga de memória e lentidão de operações como 'np.append' dentro do loop.
    # - batch_x: Entrada temporal -> Formato: (batch_size, input_len=100, num_channels=1)
    batch_x = np.zeros((b_size, self.input_len, self.num_channels), dtype=np.float32)
    # - batch_y_forecast: Alvo contínuo do preditor -> Formato: (batch_size, forecast_steps=1)
    batch_y_forecast = np.zeros((b_size, self.forecast_steps), dtype=np.float32)
    # - batch_y_clf: Alvo binário de sonolência do motorista -> Formato: (batch_size, 1)
    batch_y_clf = np.zeros((b_size, 1), dtype=np.float32)
    
    # 3. Preenchimento de cada uma das amostras do minilote:
    for i, item_idx in enumerate(batch_indexes):
        # Desempacota a tupla de metadados pré-computada para esta amostra:
        # - sig_idx: índice do sinal em memória (permite alternar entre múltiplos arquivos/sessões)
        # - start_idx: índice de início temporal t na série contínua
        # - y_forecast: valor de voltagem real no instante futuro t + input_len (alvo da previsão)
        # - y_clf: rótulo cognitivo PERCLOS correspondente àquela janela (0 = alerta, 1 = sonolento)
        sig_idx, start_idx, y_forecast, y_clf = self.indices_with_dual_labels[item_idx]
        sig = self.loaded_signals[sig_idx]
        
        # [Entrada X]: Extrai a fatia temporal contínua de 100 amostras (1.0 segundo de EEG)
        batch_x[i, :, 0] = sig[start_idx : start_idx + self.input_len].squeeze()
        
        # [Alvo 1 - Predição]: Armazena a voltagem real contínua do próximo instante (Loss: MSE)
        batch_y_forecast[i, 0] = float(np.squeeze(y_forecast))
            
        # [Alvo 2 - Classificação]: Armazena a classe de sonolência do condutor (Loss: BCE)
        batch_y_clf[i, 0] = float(y_clf)
        
    # 4. Retorno no formato multitarefa esperado pelo Keras:
    # O mapeamento chave-valor alimenta simultaneamente os dois ramos da rede.
    return batch_x, {
        "forecast_output": batch_y_forecast,      # Ramo 1 (Preditor de Voltagem)
        "classification_output": batch_y_clf      # Ramo 2 (Classificador de Sonolência)
    }
```

---

## 5. Dinâmica de Retropropagação no Treinamento

Durante a etapa de retropropagação (*backpropagation*), o gradiente total computado em relação aos parâmetros $\mathbf{W}_{\text{backbone}}$ do LSTM/GRU compartilhado é a combinação linear ponderada dos gradientes que chegam de ambos os ramos:

$$\nabla_{\mathbf{W}_{\text{backbone}}} \mathcal{L}_{\text{total}} = \alpha \cdot k \cdot \nabla_{\mathbf{W}_{\text{backbone}}} \mathcal{L}_{\text{clf}} + (1 - \alpha) \cdot \nabla_{\mathbf{W}_{\text{backbone}}} \mathcal{L}_{\text{forecast}}$$

- Se a cabeça de sonolência tentar direcionar os neurônios latentes para memorizar ruídos superficiais de alta frequência, o termo de previsão $(1 - \alpha) \cdot \nabla \mathcal{L}_{\text{forecast}}$ penaliza essa distorção caso ela prejudique a capacidade de prever a onda senoidal subjacente do EEG.
### 5.1. Monitoramento Granular em Tempo Real via `JointBatchProgressCallback`

Por se tratar de uma arquitetura multitarefa com objetivos heterogêneos (minimização do MSE da voltagem contínua e da entropia cruzada binária da sonolência), a barra de progresso padrão do Keras muitas vezes oculta a dinâmica relativa de cada componente durante a época.

Para inspecionar detalhadamente a convergência em tempo real a cada minilote, foi implementado o callback customizado `JointBatchProgressCallback` (derivado de `tf.keras.callbacks.Callback`) em `experiments/train_joint_model.py`:

```python
# Trecho de experiments/train_joint_model.py (Linhas 20-45)
class JointBatchProgressCallback(tf.keras.callbacks.Callback):
    """
    Callback customizado para monitoramento granular das métricas por minilote
    durante o treinamento do modelo conjunto (predição temporal + classificação).
    """
    def __init__(self, total_samples, batch_size):
        super().__init__()
        self.total_samples = total_samples  # Total de amostras na partição de treino
        self.batch_size = batch_size        # Tamanho nominal do minilote (ex.: 32 ou 64)
        self.processed = 0                  # Contador cumulativo de amostras processadas

    def on_epoch_begin(self, epoch, logs=None):
        # Reinicia o acumulador de amostras a cada nova época
        self.processed = 0
        logger.info(f"Epoch {epoch + 1} starting...")

    def on_train_batch_end(self, batch, logs=None):
        """
        Invocado automaticamente pelo Keras imediatamente após a retropropagação de cada lote.
        Extrai as métricas de ambos os ramos diretamente do dicionário 'logs'.
        """
        # 1. Ajuste dinâmico do tamanho do lote atual:
        # Garante exatidão caso o último minilote da partição tenha tamanho residual
        batch_size = logs.get('size', self.batch_size) if logs else self.batch_size
        self.processed = min(self.total_samples, self.processed + batch_size)
        
        # 2. Perda Total Ponderada:
        # L_total = alpha * (k * L_clf) + (1 - alpha) * L_forecast
        loss = logs.get('loss', 0.0) if logs else 0.0
        
        # 3. Acurácia de Classificação de Sonolência (PERCLOS):
        clf_acc = logs.get('classification_output_accuracy', 0.0) if logs else 0.0
        
        # 4. Erro Absoluto Médio (MAE) da Reconstrução de Sinal do Preditor:
        f_mae = logs.get('forecast_output_mae', 0.0) if logs else 0.0
        
        # 5. Impressão formatada em linha com flush imediato do buffer de saída:
        # O argumento 'flush=True' impede que a saída fique retida em buffer no terminal,
        # sendo fundamental para visualização imediata em sessões remotas e logs em background
        print(
            f"Processed {self.processed}/{self.total_samples} samples | "
            f"Total Loss: {loss:.4f} | Clf Acc: {clf_acc*100:.1f}% | Forecast MAE: {f_mae:.4f}",
            flush=True
        )
```

---

## 6. Tabela Resumo das Camadas e Dimensões dos Tensores

A tabela a seguir sumariza a trajetória dos tensores em uma execução com canal único ($C = 1$), janela de 100 amostras ($T = 100$) e horizonte de 1 passo ($K = 1$):

| Ordem | Nome da Camada | Tipo / Operação | Dimensão de Entrada | Dimensão de Saída | Papel no Modelo |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **1** | `signal_input` | `InputLayer` | - | $(B, 100, 1)$ | Recebe a janela de 1,0 s de EEG normalizado |
| **2** | `shared_backbone` | `LSTM` (`v0`) ou `GRU` (`v1`) | $(B, 100, 1)$ | $(B, 32)$ | Codifica a dinâmica temporal no estado latente $h_T$ |
| **3** | `forecast_output` | `Dense(1, activation='linear')` | $(B, 32)$ | $(B, 1)$ | Previsão linear de voltagem no instante $t+1$ (Loss: MSE) |
| **4\***| `classifier_bn_1` | `BatchNormalization` *(somente se `c_ver="v1"`)* | $(B, 32)$ | $(B, 32)$ | Estabilização inter-lotes do estado latente |
| **5** | `classifier_dense_1` | `Dense(h1, activation='relu')` | $(B, 32)$ | $(B, h_1)$ | Mapeamento não-linear de ritmos cerebrais ($64$ em `v0`, $128$ em `v1`) |
| **6** | `classifier_dropout_1` | `Dropout(0.3)` | $(B, h_1)$ | $(B, h_1)$ | Regularização estocástica |
| **7** | `classifier_dense_2` | `Dense(h2, activation='relu')` | $(B, h_1)$ | $(B, h_2)$ | Refinamento para espaço de decisão ($32$ em `v0`, $64$ em `v1`) |
| **8** | `classifier_dropout_2` | `Dropout(0.2)` | $(B, h_2)$ | $(B, h_2)$ | Regularização estocástica |
| **9** | `classification_output` | `Dense(1, activation='sigmoid')`| $(B, h_2)$ | $(B, 1)$ | Probabilidade de sonolência $\hat{y} \in [0, 1]$ (Loss: BCE) |

*\* A camada `classifier_bn_1` está presente exclusivamente quando `classifier_version="v1"`, independentemente de o backbone ser LSTM (`predict_version="v0"`) ou GRU (`predict_version="v1"`).*
