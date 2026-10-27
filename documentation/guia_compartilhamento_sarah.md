# Guia de Compartilhamento com a Sarah: Predição Autoregressiva em Sinais EEG

**Para:** Sarah  
**De:** Pedro Ulisses Quadros  
**Projeto:** `scalogram_cnn_project`  
**Data:** Outubro de 2026  
**Assunto:** Código e Documentação Técnica dos Experimentos de Predição e Rollout Autoregressivo  

---

Olá, Sarah!

Conforme combinamos, estruturei este guia e uma pasta dedicada de documentação para compartilhar com você todo o desenvolvimento, os resultados teóricos e práticos, e o código executável dos experimentos que realizei com **predição e rollout autoregressivo em sinais contínuos de EEG**.

Abaixo você encontra um resumo rápido do que foi feito, onde estão os arquivos, como rodar o código na sua máquina e os principais pontos científicos que podemos discutir em nossa próxima reunião.

---

## 1. O Que É Este Experimento?

No escopo da nossa pesquisa com análise de EEG e detecção de sonolência veicular (SEED-VIG e DROZY), investigamos se redes neurais recorrentes (**LSTM** e **GRU**) conseguem aprender a prever o comportamento futuro da onda de EEG bruta (sem depender de extração manual prévia de features).

Essa abordagem possui dois papéis fundamentais:
1. **Auto-Supervisão (*Self-Supervised Pre-Training*):** O modelo aprende a dinâmica intrínseca cerebral antes de ser acoplado ao classificador de fadiga/sonolência (*Joint Training*).
2. **Exploração dos Limites de Previsibilidade Física:** Até quantos milissegundos ou segundos à frente conseguimos prever um sinal bioelétrico contínuo antes que o caos e o ruído façam a predição perder a fase?

---

## 2. Mapa dos Arquivos Compartilhados

Dentro do projeto, criei a pasta `documentation/` com todo o material pronto:

```
documentation/
├── experimento_predicao_autoregressiva.md  # Relatório científico completo com equações, tabelas e referências
├── codigo_predicao_autoregressiva.py       # Script Python autossuficiente e executável
├── guia_compartilhamento_sarah.md          # Este guia de instruções e visão geral
└── resultado_rollout_sarah.png             # Gráfico gerado pelo script demonstrativo
```

Além disso, os modelos pré-treinados e os gráficos detalhados já estão salvos e disponíveis no repositório:
- **Melhor Modelo Pré-Treinado (Pesos Keras .h5):**  
  `outputs/models/best_rnn_predictor_seedvig_CP2.h5`  
  *(LSTM Direto, dimensão latente 32, otimizador Adam, lr=0.001, treinado em 100 Hz)*
- **Gráficos Diagnósticos em Alta Resolução:**
  - `outputs/autoregressive_rollout_diagnostic.png` (Visão geral de 4 painéis: trajetórias, decaimento de Pearson de 10ms a 5s, MSE e amplitude);
  - `outputs/high_granularity_rollout_comparison.png` (Zoom nos primeiros 2.0 segundos com resolução milissegundo a milissegundo);
  - `outputs/high_oscillation_samples_rollout.png` (Comportamento do modelo em janelas com ondas rítmicas bem definidas);
  - `outputs/autoregressive_rollout_cand_00050.png` (Gráfico anotado em português destacando as Zonas 1 e 2);
  - `outputs/tabela_grid_8_horizontes.jpg` (Tabela renderizada do Grid 8 com comparação de 1 a 50 passos).

---

## 3. Como Executar o Código

O script `documentation/codigo_predicao_autoregressiva.py` foi programado para ser autossuficiente, detectar os modelos automaticamente e rodar sem necessidade de ajustes manuais.

### Passo 1: Ativar o Ambiente Virtual
No terminal, na raiz do projeto:
```bash
source venv_wsl/bin/activate
```

### Passo 2: Executar o Script
Para rodar a demonstração com os parâmetros padrão (100 janelas no canal CP2, 5.0 segundos de rollout):
```bash
python3 documentation/codigo_predicao_autoregressiva.py
```

### Passo 3: Executar com Parâmetros Customizados (Opcional)
Se você quiser testar outros canais (ex.: `Pz`, `O1`), outro sujeito ou outros limites:
```bash
python3 documentation/codigo_predicao_autoregressiva.py \
    --channel CP2 \
    --subject 1 \
    --input-steps 100 \
    --max-horizon 500 \
    --num-windows 100 \
    --output-plot documentation/meu_teste_rollout.png
```

Ao terminar a execução, o terminal exibirá as métricas calculadas e salvará a figura comparativa no caminho especificado em `--output-plot`.

---

## 4. Principais Conclusões Científicas para Nós

Ao analisar o relatório e os gráficos, você notará alguns achados que fundamentam nossas decisões de modelagem:

1. **Predição em 1 Passo (10 ms a 100 Hz):**
   - O modelo tem desempenho praticamente perfeito ($r = 0.989$, erro $< 0.007$, retenção de amplitude de $99.94\%$).
   - A rede aprende muito bem a inércia e a derivada de curto prazo da onda.
2. **O Rollout em Malha Fechada (*Closed-Loop*):**
   - **Até 100 ms (10 passos):** A correlação de Pearson é sustentada ($r \approx 0.64$ a $0.99$). O modelo acompanha as oscilações Alfa/Teta.
   - **Entre 100 ms e 250 ms:** Ocorre rotação de fase (*phase shift*). Pequenos desvios temporais fazem a previsão entrar em oposição de fase com o sinal real.
   - **Após 250 ms (até 5.0 s):** A correlação cai para perto de zero e o traçado converge suavemente para uma linha reta na média do sinal.
3. **Por que a Linha Reta Ocorre? (Não é bug!)**
   - Na teoria de aprendizado estatístico, minimizar o MSE em sinais caóticos faz o modelo prever a **esperança condicional** $\mathbb{E}[y \mid X]$. À medida que a incerteza cresce com o tempo, a melhor previsão para não explodir o erro quadrático é a média neutra ($\mu = 0$).
4. **Impacto no Nosso Classificador Conjunto:**
   - Para a reconstrução de sinais longos sem achatamento, adotamos a estratégia **Freshest-Window** (janela de menor horizonte), que você pode conferir em detalhes no arquivo `documentation/experimento_predicao_autoregressiva.md`.

---

## 5. Como Sincronizar via Git

Se estivermos trabalhando no mesmo repositório remoto:
1. Eu fiz o commit dos novos arquivos de documentação e código no branch `main`.
2. Para baixar na sua máquina:
   ```bash
   git pull origin main
   ```
3. Toda a pasta `documentation/` e os scripts estarão disponíveis para você.

Qualquer dúvida sobre a matemática das perdas, a estrutura do código ou se quiser que testemos variações (como GRU vs LSTM ou inclusão de outros canais), me avise para conversarmos!

Abraços,  
Pedro
