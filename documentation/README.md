# Documentação do Projeto `scalogram_cnn_project`

Esta pasta reúne a documentação técnica dos experimentos, metodologias de processamento de sinais EEG e guias de reprodução para colaboração científica.

---

## Índice de Documentos

1. [**Experimento de Predição e Rollout Autoregressivo (`experimento_predicao_autoregressiva.md`)**](file:///home/Workspace/documentation/experimento_predicao_autoregressiva.md)
   - Relatório técnico e científico completo com fundamentação matemática, formulação do problema de predição autoregressiva, protocolo de prevenção de data leakage, resultados quantitativos das buscas em grade (Grids 6, 7 e 8), análise da perda de coerência temporal e referências bibliográficas (IEEE/ABNT).

2. [**Guia de Compartilhamento com a Sarah (`guia_compartilhamento_sarah.md`)**](file:///home/Workspace/documentation/guia_compartilhamento_sarah.md)
   - Documento executivo direcionado à colaboradora Sarah contendo o roteiro de reprodução prática, explicação do fenômeno de atenuação de amplitude, localização de modelos pré-treinados e instruções de execução.

3. [**Script Executável de Rollout Autoregressivo (`codigo_predicao_autoregressiva.py`)**](file:///home/Workspace/documentation/codigo_predicao_autoregressiva.py)
   - Código Python autossuficiente e documentado que carrega o melhor modelo pré-treinado, executa a simulação em malha fechada sobre os dados reais do SEED-VIG e gera o gráfico diagnóstico de publicação em 4 painéis.

4. [**Resultado Gráfico Demonstrativo (`resultado_rollout_sarah.png`)**](file:///home/Workspace/documentation/resultado_rollout_sarah.png)
   - Figura de alta resolução gerada na execução de validação do pipeline.
