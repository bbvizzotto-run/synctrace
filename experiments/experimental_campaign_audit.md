# SyncTrace Experimental Campaign Audit

## Estado verificado em 18 de agosto de 2026

O repositório contém um esqueleto de carregadores para FakeAVCeleb e AV-LipSync-TIMIT, um conjunto de dados que gera pseudo-forgeries on-the-fly, uma configuração de treinamento e um workflow manual de experimentos. Entretanto, não há dados locais em `data/`, não há GPU NVIDIA disponível nesta máquina e o workflow do GitHub requer um runner `self-hosted` para a execução padrão de 25 épocas.

O `src/engine/train.py` atual processa apenas batches aleatórios; portanto, suas métricas não são resultados de FakeAVCeleb ou AV-LipSync-TIMIT e não podem substituir as tabelas do artigo. O `src/engine/benchmark.py` também é sintético por padrão. Embora `SyncTraceDataset` possa abrir clips reais, a rotina de treinamento ainda não o utiliza e a receita de pseudo-forgery do dataset precisa se tornar configurável para ablações de severidade.

FakeAVCeleb exige aceite de licença e liberação de um link de download pela equipe do conjunto de dados. AV-LipSync-TIMIT também não está presente localmente. Resultados reais requerem que os pacotes licenciados sejam disponibilizados em caminhos locais ou em armazenamento acessível pelo ambiente de execução.

## Consequência metodológica

Antes de qualquer número ser apresentado como resultado experimental, a campanha deve implementar splits por identidade, múltiplas sementes, checkpoints selecionados somente pela validação, avaliação in-domain e cross-dataset, e medições de eficiência sob o mesmo hardware e protocolo de entrada para SyncTrace e todos os baselines.
