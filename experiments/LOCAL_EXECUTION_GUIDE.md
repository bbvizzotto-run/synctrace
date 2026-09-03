# Guia de Execução Local: SyncTrace Experimental Campaign

Este guia orienta a configuração e execução da campanha experimental SyncTrace no seu computador local equipado com uma **NVIDIA RTX 5070 Ti**.

## 1. Pré-requisitos do Sistema

Certifique-se de que os seguintes componentes estão instalados:
- **Drivers NVIDIA**: Versão 550+ (compatível com CUDA 12.x).
- **FFmpeg**: Necessário para extração de frames e áudio.
- **Python 3.12**: Recomendado o uso de Conda ou venv.

## 2. Configuração do Ambiente

Execute os seguintes comandos no seu terminal local:

```bash
# 1. Clonar o repositório (se ainda não o fez)
git clone https://github.com/bbvizzotto-run/synctrace.git
cd synctrace

# 2. Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# 3. Instalar dependências
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## 3. Diagnóstico de Hardware e Dados

Antes de iniciar o treinamento, execute o script de diagnóstico para validar a GPU e os caminhos dos datasets:

```bash
python scripts/check_local_env.py --fakeavceleb /caminho/para/fakeavceleb --avtimit /caminho/para/avtimit
```

## 4. Execução da Campanha

A campanha está configurada em `config/campaign.yaml`. Para a **RTX 5070 Ti**, otimizamos o batch size para aproveitar a VRAM disponível.

### Passo 1: Treinamento Principal (3 Sementes)
```bash
# Executa o treinamento para as sementes 42, 52 e 62
python src/engine/train.py --config config/campaign.yaml --seed 42
python src/engine/train.py --config config/campaign.yaml --seed 52
python src/engine/train.py --config config/campaign.yaml --seed 62
```

### Passo 2: Ablações
```bash
# Mamba-only ablation
python src/engine/train.py --config config/campaign.yaml --ablation mamba
# Attention-only ablation
python src/engine/train.py --config config/campaign.yaml --ablation attention
```

### Passo 3: Avaliação e Baselines
```bash
python src/engine/evaluate.py --checkpoint checkpoints/best_model.pth --dataset all
```

## 5. Coleta de Resultados

Após a conclusão, envie os arquivos gerados em `results/` e `logs/` para que eu possa processar as métricas e atualizar automaticamente as tabelas e gráficos do seu manuscrito.
