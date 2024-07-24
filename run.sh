#!/bin/bash

# Criar um ambiente virtual
python3.12 -m venv .venv
source .venv/bin/activate

# Instalar o Poetry
pip install poetry

# Fazer o build do projeto com Poetry
poetry build

# Instalar o arquivo .whl gerado na pasta dist
whl_file=$(ls dist/*.whl)
pip install "$whl_file"

# Rodar a biblioteca instalada
nfl_predict confs/scrape.yaml &&
nfl_predict confs/modelling_train_data.yaml &&
nfl_predict confs/modelling_averages.yaml &&
nfl_predict confs/training.yaml

# Desativar o ambiente virtual
deactivate