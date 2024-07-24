#!/bin/bash

# Criar o script setup_and_run.sh
cat << 'EOF' > setup_and_run.sh
#!/bin/bash

# Criar um ambiente virtual
python3.12 -m venv venv
source .venv/bin/activate

# Instalar o Poetry
pip install poetry

# Fazer o build do projeto com Poetry
poetry build

# Instalar o arquivo .whl gerado na pasta dist
whl_file=$(ls dist/*.whl)
pip install "$whl_file"

# Rodar a biblioteca instalada
nfl_predict scrape.yml &&
nfl_predict modelling_train_data.yml &&
nfl_predict modelling_averages.yml &&
nfl_predict training.yml
EOF

# Tornar o script setup_and_run.sh executável
chmod +x setup_and_run.sh

# Executar o script setup_and_run.sh
./setup_and_run.sh