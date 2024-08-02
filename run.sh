#!/bin/bash

# Rodar a biblioteca instalada
poetry run nfl_predict confs/scrape.yaml &&
poetry run nfl_predict confs/modelling_train_data.yaml &&
poetry run nfl_predict confs/modelling_averages.yaml &&
poetry run nfl_predict confs/training_random_forest.yaml &&
poetry run nfl_predict confs/training_xgboost.yaml