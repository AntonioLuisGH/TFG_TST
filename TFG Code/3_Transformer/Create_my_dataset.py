from functools import partial
import pandas as pd
from functools import lru_cache
from datasets import load_dataset


repo_name = "monash_tsf"
dataset_name = "tourism_monthly"
freq = "1M"
prediction_length = 24

# Load dataset
dataset = load_dataset(repo_name, dataset_name)

# Check the prediction_length
validation_example = dataset['validation'][0]
train_example = dataset['train'][0]
assert len(train_example["target"]) + prediction_length == len(
    validation_example["target"]
)

# Split the data
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# %% My turn

# %% Carga del dataset

df = pd.read_csv(C: \Users\anton\OneDrive\Escritorio\TFG_TST\TFG Old\TFG Old.csv', sep=";")

# %% Preprocesamiento

# Selección de variables relevantes
data = df[['datetime', 'luminosidad', 'temperatura',
           'humedad_rel', 'temp_suelo', 'electrocond', 'var_diam']]

date_var = df[['date']]

# Preprocesados de los datos
datos = df[['var_diam']]
