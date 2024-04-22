# %% Creation of my file ".Dataset"

import os
import pandas as pd
from scipy.signal import savgol_filter
from datasets import Dataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

# prediction lenth
prediction_length = 500

# Get the absolute path of the current directory
current_path = os.path.dirname(os.path.abspath(__file__))
# CSV file name
file_name = 'raw_data.csv'
# Concatenate the current path with the CSV file name
file_path = os.path.join(current_path, file_name)
# Read the CSV file
df = pd.read_csv(file_path, sep=";")


# Selección de variables relevantes
data = df[['luminosidad', 'temperatura',
           'humedad_rel', 'temp_suelo', 'electrocond', 'var_diam']]


def reemplazar_valores_incorrectos(serie):
    for i in range(1, len(serie) - 1):
        if serie[i] < 100:
            serie[i] = serie[i - 1]
    return serie


# Aplicar la función a la columna 'var_diam'
data['var_diam'] = reemplazar_valores_incorrectos(data['var_diam'])

# %%
# Preprocesados de los datos
datos = data[['var_diam']]
ventana = 100  # Tamaño de la ventana de la media móvil
# Eliminamos la pendiente de nuestros datos
datos_sin = datos['var_diam']-datos['var_diam'].rolling(window=ventana).mean()
# Eliminamos los valores NaN
datos_sin_pendiente = datos_sin.dropna()
# Suavizamos la señal
datos_suavizados = savgol_filter(datos_sin_pendiente, 11, 2)
df = df[99:]
df['var_diam_sua_py'] = pd.Series(datos_suavizados, index=df.index)

# Selección de variables relevantes
data = df[['luminosidad', 'temperatura',
           'humedad_rel', 'temp_suelo', 'electrocond', 'var_diam_sua_py']]

# %%
# Normalización de los datos
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
data[['luminosidad', 'temperatura',
      'humedad_rel', 'temp_suelo', 'electrocond', 'var_diam_sua_py']] = scaled_data

# Relevant Variable Selection
data_validation = data
data_test = data.iloc[:-prediction_length]
data_train = data.iloc[:-2*prediction_length]

# Create empty dictionaries to store the data
dict_validation = {'start': [], 'target': [],
                   'feat_static_cat': [], 'feat_dynamic_real': [], 'item_id': []}

dict_test = {'start': [], 'target': [],
             'feat_static_cat': [], 'feat_dynamic_real': [], 'item_id': []}

dict_train = {'start': [], 'target': [],
              'feat_static_cat': [], 'feat_dynamic_real': [], 'item_id': []}

# Populate the dictionaries with the corresponding data
for i in range(1, 7):

    dict_validation['target'].append(data_validation.iloc[:, i-1].values.astype('float32'))
    dict_test['target'].append(data_test.iloc[:, i-1].values.astype('float32'))
    dict_train['target'].append(data_train.iloc[:, i-1].values.astype('float32'))

    for d in [dict_validation, dict_test, dict_train]:
        d['start'].append(pd.Timestamp('2020-10-01 01:06:48'))
        d['feat_static_cat'].append(i)
        d['feat_dynamic_real'].append(None)
        d['item_id'].append(f'T{i}')

# Convert the dictionaries into pandas DataFrames
dataframe_validation = pd.DataFrame(dict_validation)
dataframe_test = pd.DataFrame(dict_test)
dataframe_train = pd.DataFrame(dict_train)

# Convert DataFrames into GluonTS Datasets
dataset_validation = Dataset.from_pandas(dataframe_validation)
dataset_test = Dataset.from_pandas(dataframe_test)
dataset_train = Dataset.from_pandas(dataframe_train)


# %% SHOWING DATA

test_dataset = dataset_test
train_dataset = dataset_train
# for var in range(6):
var = 5
# Show data
figure, axes = plt.subplots()
ejex = list(range(len(test_dataset[var]["target"])))
axes.plot(ejex[-prediction_length:], test_dataset[var]
          ["target"][-prediction_length:], color="red")
axes.plot(train_dataset[var]["target"], color="blue")

# ZOOM

# Show last values
figure, axes = plt.subplots()
ejex = list(range(len(test_dataset[var]["target"])))
# axes.plot(ejex[-prediction_length:], test_dataset[var]
#           ["target"][-prediction_length:], color="red")
# axes.plot(ejex[-3*prediction_length:-prediction_length], train_dataset[var]
#           ["target"][-2*prediction_length:], color="blue")

axes.plot(ejex[:prediction_length], train_dataset[var]
          ["target"][:prediction_length:], color="blue")


# %%

figure, ax = plt.subplots(figsize=(12, 6))  # Ancho de 12 pulgadas y alto de 6 pulgadas
vector = data1[['var_diam']]  # Corregido: quita los corchetes adicionales
plt.ylim(129, 132)
# Corregido: utiliza comillas simples para el nombre de la columna
ax.set_title("df['var_diam']")
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.plot(vector, color="blue")
