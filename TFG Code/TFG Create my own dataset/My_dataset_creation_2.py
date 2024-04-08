import os
import pandas as pd
from scipy.signal import savgol_filter
import numpy as np
from datetime import datetime
from pandas._libs.tslibs.timestamps import Timestamp
import json
# %%

# Obtiene la ruta absoluta del directorio actual
ruta_actual = os.path.dirname(os.path.abspath(__file__))
# Nombre del archivo CSV
nombre_archivo = 'medidas_oct_2020.csv'
# Une la ruta actual con el nombre del archivo CSV
ruta_archivo = os.path.join(ruta_actual, nombre_archivo)
# Lee el archivo CSV
df = pd.read_csv(ruta_archivo, sep=";")

# %% Preprocesamiento

# Selección de variables relevantes
data = df[['datetime', 'luminosidad', 'temperatura',
           'humedad_rel', 'temp_suelo', 'electrocond', 'var_diam']]

date_var = df[['date']]

# Preprocesados de los datos
datos = df[['var_diam']]
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
data = df[['datetime', 'luminosidad', 'temperatura',
           'humedad_rel', 'temp_suelo', 'electrocond', 'var_diam_sua_py']]

# %% Dividimos en vectores

df_dic = df.to_dict()

# %%
a_temperatura = np.array(
    list(df_dic['temperatura'].values()), dtype=np.float32)
a_luminosidad = np.array(
    list(df_dic['luminosidad'].values()), dtype=np.float32)
a_humedad_rel = np.array(
    list(df_dic['humedad_rel'].values()), dtype=np.float32)
a_temp_suelo = np.array(list(df_dic['temp_suelo'].values()), dtype=np.float32)
a_electrocond = np.array(
    list(df_dic['electrocond'].values()), dtype=np.float32)
a_var_diam_sua_py = np.array(
    list(df_dic['var_diam_sua_py'].values()), dtype=np.float32)

# Reducir los arrays a 15 elementos
array_train = [a_temperatura[:len(a_temperatura)-(48*2)], a_luminosidad[:len(a_temperatura)-(48*2)], a_humedad_rel[:len(a_temperatura)-(
    48*2)], a_temp_suelo[:len(a_temperatura)-(48*2)], a_electrocond[:len(a_temperatura)-(48*2)], a_var_diam_sua_py[:len(a_temperatura)-(48*2)]]
array_test = [a_temperatura[:len(a_temperatura)-(48)], a_luminosidad[:len(a_temperatura)-(48)], a_humedad_rel[:len(a_temperatura)-(
    48)], a_temp_suelo[:len(a_temperatura)-(48)], a_electrocond[:len(a_temperatura)-(48)], a_var_diam_sua_py[:len(a_temperatura)-(48)]]
array_validation = [a_temperatura, a_luminosidad,
                    a_humedad_rel, a_temp_suelo, a_electrocond, a_var_diam_sua_py]

# %% Creamos diccionario de verdad

# Crear el diccionario
train = {}
# Asignar valores al diccionario
for i, array in enumerate(array_train):
    train[i] = {
        'feat_dynamic_real': None,
        # 'feat_static_cat' corregido a Array of uint64
        'feat_static_cat': np.array([i], dtype=np.uint64),
        'item_id': f'T{i+1}',  # 'item_id' se ajusta según el índice
        # 'start' corregido a _libs.tslibs.timestamps.Timestamp
        'start': Timestamp('2015-01-01 00:00:01'),
        # 'target' corregido a Array of float32
        'target': array.astype(np.float32)
    }

# Crear el diccionario
test = {}
# Asignar valores al diccionario
for i, array in enumerate(array_test):
    test[i] = {
        'feat_dynamic_real': None,
        # 'feat_static_cat' corregido a Array of uint64
        'feat_static_cat': np.array([i], dtype=np.uint64),
        'item_id': f'T{i+1}',  # 'item_id' se ajusta según el índice
        # 'start' corregido a _libs.tslibs.timestamps.Timestamp
        'start': Timestamp('2015-01-01 00:00:01'),
        # 'target' corregido a Array of float32
        'target': array.astype(np.float32)
    }

# Crear el diccionario
validation = {}
# Asignar valores al diccionario
for i, array in enumerate(array_validation):
    validation[i] = {
        'feat_dynamic_real': None,
        # 'feat_static_cat' corregido a Array of uint64
        'feat_static_cat': np.array([i], dtype=np.uint64),
        'item_id': f'T{i+1}',  # 'item_id' se ajusta según el índice
        # 'start' corregido a _libs.tslibs.timestamps.Timestamp
        'start': Timestamp('2015-01-01 00:00:01'),
        # 'target' corregido a Array of float32
        'target': array.astype(np.float32)
    }

# %% creamos el json

with open("train.jsonl", "w") as f:
    for line in train:
        f.write(json.dumps(line)+"\n")

with open("test.jsonl", "w") as f:
    for line in test:
        f.write(json.dumps(line)+"\n")

with open("validation.jsonl", "w") as f:
    for line in validation:
        f.write(json.dumps(line)+"\n")
