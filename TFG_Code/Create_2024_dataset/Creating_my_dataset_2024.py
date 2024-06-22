# %% Creation of my file ".Dataset"

import os
import pandas as pd
from scipy.signal import savgol_filter
from datasets import Dataset


# %%
# Obtiene la ruta absoluta del directorio actual
ruta_actual = os.path.dirname(os.path.abspath(__file__))
# Nombre del archivo CSV
nombre_archivo = 'Sand_1.csv'
# Une la ruta actual con el nombre del archivo CSV
ruta_archivo = os.path.join(ruta_actual, nombre_archivo)
# Lee el archivo CSV
df = pd.read_csv(ruta_archivo, sep=";")

# %% Preprocesamiento

# Preprocesados de los datos
datos = df[['Temperature']]
ventana = 100  # Tamaño de la ventana de la media móvil
# Eliminamos la pendiente de nuestros datos
datos_sin = datos['Temperature'] - \
    datos['Temperature'].rolling(window=ventana).mean()
# Eliminamos los valores NaN
datos_sin_pendiente = datos_sin.dropna()
# Suavizamos la señal
datos_suavizados = savgol_filter(datos_sin_pendiente, 11, 2)
df = df[99:]
 df['Temperature'] = pd.Series(datos_suavizados, index=df.index)

# Selección de variables relevantes
data_validation = df[['Identificator', 'Date', 'Month', 'Day', 'Year', 'Hour', 'Minute', 'Second', 'Temperature', 'Relative_humidity',
                      'Light', 'Soil_temperature', 'Permittivity', 'Electroconductivity', 'Volumetric_water_content', 'Diameter', 'Battery_voltage']]
data_test = df[['Identificator', 'Date', 'Month', 'Day', 'Year', 'Hour', 'Minute', 'Second', 'Temperature', 'Relative_humidity', 'Light',
                'Soil_temperature', 'Permittivity', 'Electroconductivity', 'Volumetric_water_content', 'Diameter', 'Battery_voltage']].iloc[:-48]
data_train = df[['Identificator', 'Date', 'Month', 'Day', 'Year', 'Hour', 'Minute', 'Second', 'Temperature', 'Relative_humidity', 'Light',
                 'Soil_temperature', 'Permittivity', 'Electroconductivity', 'Volumetric_water_content', 'Diameter', 'Battery_voltage']].iloc[:-96]

# %% Converting dataframe to hugginface dataset

# Crear un diccionario vacío para almacenar los datos
dict_validation = {'start': [], 'target': [],
                   'feat_static_cat': [], 'feat_dynamic_real': [], 'item_id': []}

dict_test = {'start': [], 'target': [],
             'feat_static_cat': [], 'feat_dynamic_real': [], 'item_id': []}

dict_train = {'start': [], 'target': [],
              'feat_static_cat': [], 'feat_dynamic_real': [], 'item_id': []}

# Rellenar el diccionario con los datos correspondientes
for i in range(1, 7):
    dict_validation['start'].append(pd.Timestamp('2020-10-01 01:06:48'))
    dict_validation['target'].append(
        data_validation.iloc[:, i-1].values.astype('float32'))
    dict_validation['feat_static_cat'].append(i)
    dict_validation['feat_dynamic_real'].append(None)
    dict_validation['item_id'].append(f'T{i}')

    dict_test['start'].append(pd.Timestamp('2020-10-01 01:06:48'))
    dict_test['target'].append(
        data_test.iloc[:, i-1].values.astype('float32'))
    dict_test['feat_static_cat'].append(i)
    dict_test['feat_dynamic_real'].append(None)
    dict_test['item_id'].append(f'T{i}')

    dict_train['start'].append(pd.Timestamp('2020-10-01 01:06:48'))
    dict_train['target'].append(
        data_train.iloc[:, i-1].values.astype('float32'))
    dict_train['feat_static_cat'].append(i)
    dict_train['feat_dynamic_real'].append(None)
    dict_train['item_id'].append(f'T{i}')

# Convertir el diccionario en un DataFrame de pandas
dataframe_validation = pd.DataFrame(dict_validation)
dataframe_test = pd.DataFrame(dict_test)
dataframe_train = pd.DataFrame(dict_train)

# Convertir DataFrame de pandas a un Dataset de hugginface

dataset_validation = Dataset.from_pandas(dataframe_validation)
dataset_test = Dataset.from_pandas(dataframe_test)
dataset_train = Dataset.from_pandas(dataframe_train)
