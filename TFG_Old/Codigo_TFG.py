# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 10:45:16 2023

@author: benit
"""


# %% Importación de bibliotecas


import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, Flatten, Conv1D, MaxPooling1D, Dropout
from scipy.signal import savgol_filter
import time


# %% Carga del dataset

df = pd.read_csv(
    'C:/Users/anton/OneDrive/Escritorio/TFG_TST/TFG_Old/medidas_oct_2020.csv', sep=";")

# %% Preprocesamiento


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


# %% Transformación de datos

# Transformación de la variable 'date' en un objeto datetime
data['datetime'] = pd.to_datetime(data['datetime'])

# Tomamos los datos con cierto salto entre ellos (downsampling)
# data = data.iloc[::2]
# data = data.iloc[1::8]
# data = data.iloc[1::8]
# data = data.iloc[1::12]
# data = data.iloc[1::16]
date = data['datetime']
# Establecimiento de la variable 'date' como índice
data.set_index('datetime', inplace=True)
date_var = df[['date']]

# date_test = date_var.iloc[1::16]


# %% Normalización de los datos

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
scaled_data_y = scaler.fit_transform(data[['var_diam_sua_py']])
# scaled_data_y = scaler.fit_transform(data[['var_diam']])


def train_test_split(data, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        if out_end_ix > len(data):
            break
        seq_x, seq_y = data[i:end_ix, :-1], data[out_end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# %% Definición del número de pasos de tiempo de entrada y salida
n_steps_in = 50
n_steps_out = 1

# %% División de los datos en conjuntos de entrenamiento y prueba
train_size = int(len(scaled_data) * 0.80)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]
X_train, y_train = train_test_split(train_data, n_steps_in, n_steps_out)
X_test, y_test = train_test_split(test_data, n_steps_in, n_steps_out)

# date_test = date_test[train_size+n_steps_in+n_steps_out-2:]
# date_test = date_var[train_size+n_steps_in+n_steps_out-2:]

data1 = y_train
# %% Definición de la arquitectura del modelo
model = Sequential()
model.add(LSTM(16, activation='tanh', input_shape=(n_steps_in, 5)))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()

######################################################################################################################################################

# #Definición de la arquitectura del modelo
# model.add(SimpleRNN(16, activation='relu', input_shape=(n_steps_in, 5)))
# model.add(SimpleRNN(16, activation='sigmoid', input_shape=(n_steps_in, 5)))
# model.add(SimpleRNN(16, activation='linear', input_shape=(n_steps_in, 5)))
# model.add(SimpleRNN(16, activation='tanh', input_shape=(n_steps_in, 5)))
# model.add(LSTM(8, activation='sigmoid', input_shape=(n_steps_in, 5)))
# model.add(LSTM(8, activation='relu', input_shape=(n_steps_in, 5)))
# model.add(LSTM(8, activation='tanh', input_shape=(n_steps_in, 5)))

# #Definición de la arquitectura del modelo
# model = Sequential()
# # model.add(SimpleRNN(16, activation='tanh', input_shape=(n_steps_in, 5)))
# # model.add(LSTM(4, activation='tanh', input_shape=(n_steps_in, 5)))
# # model.add(LSTM(8, activation='relu', input_shape=(n_steps_in, 5)))
# model.add(LSTM(16, activation='relu', input_shape=(n_steps_in, 5)))
# model.add(Dropout(0.3))
# model.add(Dense(32,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')
# model.summary()

# #Definición de la arquitectura del modelo
# model = Sequential()
# model.add(SimpleRNN(16, activation='relu', input_shape=(n_steps_in, 5)))
# model.add(Dense(32,activation='relu'))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')
# model.summary()

# # Definición de la arquitectura del modelo ff
# model = Sequential()
# model.add(Flatten(input_shape=(n_steps_in, 5)))
# model.add(Dense(8, activation='sigmoid'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')
# model.summary()

# Definición de la arquitectura del modelo conv
# model = Sequential()
# model.add(Conv1D(filters=8, kernel_size=3, activation='sigmoid', input_shape=(n_steps_in, 5)))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')
# model.summary()


# # Definición de la arquitectura del modelo
# model = Sequential()
# model.add(SimpleRNN(16, activation='tanh', input_shape=(n_steps_in, 5)))
# # model.add(Dropout(0.3))
# model.add(Dense(32, activation='relu'))
# # model.add(Dropout(0.2))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')

####################################################################################################################################


# %% Entrenamiento del modelo

# Inicia el temporizador
start_time = time.time()

# Entrenamiento del modelo
history = model.fit(X_train, y_train, epochs=100, batch_size=16,
                    validation_data=(X_test, y_test), verbose=1)

# Finaliza el temporizador
end_time = time.time()

# Calcula el tiempo transcurrido en segundos
elapsed_time = end_time - start_time


# %% Predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test)

# Desnormalización de los datos
y_test_reshaped = y_test.reshape(-1, 1)
y_test = scaler.inverse_transform(y_test_reshaped)
y_pred = scaler.inverse_transform(y_pred)


# %% Cálculo de errores

# Cáculo de errores (Todas las muestras)
error = np.abs(y_pred - y_test)

#  Cálculo del error cuadrático medio (MSE) (Todas las muestras)
mse = np.mean((y_pred - y_test) ** 2)
print('MSE:', mse)
mse_log = np.log(mse)
print('MSE_log:', mse_log)

# Cálculo del coeficiente de determinación (R^2) (Todas las muestras)
ssr = np.sum((y_pred - y_test) ** 2)
sst = np.sum((y_test - np.mean(y_test)) ** 2)
r2 = 1 - (ssr / sst)
print('R^2:', r2)

# Cáculo de errores (100 primeras muestras)
y_test_100 = y_test[:100]
y_pred_100 = y_pred[:100]
error_100 = error[:100]

# Cálculo del error cuadrático medio (MSE) (100 primeras muestras)
mse = np.mean((y_pred_100 - y_test_100) ** 2)
print('MSE:', mse)
mse_log = np.log(mse)
print('MSE_log:', mse_log)

# Cálculo del coeficiente de determinación (R^2) (100 primeras muestras)
ssr = np.sum((y_pred_100 - y_test_100) ** 2)
sst = np.sum((y_test_100 - np.mean(y_test_100)) ** 2)
r2 = 1 - (ssr / sst)
print('R^2:', r2)


# %% Visualización de los resultados

# Visualización de los resultados (Todas las muestras)
plt.plot(y_test, label='Real')
plt.plot(y_pred, label='Predicción')
plt.legend()
plt.show()

# Visualización de los resultados (100 primeras muestras)
plt.plot(y_test_100, label='Real')
plt.plot(y_pred_100, label='Predicción')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()

val = history.history['val_loss']
train = history.history['loss']

# %%
figure, axes = plt.subplots()
axes.set_title("y_train")
axes.plot(y_train, color="blue")

# %%
figure, ax = plt.subplots(figsize=(12, 6))  # Ancho de 12 pulgadas y alto de 6 pulgadas
ax.set_title("data[['var_diam_sua_py']]")
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.plot(data[['var_diam_sua_py']], color="blue")

# %%
figure, ax = plt.subplots(figsize=(12, 6))  # Ancho de 12 pulgadas y alto de 6 pulgadas
ax.set_title("y_train")
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.plot(y_train, color="blue")

# %%
figure, ax = plt.subplots(figsize=(12, 6))  # Ancho de 12 pulgadas y alto de 6 pulgadas
vector_original = df['var_diam']  # Corregido: quita los corchetes adicionales
vector_filtrado = [elemento for elemento in vector_original if elemento >= 100]
plt.ylim(129, 132)
# Corregido: utiliza comillas simples para el nombre de la columna
ax.set_title("df['var_diam']")
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.plot(vector_filtrado, color="blue")
