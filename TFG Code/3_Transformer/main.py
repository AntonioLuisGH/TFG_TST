# %% LIBRARIES

from Multi_Load_Dataset import load_and_preprocess_dataset
from TST_Define_Model import define_my_model
from Multi_Create_DataLoader import create_train_dataloader, create_backtest_dataloader
from Multi_Train_Model import train_model
from Multi_Evaluate_Model import forecasting, see_metrics, plot
import matplotlib.pyplot as plt

# %% DEFINE VARIABLES

freq = "1H"
prediction_length = 48

# %% LOAD, SPLIT AND PREPROCESS DATASET

multi_variate_train_dataset, multi_variate_test_dataset, num_of_variates, test_dataset = load_and_preprocess_dataset(
    freq, prediction_length)

# %% DEFINE MODEL

model = define_my_model(
    num_of_variates, multi_variate_train_dataset, freq, prediction_length)

# %% CREATE DATA LOADERS

train_dataloader = create_train_dataloader(
    config=model.config,
    freq=freq,
    data=multi_variate_train_dataset,
    batch_size=256,
    num_batches_per_epoch=100,
    num_workers=2,
)

test_dataloader = create_backtest_dataloader(
    config=model.config,
    freq=freq,
    data=multi_variate_test_dataset,
    batch_size=32,
)

# %% TRAIN MODEL

train_model(model, train_dataloader)

# %% INFERENCE

forecasts = forecasting(model, test_dataloader)
see_metrics(forecasts, test_dataset, prediction_length, freq, "metrics.txt")
plot(forecasts, 0, 5, multi_variate_test_dataset, freq, prediction_length)


# %%

# Datos
ts_index = 0  # Índice de la serie de tiempo
mv_index = 0  # Índice de la variable multivariante


# Vectores
# Tomando solo los primeros 1000 elementos
target_vector = multi_variate_test_dataset[ts_index]["target"][0][:1000]
# Tomando solo los primeros 1000 elementos
forecast_vector = forecasts[ts_index, ..., mv_index][0][:1000]

# Ajustar tamaño de la figura
plt.figure(figsize=(10, 6))  # Especifica el tamaño de la figura (ancho, alto)

# Crear la gráfica
plt.plot(target_vector, label='Target')
plt.plot(forecast_vector, label='Forecast')
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.title('Comparación entre Target y Forecast')
plt.legend()
plt.show()
