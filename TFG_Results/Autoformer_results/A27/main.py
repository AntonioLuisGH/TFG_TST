# %% LIBRARIES

from T1_Load_Dataset import load_and_preprocess_dataset
from T2_Define_Model import define_my_model
from T3_Create_DataLoader import create_train_dataloader, create_backtest_dataloader
from T4_Train_Model import train_model
from T5_Evaluate_Model import forecasting, see_metrics, plot


# %% DEFINE VARIABLES

# Choose between "Transformer", "Informer", "Autoformer"
model_variant = "Autoformer"

freq = "7min52s"
prediction_length = 366
num_of_epochs = 10

# %% LOAD, SPLIT AND PREPROCESS DATASET

(multi_variate_train_dataset,
 multi_variate_test_dataset,
 num_of_variates,
 test_dataset) = load_and_preprocess_dataset(freq,
                                             prediction_length)

# %% DEFINE MODEL

model = define_my_model(num_of_variates,
                        multi_variate_train_dataset,
                        model_variant,
                        freq,
                        prediction_length)

# %% CREATE DATA LOADERS

train_dataloader = create_train_dataloader(
    config=model.config,
    freq=freq,
    data=multi_variate_train_dataset,
    batch_size=64,
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

train_model(num_of_epochs,
            model,
            train_dataloader,
            model_variant + "_Loss")

# %% INFERENCE

forecasts = forecasting(model,
                        test_dataloader)
see_metrics(forecasts,
            test_dataset,
            prediction_length,
            freq,
            "metrics.txt",
            model_variant + "_Metrics")

plot(forecasts, 0, 0, multi_variate_test_dataset,
     freq, prediction_length, model_variant + "_Temperature")
plot(forecasts, 0, 1, multi_variate_test_dataset,
     freq, prediction_length, model_variant + "_Relative_humidity")
plot(forecasts, 0, 2, multi_variate_test_dataset,
     freq, prediction_length, model_variant + "_Light")
plot(forecasts, 0, 3, multi_variate_test_dataset,
     freq, prediction_length, model_variant + "_Soil_Temperature")
plot(forecasts, 0, 4, multi_variate_test_dataset,
     freq, prediction_length, model_variant + "_Permittivity")
plot(forecasts, 0, 5, multi_variate_test_dataset,
     freq, prediction_length, model_variant + "_Electroconductivity")
plot(forecasts, 0, 6, multi_variate_test_dataset,
     freq, prediction_length, model_variant + "_Diameter")
