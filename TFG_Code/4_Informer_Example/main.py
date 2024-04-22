# %% LIBRARIES

from I_Load_Dataset import load_and_preprocess_dataset
from I_Define_Model import define_my_model
from I_Create_DataLoader import create_train_dataloader, create_backtest_dataloader
from I_Train_Model import train_model
from I_Evaluate_Model import forecasting, see_metrics, plot

# %% DEFINE VARIABLES

repo_name = "monash_tsf"
dataset_name = "traffic_hourly"
freq = "1H"
prediction_length = 48

# %% LOAD, SPLIT AND PREPROCESS DATASET

multi_variate_train_dataset, multi_variate_test_dataset, num_of_variates, test_dataset = load_and_preprocess_dataset(
    repo_name, dataset_name, freq, prediction_length)

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
see_metrics(forecasts, test_dataset, prediction_length, freq)
plot(forecasts, 0, 334, multi_variate_test_dataset, freq, prediction_length)
