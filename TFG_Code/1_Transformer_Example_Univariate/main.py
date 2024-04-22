# %% LIBRARIES

from TST_Load_Dataset import load_and_preprocess_dataset
from TST_Define_Model import define_my_model
from TST_Create_DataLoader import create_train_dataloader, create_backtest_dataloader
from TST_Train_Model import train_model
from TST_Evaluate_Model import forecasting, see_metrics, plot

# %% DEFINE VARIABLES

repo_name = "monash_tsf"
dataset_name = "tourism_monthly"
freq = "1M"
prediction_length = 24

# %% LOAD, SPLIT AND PREPROCESS DATASET

train_dataset, test_dataset = load_and_preprocess_dataset(
    repo_name, dataset_name, freq, prediction_length)

# %% DEFINE MODEL

model = define_my_model(train_dataset, freq, prediction_length)

# %% CREATE DATA LOADERS

train_dataloader = create_train_dataloader(
    config=model.config,
    freq=freq,
    data=train_dataset,
    batch_size=256,
    num_batches_per_epoch=100,
)

test_dataloader = create_backtest_dataloader(
    config=model.config,
    freq=freq,
    data=test_dataset,
    batch_size=64,
)

batch = next(iter(train_dataloader))

# %% FORWARD PASS

# Let's perform a single forward pass with the batch we just created:
outputs = model(
    past_values=batch["past_values"],
    past_time_features=batch["past_time_features"],
    past_observed_mask=batch["past_observed_mask"],
    static_categorical_features=batch["static_categorical_features"]
    if model.config.num_static_categorical_features > 0
    else None,
    static_real_features=batch["static_real_features"]
    if model.config.num_static_real_features > 0
    else None,
    future_values=batch["future_values"],
    future_time_features=batch["future_time_features"],
    future_observed_mask=batch["future_observed_mask"],
    output_hidden_states=True,
)

# %% TRAIN MODEL

train_model(model, train_dataloader)


# %% INFERENCE

forecasts = forecasting(model, test_dataloader)
see_metrics(forecasts, test_dataset, prediction_length, freq, "metrics.txt")
see_metrics(forecasts, test_dataset, prediction_length, freq)
plot(forecasts, 334, test_dataset, prediction_length, freq)
