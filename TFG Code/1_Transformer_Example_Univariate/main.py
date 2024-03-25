# %% LIBRARIES

from TST_Load_Dataset import load_and_preprocess_dataset
from TST_Define_Model import define_my_model
from TST_Create_DataLoader import create_train_dataloader, create_backtest_dataloader, create_test_dataloader

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
