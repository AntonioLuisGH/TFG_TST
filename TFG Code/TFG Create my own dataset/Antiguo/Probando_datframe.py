# %% LIBRARIES

from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from functools import partial
import pandas as pd
from functools import lru_cache
from datasets import load_dataset
import optional

# %% LOAD AND PREPROCESS
freq = '1H'
prediction_length = 48
train_dataset = load_dataset("monash_tsf", "traffic_hourly", split='train')
# train_dataset = dataset.to_pandas()

test_dataset = load_dataset("monash_tsf", "traffic_hourly", split='test')
# test_dataset = dataset.to_pandas()

# Update start to pd.Period


@lru_cache(10_000)
def convert_to_pandas_period(date, freq):
    return pd.Period(date, freq)


def transform_start_field(batch, freq):
    batch["start"] = [convert_to_pandas_period(
        date, freq) for date in batch["start"]]
    return batch


train_dataset.set_transform(partial(transform_start_field, freq=freq))
test_dataset.set_transform(partial(transform_start_field, freq=freq))

# Convert the dataset into a multivariate time series
num_of_variates = len(train_dataset)

train_grouper = MultivariateGrouper(max_target_dim=num_of_variates)
test_grouper = MultivariateGrouper(
    max_target_dim=num_of_variates,
    # number of rolling test windows
    num_test_dates=len(test_dataset) // num_of_variates,
)

multi_variate_train_dataset = train_grouper(train_dataset)
multi_variate_test_dataset = test_grouper(test_dataset)
