# %% LIBRARIES

from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from functools import partial
import pandas as pd
from functools import lru_cache
import matplotlib.pyplot as plt
from datasets import load_dataset
from Creating_my_dataset import load_my_own_dataset

# %% LOAD AND PREPROCESS


def load_and_preprocess_dataset(freq, prediction_length):

    validationtest_dataset, test_dataset, train_dataset = load_my_own_dataset()

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

    return multi_variate_train_dataset, multi_variate_test_dataset, num_of_variates, test_dataset
