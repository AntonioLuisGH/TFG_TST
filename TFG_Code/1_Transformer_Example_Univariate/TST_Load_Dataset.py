# %% LIBRARIES

from functools import partial
<<<<<<< HEAD
import pandas as pd
from functools import lru_cache
=======
import numpy as np
import pandas as pd
from functools import lru_cache
import matplotlib.pyplot as plt
>>>>>>> 5858acfe1f1758e3f28d9b766481bdd68d407f85
from datasets import load_dataset

# %% LOAD AND PREPROCESS


def load_and_preprocess_dataset(repo_name, dataset_name, freq, prediction_length):
    # Load dataset
    dataset = load_dataset(repo_name, dataset_name)

    # Check the prediction_length
    validation_example = dataset['validation'][0]
    train_example = dataset['train'][0]
    assert len(train_example["target"]) + prediction_length == len(
        validation_example["target"]
    )

    # Split the data
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

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

    return train_dataset, test_dataset
