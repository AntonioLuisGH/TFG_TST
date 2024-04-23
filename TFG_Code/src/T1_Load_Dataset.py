# %% LIBRARIES

from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from functools import partial
import pandas as pd
from functools import lru_cache
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter
from datasets import Dataset
from sklearn.preprocessing import MinMaxScaler

# %% LOAD AND PREPROCESS


def load_and_preprocess_dataset(freq, prediction_length):

    validationtest_dataset, test_dataset, train_dataset = load_my_own_dataset(
        prediction_length)

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


# %% LOAD DATASET FROM CSV


def load_my_own_dataset(prediction_length):

    # Get the absolute path of the current directory
    current_path = os.path.dirname(os.path.abspath(__file__))
    # CSV file name
    file_name = 'raw_data.csv'
    # Concatenate the current path with the CSV file name
    file_path = os.path.join(current_path, file_name)
    # Read the CSV file
    df = pd.read_csv(file_path, sep=";")

    # Relevant variables selection
    data = df[['luminosidad', 'temperatura',
               'humedad_rel', 'temp_suelo', 'electrocond', 'var_diam']]

    # Replace wrong measurements
    def replace_incorrect_values(serie, lim):
        for i in range(1, len(serie) - 1):
            if serie[i] < lim:
                serie[i] = serie[i - 1]
        return serie

    data['var_diam'] = replace_incorrect_values(data['var_diam'], 100)
    data['temp_suelo'] = replace_incorrect_values(data['var_diam'], 0.6)

    # %%
    # Data preprocessing

    # Remove trend from our data
    for col in data.columns:
        data[col] = data[col] - data[col].rolling(window=100).mean()

    # Remove NaN values
    data = data.dropna()

    # Smooth the signal
    for col in data.columns:
        data[col] = savgol_filter(data[col], 11, 2)

    # %%
    # Data normalization
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    data[['luminosidad', 'temperatura',
          'humedad_rel', 'temp_suelo', 'electrocond', 'var_diam']] = scaled_data

    # %%
    # Data split
    data_validation = data
    data_test = data.iloc[:-prediction_length]
    data_train = data.iloc[:-2*prediction_length]

    # Create empty dictionaries to store the data
    dict_validation = {'start': [], 'target': [],
                       'feat_static_cat': [], 'feat_dynamic_real': [], 'item_id': []}

    dict_test = {'start': [], 'target': [],
                 'feat_static_cat': [], 'feat_dynamic_real': [], 'item_id': []}

    dict_train = {'start': [], 'target': [],
                  'feat_static_cat': [], 'feat_dynamic_real': [], 'item_id': []}

    # Populate the dictionaries with the corresponding data
    for i in range(1, 7):

        dict_validation['target'].append(data_validation.iloc[:, i-1].values.astype('float32'))
        dict_test['target'].append(data_test.iloc[:, i-1].values.astype('float32'))
        dict_train['target'].append(data_train.iloc[:, i-1].values.astype('float32'))

        for d in [dict_validation, dict_test, dict_train]:
            d['start'].append(pd.Timestamp('2020-10-01 01:06:48'))
            d['feat_static_cat'].append(i)
            d['feat_dynamic_real'].append(None)
            d['item_id'].append(f'T{i}')

    # Convert the dictionaries into pandas DataFrames
    dataframe_validation = pd.DataFrame(dict_validation)
    dataframe_test = pd.DataFrame(dict_test)
    dataframe_train = pd.DataFrame(dict_train)

    # Convert DataFrames into GluonTS Datasets
    dataset_validation = Dataset.from_pandas(dataframe_validation)
    dataset_test = Dataset.from_pandas(dataframe_test)
    dataset_train = Dataset.from_pandas(dataframe_train)

    # %% SHOWING DATA

    test_dataset = dataset_test
    train_dataset = dataset_train

    # Show data
    for var in range(6):

        figure, axes = plt.subplots(figsize=(12, 6))
        ejex = list(range(len(test_dataset[var]["target"])))
        plt.setp(axes.get_xticklabels(), rotation=30, horizontalalignment='right')
        axes.plot(ejex[-prediction_length:], test_dataset[var]
                  ["target"][-prediction_length:], color="red")
        axes.plot(train_dataset[var]["target"], color="blue")
        axes.set_title(data.columns[var])  # Set title for main data plo

        # ZOOM
        # Show last values
        figure, axes = plt.subplots()
        ejex = list(range(len(test_dataset[var]["target"])))
        axes.plot(ejex[-prediction_length:], test_dataset[var]
                  ["target"][-prediction_length:], color="red")
        axes.plot(ejex[-3*prediction_length:-prediction_length], train_dataset[var]
                  ["target"][-2*prediction_length:], color="blue")
        axes.set_title(data.columns[var])

    return dataset_validation, dataset_test, dataset_train
