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
import matplotlib.dates as mdates
import numpy as np

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
    # Directory name where the CSV file is located
    directory_name = 'Create_2024_dataset'
    # CSV file name
    file_name = 'Clay_2.csv'
    # Concatenate the current path with the directory name and file name
    file_path = os.path.join(current_path, directory_name, file_name)
    # Read the CSV file
    df = pd.read_csv(file_path, sep=";")

    # Relevant variable selection
    data = df[['Temperature', 'Relative_humidity', 'Light', 'Soil_temperature',
               'Permittivity', 'Electroconductivity', 'Diameter']]

    # %%
    # Data preprocessing

    # Remove NaN values from original data
    index_nan = data[data.isna().any(axis=1)].index
    data = data.dropna()

    # Remove trend from our data
    window_value = 100
    for col in ['Diameter']:
        data.loc[:, col] = data[col] - data[col].rolling(window=window_value).mean()

    # Remove NaN values from removing the trend
    data = data.dropna()

    # Plot nan index with Nan values
    plt.figure(figsize=(10, 2))
    plt.plot(index_nan, np.ones_like(index_nan), 'ro', markersize=2)
    plt.title(
        f'Nan index distribution. \n Number of eliminated mesaurements: {len(index_nan)}')
    plt.xlabel('Index')
    plt.ylabel('Frecuency')
    plt.yticks([])
    plt.grid(True)
    plt.xlim(0, len(df))
    plt.show()

    # Smooth the signal
    for col in data.columns:
        data[col] = savgol_filter(data[col], 11, 2)

    # %%
    # Taking frequency from the dates
    dates = df['Date']
    dates = dates.drop(index_nan)
    dates = pd.to_datetime(dates)

    # Calculate the time intervals between consecutive dates
    intervals = dates.diff()

    # Calculate the average of the intervals
    sampling_period = intervals.mean()
    print("\nAverage of the sampling period:\n", sampling_period)

    # %%
    # Data normalization
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    data[['Temperature', 'Relative_humidity', 'Light', 'Soil_temperature',
          'Permittivity', 'Electroconductivity', 'Diameter']] = scaled_data

    # %%
    # Data split
    data_test = data
    data_validation = data.iloc[:-prediction_length]
    data_train = data.iloc[:-2*prediction_length]

    # Create empty dictionaries to store the data
    dict_validation = {'start': [], 'target': [],
                       'feat_static_cat': [], 'feat_dynamic_real': [], 'item_id': []}

    dict_test = {'start': [], 'target': [],
                 'feat_static_cat': [], 'feat_dynamic_real': [], 'item_id': []}

    dict_train = {'start': [], 'target': [],
                  'feat_static_cat': [], 'feat_dynamic_real': [], 'item_id': []}

    # Populate the dictionaries with the corresponding data
    for i in range(1, 8):

        dict_validation['target'].append(data_validation.iloc[:, i-1].values.astype('float32'))
        dict_test['target'].append(data_test.iloc[:, i-1].values.astype('float32'))
        dict_train['target'].append(data_train.iloc[:, i-1].values.astype('float32'))

        for d in [dict_validation, dict_test, dict_train]:
            d['start'].append(pd.Timestamp('2022-01-01 13:14:26'))
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
    # Initial parameters
    start_date = "2022-01-01"  # start date in "YYYY-MM-DD" format
    frequency = '7min52s'  # frequency of observations ('D' for daily, 'M' for monthly, etc.)

    # Generate dates for the x-axis

    def generate_dates(start, num_periods, freq):
        return pd.date_range(start=start, periods=num_periods, freq=freq)

    for var in range(7):
        # Generate dates for train and test data
        num_periods_train = len(train_dataset[var]["target"])
        num_periods_test = len(test_dataset[var]["target"])

        train_dates = generate_dates(start_date, num_periods_train, frequency)
        test_dates = generate_dates(start_date, num_periods_test, frequency)

        # Plot full data
        figure, axes = plt.subplots(figsize=(20, 6.4))
        plt.setp(axes.get_xticklabels(), rotation=30, horizontalalignment='right')

        # Plot train data
        axes.plot(train_dates, train_dataset[var]["target"], color="blue", label="Train")
        # Plot test data
        axes.plot(test_dates[-2*prediction_length:], test_dataset[var]
                  ["target"][-2*prediction_length:], color="red", label="Test")

        axes.set_title(data.columns[var].replace('_', ' '))  # Set title for the plot
        axes.legend()  # Show legend
        axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Date format
        figure.autofmt_xdate()  # Format dates

        # Plot last segment (zoom)
        figure, axes = plt.subplots()
        plt.setp(axes.get_xticklabels(), rotation=30, horizontalalignment='right')

        # Plot train data (last 3*prediction_length)
        axes.plot(train_dates[-3*prediction_length:], train_dataset[var]
                  ["target"][-3*prediction_length:], color="blue", label="Train (zoom)")
        # Plot test data
        axes.plot(test_dates[-2*prediction_length:], test_dataset[var]["target"]
                  [-2*prediction_length:], color="red", label="Test (zoom)")

        axes.set_title(data.columns[var].replace('_', ' '))  # Set title for the plot
        axes.legend()  # Show legend
        axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Date format
        figure.autofmt_xdate()  # Format dates

    plt.show()

    return dataset_validation, dataset_test, dataset_train
