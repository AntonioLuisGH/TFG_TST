# %% Creation of my file ".Dataset"

import os
import pandas as pd
from scipy.signal import savgol_filter
from datasets import Dataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# prediction lenth
prediction_length = 500

# Get the absolute path of the current directory
current_path = os.path.dirname(os.path.abspath(__file__))
# CSV file name
file_name = 'Clay_2.csv'
# Concatenate the current path with the CSV file name
file_path = os.path.join(current_path, file_name)
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
plt.title(f'Nan index distribution. \n Number of eliminated mesaurements: {len(index_nan)}')
plt.xlabel('Index')
plt.ylabel('Frecuency')
plt.yticks([])
plt.grid(True)
plt.xlim(0, len(df))
plt.show()

# # Smooth the signal
# for col in data.columns:
#     data[col] = savgol_filter(data[col], 11, 2)

# %%
# Taking frequency from the dates
dates = df['Date']
dates = dates.drop(index_nan)
dates = pd.to_datetime(dates)

# Calculate the time intervals between consecutive dates
intervals = dates.diff()

# Calculate the average of the intervals
mean_intervals = intervals.mean()
print("Average of the time intervals:", mean_intervals)

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
        d['start'].append(pd.Timestamp('2020-01-01 00:03:47'))
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
start_date = "2020-01-01"  # start date in "YYYY-MM-DD" format
frequency = '7min50s'  # frequency of observations ('D' for daily, 'M' for monthly, etc.)

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
    figure, axes = plt.subplots(figsize=(20, 6))
    plt.setp(axes.get_xticklabels(), rotation=30, horizontalalignment='right')

    # Plot train data
    axes.plot(train_dates, train_dataset[var]["target"], color="blue", label="Train")
    # Plot test data
    axes.plot(test_dates[-2*prediction_length:], test_dataset[var]
              ["target"][-2*prediction_length:], color="red", label="Test")

    axes.set_title(data.columns[var])  # Set title for the plot
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

    axes.set_title(data.columns[var])  # Set title for the plot
    axes.legend()  # Show legend
    axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Date format
    figure.autofmt_xdate()  # Format dates

plt.show()
