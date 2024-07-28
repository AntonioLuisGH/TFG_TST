# %% Creation of my file ".Dataset"

from scipy.signal import savgol_filter
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# Prediction length
prediction_length = 500

# Get the absolute path of the current directory
current_path = os.path.dirname(os.path.abspath("__file__"))  # Adjusted to run in any context
# CSV file name
file_name = 'Sand_1.csv'
# Concatenate the current path with the CSV file name
file_path = os.path.join(current_path, file_name)
# Read the CSV file
df = pd.read_csv(file_path, sep=";")

# Relevant variable selection
data = df[['Date', 'Temperature', 'Relative_humidity', 'Light', 'Soil_temperature',
           'Permittivity', 'Electroconductivity', 'Diameter']]

# Define units for each variable
units = {
    'Temperature': '°C',
    'Relative_humidity': '%',
    'Light': 'Lux',
    'Soil_temperature': '°C',
    'Permittivity': 'F/m',
    'Electroconductivity': 'S/m',
    'Diameter': 'mm'
}


# %% Data preprocessing

# Remove NaN values from original data
index_nan = data[data.isna().any(axis=1)].index
data = data.dropna()

# %% Convert the 'Date' column to datetime
data['Date'] = pd.to_datetime(df['Date'].drop(index_nan))

# Calculate the time intervals between consecutive dates
intervals = data['Date'].diff()

# Calculate the average of the intervals
mean_intervals = intervals.mean()
print("Average of the time intervals:", mean_intervals)

# %% Resample the data to a regular frequency
# Set the Date column as the index
data.set_index('Date', inplace=True)

# Define the resampling frequency
resampling_frequency = '7min52s'

# Resample the data
resampled_data = data.resample(resampling_frequency).mean().interpolate()


# %% Plotting Original vs. TrendData Data

# Initial parameters
start_date = "2022-01-01"  # start date in "YYYY-MM-DD" format
frequency = '7min52s'  # frequency of observations ('D' for daily, 'M' for monthly, etc.)

# Define a list of colors to use
colors = plt.cm.tab10.colors  # Use a colormap with up to 10 distinct colors

# Create subplots for the comparison
for var, color in zip(['Diameter'], colors):
    # Create a single figure
    figure, ax = plt.subplots(figsize=(24, 4), constrained_layout=True)

    # Plot Original Data
    ax.plot(data.index, data[var],
            color=color, label="Original", alpha=0.7)
    ax.set_title('Diameter Values from the First Sand Dataset', fontsize=14)
    ax.set_ylabel(f'{var} ({units.get(var, "")})', fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.grid(True)

    # Rotate X-ticks
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')

    # Save the plot to a PDF file
    plt.savefig('Diameter_Sand_1.pdf', format='pdf')
    plt.show()


# %% REPEAT #####################################################
    # Prediction length
    prediction_length = 500

    # Get the absolute path of the current directory
    current_path = os.path.dirname(os.path.abspath(
        "__file__"))  # Adjusted to run in any context
    # CSV file name
    file_name = 'Sand_2.csv'
    # Concatenate the current path with the CSV file name
    file_path = os.path.join(current_path, file_name)
    # Read the CSV file
    df = pd.read_csv(file_path, sep=";")

    # Relevant variable selection
    data = df[['Date', 'Temperature', 'Relative_humidity', 'Light', 'Soil_temperature',
               'Permittivity', 'Electroconductivity', 'Diameter']]

    # Define units for each variable
    units = {
        'Temperature': '°C',
        'Relative_humidity': '%',
        'Light': 'Lux',
        'Soil_temperature': '°C',
        'Permittivity': 'F/m',
        'Electroconductivity': 'S/m',
        'Diameter': 'mm'
    }

    # %% Data preprocessing

    # Remove NaN values from original data
    index_nan = data[data.isna().any(axis=1)].index
    data = data.dropna()

    # %% Convert the 'Date' column to datetime
    data['Date'] = pd.to_datetime(df['Date'].drop(index_nan))

    # Calculate the time intervals between consecutive dates
    intervals = data['Date'].diff()

    # Calculate the average of the intervals
    mean_intervals = intervals.mean()
    print("Average of the time intervals:", mean_intervals)

    # %% Resample the data to a regular frequency
    # Set the Date column as the index
    data.set_index('Date', inplace=True)

    # Define the resampling frequency
    resampling_frequency = '7min52s'

    # Resample the data
    resampled_data = data.resample(resampling_frequency).mean().interpolate()

    # %% Plotting Original vs. TrendData Data

    # Initial parameters
    start_date = "2022-01-01"  # start date in "YYYY-MM-DD" format
    frequency = '7min52s'  # frequency of observations ('D' for daily, 'M' for monthly, etc.)

    # Define a list of colors to use
    colors = plt.cm.tab10.colors  # Use a colormap with up to 10 distinct colors

    # Create subplots for the comparison
    for var, color in zip(['Light'], colors):
        # Create a single figure
        figure, ax = plt.subplots(figsize=(24, 4), constrained_layout=True)

        # Plot Original Data
        ax.plot(data.index, data[var],
                color=color, label="Original", alpha=0.7)
        ax.set_title('Light Values from the Second Sand Dataset', fontsize=14)
        ax.set_ylabel(f'{var} ({units.get(var, "")})', fontsize=12)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.grid(True)

        # Rotate X-ticks
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')

        # Save the plot to a PDF file
        plt.savefig('Light_Sand_1.pdf', format='pdf')
        plt.show()
