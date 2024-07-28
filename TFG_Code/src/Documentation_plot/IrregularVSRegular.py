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
file_name = 'Clay_2_new.csv'
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

# Remove trend from our data
window_value = 100
for col in ['Diameter']:
    data.loc[:, col] = data[col] - data[col].rolling(window=window_value).mean()

# Remove NaN values from removing the trend
data = data.dropna()

# Plot nan index with NaN values
plt.figure(figsize=(10, 2))
plt.plot(index_nan, np.ones_like(index_nan), 'ro', markersize=2)
plt.title(f'Nan index distribution. \n Number of eliminated measurements: {len(index_nan)}')
plt.xlabel('Index')
plt.ylabel('Frequency')
plt.yticks([])
plt.grid(True)
plt.xlim(0, len(df))

# Save the plot to a PDF file
plt.savefig('nan_index_distribution.pdf', format='pdf')
plt.show()

# Smooth the signal

for col in data.columns[1:]:
    data[col] = savgol_filter(data[col], 11, 2)

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

# %% Plotting Original vs. Resampled Data

# Initial parameters
start_date = "2022-01-01"  # start date in "YYYY-MM-DD" format
frequency = '7min52s'  # frequency of observations ('D' for daily, 'M' for monthly, etc.)

# Define a list of colors to use
colors = plt.cm.tab10.colors  # Use a colormap with up to 10 distinct colors

# Create subplots for the comparison
for var, color in zip(data.columns, colors):
    # Plot full data with increased figure size
    figure, axes = plt.subplots(nrows=2, ncols=1, figsize=(24, 8), constrained_layout=True)

    # Plot Original Data
    axes[0].plot(data.index[-9525:-4395], data[var][-9525:-4395],
                 color=color, label="Original", alpha=0.7)
    title = f'{var} (Original)'
    axes[0].set_title(title.replace("_", " "), fontsize=14)
    y_label = f'{var} ({units.get(var, "")})'
    axes[0].set_ylabel(y_label.replace("_", " "), fontsize=12)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[0].grid(True)

    # Plot Resampled Data
    axes[1].plot(resampled_data.index[-9525:-4395], resampled_data[var][-9525:-4395],
                 color=color, label="Resampled", alpha=0.7)
    title = f'{var} (Resampled)'
    axes[1].set_title(title.replace("_", " "), fontsize=14)
    y_label = f'{var} ({units.get(var, "")})'
    axes[1].set_ylabel(y_label.replace("_", " "), fontsize=12)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[1].grid(True)

    # Set common X-label
    axes[1].set_xlabel('Date', fontsize=14)

    # Rotate X-ticks for both plots
    plt.setp(axes[0].get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.setp(axes[1].get_xticklabels(), rotation=30, horizontalalignment='right')

    # Save the plot to a PDF file
    plt.savefig(f'comparison_{var}.pdf', format='pdf')
    plt.show()
