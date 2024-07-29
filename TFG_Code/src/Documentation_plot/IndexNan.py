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
plt.title(f'Measurements with NaN values distribution')
plt.xlabel('Index')
plt.yticks([])
plt.grid(True)
plt.xlim(0, len(df))

# Save the plot to a PDF file
plt.savefig('nan_index_distribution.pdf', format='pdf')
plt.show()
