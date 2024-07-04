import os
import pandas as pd
import re


def extract_number(folder_name):
    match = re.search(r'\d+', folder_name)
    return int(match.group()) if match else float('inf')


# Get the path to the main folder (the same where the script is located)
main_folder = os.getcwd()

# Initialize a list to store the data
data = []

# Iterate over all subfolders in the main folder and sort them
for folder_name in sorted(os.listdir(main_folder), key=extract_number):
    folder_path = os.path.join(main_folder, folder_name)
    if os.path.isdir(folder_path) and folder_name.startswith('T'):
        # Path to the plots subfolder within the current folder
        plots_folder = os.path.join(folder_path, 'plots')
        # Path to the metrics.txt file in the plots subfolder
        metrics_file = os.path.join(plots_folder, 'metrics.txt')
        if os.path.exists(metrics_file):
            # Read the metrics.txt file and extract the data
            with open(metrics_file, 'r') as file:
                lines = file.readlines()
                # Skip the first header line
                for line in lines[1:]:
                    # Split the line into components
                    components = line.split()
                    if len(components) == 3:
                        variable, mse, r_squared = components
                        # Add a row to the data
                        data.append([folder_name, variable, float(mse), float(r_squared)])

# Create a pandas DataFrame with the data
df = pd.DataFrame(data, columns=['Folder', 'Variable', 'MSE', 'R_squared'])

# Save the DataFrame to an Excel file
output_file = 'metrics_summary.xlsx'
df.to_excel(output_file, index=False)

print(f"Data exported to {output_file}")
