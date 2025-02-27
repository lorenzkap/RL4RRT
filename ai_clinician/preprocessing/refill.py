import pandas as pd
import os

# Define the path to your dataset and output directory
dataset_path = "/home/mahdi/RRT_mimic_iv/data/mimiciv/mimic_dataset.csv"
output_dir = "/home/mahdi/RRT_mimic_iv/data/mimiciv_refilled"

# Load the dataset into a pandas DataFrame
df = pd.read_csv(dataset_path)

# Display the number of NaN values before replacement
print("Number of NaN values before replacement:", df['died_within_48h_of_out_time'].isna().sum())

# Replace NaN in 'died_within_48h_of_out_time' with 'False' where 'morta_90' is 0
df.loc[df['morta_90'] == 0, 'died_within_48h_of_out_time'] = df.loc[df['morta_90'] == 0, 'died_within_48h_of_out_time'].fillna(False)

# Display the number of NaN values after replacement
print("Number of NaN values after replacement:", df['died_within_48h_of_out_time'].isna().sum())

# Optionally, save the modified DataFrame back to a new CSV file in the output directory
output_filename = "mimic_dataset.csv"
output_path = os.path.join(output_dir, output_filename)
df.to_csv(output_path, index=False)

print(f"Processed data saved to: {output_path}")
