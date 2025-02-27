import os
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm

def convert_inches_to_cm(height_inches):
    """Convert height from inches to centimeters."""
    return height_inches * 2.54 if pd.notnull(height_inches) else np.nan

def load_ce_files(ce_files_pattern):
    """
    Load all CE files matching the pattern into a single DataFrame.
    
    Parameters:
    -----------
    ce_files_pattern : str
        Path pattern for CE files
    
    Returns:
    --------
    pd.DataFrame
        A DataFrame containing all CE data
    """
    ce_files = sorted(glob.glob(ce_files_pattern))
    dataframes = []
    for filename in tqdm(ce_files, desc="Loading CE Files"):
        try:
            dataframes.append(pd.read_csv(filename))
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
    return pd.concat(dataframes, ignore_index=True)

def create_icustayid_height_mapping(ce_data):
    """
    Create a mapping of icustayid to height in cm using preloaded CE data.
    
    Parameters:
    -----------
    ce_data : pd.DataFrame
        Preloaded CE data
    
    Returns:
    --------
    dict
        A dictionary mapping icustayid to height in cm
    """
    height_mapping = {}

    # First, find heights in cm (itemid 226730)
    heights_cm = ce_data[ce_data['itemid'] == 226730].dropna(subset=['valuenum'])
    heights_cm = heights_cm[['icustayid', 'valuenum']].drop_duplicates(subset=['icustayid'])
    height_mapping.update(heights_cm.set_index('icustayid')['valuenum'].to_dict())

    # Next, find heights in inches (itemid 226707), and convert to cm if not already found
    heights_inches = ce_data[ce_data['itemid'] == 226707].dropna(subset=['valuenum'])
    heights_inches = heights_inches[['icustayid', 'valuenum']].drop_duplicates(subset=['icustayid'])

    for icustayid, height_in in heights_inches.itertuples(index=False):
        if icustayid not in height_mapping:  # Only update if not already in mapping
            height_mapping[icustayid] = convert_inches_to_cm(height_in)
    
    return height_mapping

def main():
    # Input dataset path
    input_dataset_path = r"../../data/mimiciv_3.1/mimic_dataset.csv"
    
    # CE files pattern
    ce_files_pattern = r"../../data/raw_data/ce*.csv"
    
    # Read the main dataset
    print("Reading main dataset...")
    df = pd.read_csv(input_dataset_path)
    
    # Load all CE data into memory
    print("Loading CE files into memory...")
    ce_data = load_ce_files(ce_files_pattern)
    
    # Create a mapping of icustayid to height in cm
    print("Creating icustayid to height mapping...")
    icustay_height_mapping = create_icustayid_height_mapping(ce_data)
    
    # Map the heights to the main dataset using the precomputed mapping
    print("Mapping heights to main dataset...")
    df['Height_cm'] = df['icustayid'].map(icustay_height_mapping)
    
    # Output path for new dataset
    output_path = os.path.join(
        os.path.dirname(input_dataset_path), 
        'mimic_dataset_cm.csv'
    )
    
    # Save the updated dataset
    print("Saving updated dataset...")
    df.to_csv(output_path, index=False)
    
    print(f"Updated dataset saved to {output_path}")
    
    # Optional: Print summary statistics
    print("\nHeight Column Summary:")
    print(df['Height_cm'].describe())

if __name__ == "__main__":
    main()
