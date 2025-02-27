import os
import sys
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm

def convert_inches_to_cm(height_inches):
    """Convert height from inches to centimeters."""
    return height_inches * 2.54 if pd.notnull(height_inches) else np.nan

def convert_lbs_to_kg(weight_lbs):
    """Convert weight from pounds to kilograms."""
    return weight_lbs * 0.453592 if pd.notnull(weight_lbs) else np.nan

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

def create_icustayid_weight_mapping(ce_data):
    """
    Create a mapping of icustayid to weight in kg using preloaded CE data.
    
    Parameters:
    -----------
    ce_data : pd.DataFrame
        Preloaded CE data
    
    Returns:
    --------
    dict
        A dictionary mapping icustayid to weight in kg
    """
    weight_mapping = {}

    # First, find weights in kg (itemid 226512)
    weights_kg = ce_data[ce_data['itemid'] == 226512].dropna(subset=['valuenum'])
    weights_kg = weights_kg[['icustayid', 'valuenum']].drop_duplicates(subset=['icustayid'])
    weight_mapping.update(weights_kg.set_index('icustayid')['valuenum'].to_dict())

    # Next, find weights in lbs (itemid 226531), and convert to kg if not already found
    weights_lbs = ce_data[ce_data['itemid'] == 226531].dropna(subset=['valuenum'])
    weights_lbs = weights_lbs[['icustayid', 'valuenum']].drop_duplicates(subset=['icustayid'])

    for icustayid, weight_lbs in weights_lbs.itertuples(index=False):
        if icustayid not in weight_mapping:  # Only update if not already in mapping
            weight_mapping[icustayid] = convert_lbs_to_kg(weight_lbs)
    
    return weight_mapping

def main():
    # Check if input and output dataset paths are provided as arguments
    if len(sys.argv) < 3:
        print("Usage: python 08_height_weight.py <input_dataset_path> <output_dataset_path>")
        sys.exit(1)
    
    # Read command line arguments
    input_dataset_path = sys.argv[1]
    output_dataset_path = sys.argv[2]
    
    # CE files pattern remains unchanged
    ce_files_pattern = r"data/raw_data/ce*.csv"
    
    # Read the main dataset
    print("Reading main dataset...")
    df = pd.read_csv(input_dataset_path)
    
    # Load all CE data into memory
    print("Loading CE files into memory...")
    ce_data = load_ce_files(ce_files_pattern)
    
    # Create mappings of icustayid to height in cm and weight in kg
    print("Creating icustayid to height mapping...")
    icustay_height_mapping = create_icustayid_height_mapping(ce_data)
    
    print("Creating icustayid to weight mapping...")
    icustay_weight_mapping = create_icustayid_weight_mapping(ce_data)
    
    # Map the heights and weights to the main dataset using the precomputed mappings
    print("Mapping heights to main dataset...")
    df['Height_cm'] = df['icustayid'].map(icustay_height_mapping)
    
    print("Mapping weights to main dataset...")
    df['Weight_kg'] = df['icustayid'].map(icustay_weight_mapping)
    
    # Save the updated dataset to the provided output path
    print("Saving updated dataset...")
    df.to_csv(output_dataset_path, index=False)
    
    print(f"Updated dataset saved to {output_dataset_path}")
    
    # Optional: Print summary statistics
    print("\nHeight Column Summary:")
    print(df['Height_cm'].describe())
    
    print("\nWeight Column Summary:")
    print(df['Weight_kg'].describe())

if __name__ == "__main__":
    main()
