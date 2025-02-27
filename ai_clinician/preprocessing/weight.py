import os
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm

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
    # Input dataset path
    input_dataset_path = r"../../data/mimiciv_3.1/mimic_dataset_cm.csv"
    
    # CE files pattern
    ce_files_pattern = r"../../data/raw_data/ce*.csv"
    
    # Read the main dataset
    print("Reading main dataset...")
    df = pd.read_csv(input_dataset_path)
    
    # Load all CE data into memory
    print("Loading CE files into memory...")
    ce_data = load_ce_files(ce_files_pattern)
    
    # Create a mapping of icustayid to weight in kg
    print("Creating icustayid to weight mapping...")
    icustay_weight_mapping = create_icustayid_weight_mapping(ce_data)
    
    # Map the weights to the main dataset using the precomputed mapping
    print("Mapping weights to main dataset...")
    df['Weight_kg'] = df['icustayid'].map(icustay_weight_mapping)
    
    # Output path for new dataset
    output_path = os.path.join(
        os.path.dirname(input_dataset_path), 
        'mimic_dataset_cm_kg.csv'
    )
    
    # Save the updated dataset
    print("Saving updated dataset...")
    df.to_csv(output_path, index=False)
    
    print(f"Updated dataset saved to {output_path}")
    
    # Optional: Print summary statistics
    print("\nWeight Column Summary:")
    print(df['Weight_kg'].describe())

if __name__ == "__main__":
    main()