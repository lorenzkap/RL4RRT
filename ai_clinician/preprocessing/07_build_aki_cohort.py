import pandas as pd
import numpy as np
import os
import argparse
import tqdm
from ai_clinician.preprocessing.columns import *
from ai_clinician.preprocessing.utils import load_csv

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

def outlier_stay_ids(df):
    """Returns a list of ICU stay IDs that should be removed from the dataset."""
    outliers = set()
    
    # check for patients with extreme UO = outliers = to be deleted (>40 litres of UO per 4h!!)
    outliers |= set(df[df[C_OUTPUT_STEP] > 12000][C_ICUSTAYID].unique())

    # some have bili = 999999
    outliers |= set(df[df[C_TOTAL_BILI] > 10000][C_ICUSTAYID].unique())

    # check for patients with extreme INTAKE = outliers = to be deleted (>10 litres of intake per 4h!!)
    outliers |= set(df[df[C_INPUT_STEP] > 10000][C_ICUSTAYID].unique())

    return outliers
    
def treatment_stopped_stay_ids(df):
    a = df[[C_BLOC, C_ICUSTAYID, C_MORTA_90, C_MAX_DOSE_VASO, C_SOFA]]
    grouped = a.groupby(C_ICUSTAYID)
    d = pd.merge(grouped.agg('max'),
                grouped.size().rename(C_NUM_BLOCS),
                how='left',
                left_index=True,
                right_index=True).drop(C_BLOC, axis=1)
    last_bloc = a.sort_values(C_BLOC, ascending=False).drop_duplicates(C_ICUSTAYID).rename({
        C_MAX_DOSE_VASO: C_LAST_VASO,
        C_SOFA: C_LAST_SOFA
    }, axis=1).drop(C_MORTA_90, axis=1)
    d = pd.merge(d,
                last_bloc,
                how='left',
                left_index=True,
                right_on=C_ICUSTAYID).set_index(C_ICUSTAYID, drop=True)
    
    stopped_treatment = d[
        (d[C_MORTA_90] == 1) & 
        (pd.isna(d[C_LAST_VASO]) | (d[C_LAST_VASO] < 0.01)) &
        (d[C_MAX_DOSE_VASO] > 0.3) &
        (d[C_LAST_SOFA] >= d[C_SOFA] / 2) &
        (d[C_NUM_BLOCS] < 20)
    ].index
    
    return stopped_treatment

def check_aki_per_time_step(patient_data):
    patient_data['AKI'] = False
    # Check if there are at least two entries for calculating the baseline
    if len(patient_data) > 1:
        baseline_creatinine = patient_data['Creatinine'].iloc[1]
    else:
        # If there are fewer than two entries, skip processing for this patient
        return patient_data
    
    # Initialize AKI status as False for all time points
    patient_data['AKI_cret'] = False
    patient_data['AKI_urin'] = False
    patient_data['AKI_RRT'] = False
    
    rrt_cols = [
        'Ultrafiltrate_Output',
        'Blood_Flow',
        'Hourly_Patient_Fluid_Removal',
        'Dialysate_Rate',
        'Hemodialysis_Output',
        'Citrate',
        'Prefilter_Replacement_Rate',
        'Postfilter_Replacement_Rate'
    ]
    
    # Create 'action' column for RRT
    rrt_actions = (~patient_data[rrt_cols].isna() & (patient_data[rrt_cols] != 0)).any(axis=1)
    patient_data['AKI_RRT'] = rrt_actions
    
    # Start checking from the first time step
    for i in range(1, len(patient_data)):
        current_index = patient_data.index[i]
        current_creatinine = patient_data['Creatinine'].iloc[i]
        
        # AKI criteria based on creatinine levels
        if (current_creatinine >= baseline_creatinine + 0.3) or \
           (current_creatinine >= 1.5 * baseline_creatinine) or \
           (current_creatinine > 4):
            patient_data.loc[current_index, 'AKI_cret'] = True

    # Check urinary output criteria
    total_hours_below_threshold = 0
    for i in range(len(patient_data)):
        current_index = patient_data.index[i]
        
        # Determine urinary output per hour based on weight
        weight = patient_data['Weight_kg'].iloc[i]
        if pd.isna(weight) or weight == 0:
            urinary_output_per_hour = patient_data['output_step'].iloc[i] / 4. / 80.
        else:
            urinary_output_per_hour = patient_data['output_step'].iloc[i] / 4. / weight

        if urinary_output_per_hour < 0.5:
            total_hours_below_threshold += 4
        else:
            total_hours_below_threshold = 0  # reset if not consistently below threshold

        if total_hours_below_threshold >= 6:
            patient_data.loc[current_index, 'AKI_urin'] = True

    # Combine criteria to set overall AKI status
    patient_data['AKI'] = patient_data['AKI_cret'] | patient_data['AKI_urin'] | patient_data['AKI_RRT']
    patient_data.drop(columns=['AKI_cret', 'AKI_urin', 'AKI_RRT'], inplace=True)
    
    return patient_data

def find_first_aki_and_trim(patient_data, window_before_hours=0, window_after_hours=240, time_interval_hours = 4):
    # Find the first occurrence of AKI
    aki_indices = patient_data.index[patient_data['AKI']].tolist()
    window_before_steps = window_before_hours//time_interval_hours
    window_after_steps = window_after_hours//time_interval_hours
    # If there's an AKI occurrence, keep all rows after the first AKI
    if aki_indices:
        first_aki_index = aki_indices[0]
        first_index = max(0, first_aki_index-window_before_steps)
        last_index = min(aki_indices[-1],first_aki_index + window_after_steps )
        return patient_data.loc[first_index:last_index].reset_index(drop=True)
    else:
        return pd.DataFrame()  # Return empty dataframe if no AKI


def died_in_icu_stay_ids(df):
    # exclude patients who died in ICU during data collection period
    died_in_icu = df[
        (df[C_DIED_WITHIN_48H_OF_OUT_TIME] == 1) &
        (df[C_DELAY_END_OF_RECORD_AND_DISCHARGE_OR_DEATH] < 24)
    ][C_ICUSTAYID].unique()
    return died_in_icu

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Filters the state/action '
        'dataframe to generate a aki cohort.'))
    parser.add_argument('input', type=str,
                        help='Path to patient states and actions CSV file')
    parser.add_argument('qstime', type=str,
                        help='Path to qstime.csv file')
    parser.add_argument('output', type=str,
                        help='Directory in which to write output')
    parser.add_argument('--data', dest='data_dir', type=str, default=None,
                        help='Directory in which raw and preprocessed data is stored (default is ../data/ directory)')
    parser.add_argument('--no-outlier-exclusion', dest='outlier_exclusion', default=True, action='store_false',
                        help="Don't exclude outliers by lab values")
    
    args = parser.parse_args()
    data_dir = args.data_dir or os.path.join(PARENT_DIR, 'data')
    out_dir = args.output
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    df = load_csv(args.input)
    qstime = load_csv(args.qstime)
    qstime = qstime.set_index(C_ICUSTAYID, drop=True)
    
    print("Before filtering:", len(set(df[C_ICUSTAYID])), "ICU stays")  # count before

    if args.outlier_exclusion:
        outliers = outlier_stay_ids(df)
        print(len(outliers), "outliers to remove")
        df = df[~df[C_ICUSTAYID].isin(outliers)]
        
    # stopped_treatment = treatment_stopped_stay_ids(df)
    # print(len(stopped_treatment), "stays to remove because treatment was stopped and patient died")
    # df = df[~df[C_ICUSTAYID].isin(stopped_treatment)]

    died_in_icu = died_in_icu_stay_ids(df)    
    print(len(died_in_icu), "patients to remove because died in ICU during data collection")
    df = df[~df[C_ICUSTAYID].isin(died_in_icu)]

    print("After filtering:", len(set(df[C_ICUSTAYID])), "ICU stays")  # count after

    ############################################################
    # Filter rows where 'Ultrafiltrate_Output' has a valid entry
    ultrafiltrate_entries = df[df['Ultrafiltrate_Output'].notna()]

    # Filter rows where any of the specified columns have valid entries
    filtered_entries = df[
        df['Ultrafiltrate_Output'].notna() | 
        df['Blood_Flow'].notna() | 
        df['Hourly_Patient_Fluid_Removal'].notna() | 
        df['Dialysate_Rate'].notna()
    ]

    # Count the distinct 'icustayid's with at least one non-null 'Ultrafiltrate_Output' entry
    distinct_icustayid_count = ultrafiltrate_entries['icustayid'].nunique()

    print("Number of distinct icustayid with at least one 'Ultrafiltrate_Output' entry:", distinct_icustayid_count)

    # Count the distinct 'icustayid's with at least one valid entry in any of the specified columns
    distinct_icustayid_counts = filtered_entries['icustayid'].nunique()

    print("Number of distinct icustayid with at least one entry in specified columns:", distinct_icustayid_counts)
    ############################################################

    aki_df = df.groupby('icustayid').apply(check_aki_per_time_step).reset_index(drop=True)
    aki_patients = aki_df.groupby('icustayid').apply(find_first_aki_and_trim).reset_index(drop=True)

    ############################################################
    # Filter rows where 'Ultrafiltrate_Output' has a valid entry
    ultrafiltrate_entries = aki_patients[aki_patients['Ultrafiltrate_Output'].notna()]

    # Filter rows where any of the specified columns have valid entries
    filtered_entries = df[
        df['Ultrafiltrate_Output'].notna() | 
        df['Blood_Flow'].notna() | 
        df['Hourly_Patient_Fluid_Removal'].notna() | 
        df['Dialysate_Rate'].notna()
    ]

    # Count the distinct 'icustayid's with at least one non-null 'Ultrafiltrate_Output' entry
    distinct_icustayid_count = ultrafiltrate_entries['icustayid'].nunique()

    print("Number of distinct icustayid with at least one 'Ultrafiltrate_Output' entry:", distinct_icustayid_count)

    # Count the distinct 'icustayid's with at least one valid entry in any of the specified columns
    distinct_icustayid_counts = filtered_entries['icustayid'].nunique()

    print("Number of distinct icustayid with at least one entry in specified columns:", distinct_icustayid_counts)

    ############################################################

    # aki_patients = df.groupby('icustayid').apply(check_aki_per_time_step).reset_index(drop=True)

    aki_patients = aki_patients.groupby('icustayid').agg({
        C_MORTA_90: 'first',
        C_SOFA: 'max',
        C_SIRS: 'max', 
        'timestep': 'first',
    }).rename({
        C_SOFA: C_MAX_SOFA,
        C_SIRS: C_MAX_SIRS,
    }, axis=1)

    # print("aki_patients BEFORE:",aki_patients.head())
    aki_patients.rename(columns={'timestep':'onset_time'}, inplace=True)

    # aki_patients = aki_patients.merge(qstime[C_ONSET_TIME], how='left', on="icustayid").reset_index(drop=True)
    print("Write")
    # print("aki_patients AFTER:",aki_patients.head())

    aki_patients = aki_patients.reset_index()
    # aki_patients[["morta_90","max_SOFA","max_SIRS","onset_time"]].to_csv(os.path.join(out_dir, "aki_cohort.csv"))
    aki_patients[['icustayid', 'morta_90', 'max_SOFA', 'max_SIRS', 'onset_time']].to_csv(os.path.join(out_dir, "aki_cohort.csv"),  index=False)

    # Ensure icustayid is a column by resetting index if needed
    # aki_patients = aki_patients.reset_index()

    # Load the onset.csv file
    onset_path = os.path.join(data_dir, 'intermediates', 'onset.csv')
    onset_df = load_csv(onset_path)

    # Filter onset_df to include only rows with icustayids present in aki_patients
    updated_onset_df = onset_df[onset_df['icustayid'].isin(aki_patients['icustayid'])]

    # Merge to update onset_time from aki_patients
    updated_onset_df = updated_onset_df.merge(
        aki_patients[['icustayid', 'onset_time']], 
        on='icustayid', 
        how='inner',  # Keeps only matching icustayids
        suffixes=('_old', '')  # Avoids column name conflict
    )

    # Drop the old onset_time column and rename the new onset_time column if necessary
    updated_onset_df.drop(columns=['onset_time_old'], inplace=True)

    # Save the updated DataFrame, replacing the original onset.csv
    updated_onset_df.to_csv(onset_path, index=False)