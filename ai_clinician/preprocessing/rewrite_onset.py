import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm
from ai_clinician.preprocessing.columns import C_ICUSTAYID
from ai_clinician.preprocessing.utils import load_csv, load_intermediate_or_raw_csv
from ai_clinician.preprocessing.derived_features import calculate_onset

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Calculates the presumed time '
        'of sepsis onset for each patient, and generates a sepsis_onset.csv file '
        'in the data/intermediates directory.'))
    parser.add_argument('--data', dest='data_dir', type=str, default=None,
                        help='Directory in which raw and preprocessed data is stored (default is ../data/ directory)')
    parser.add_argument('--out', dest='output_dir', type=str, default=None,
                        help='Directory in which to output (default is data/intermediates directory)')
    parser.add_argument('--cohort', dest='cohort_dir', type=str, default=None,
                        help='Directory of cohort')

    args = parser.parse_args()
    data_dir = args.data_dir or os.path.join(PARENT_DIR, 'data')
    out_dir = args.output_dir or os.path.join(data_dir, 'intermediates')
    cohort_dir = args.cohort_dir or os.path.join(data_dir, 'mimiciv')

    onset_data = load_csv(os.path.join(data_dir, "intermediates", "sepsis_onset.csv"))
    cohort_data = load_csv(os.path.join(data_dir, cohort_dir, "sepsis_cohort.csv"))

    onset_data = onset_data.drop(columns=['onset_time'])

    onset_data = onset_data.merge(cohort_data['onset_time'], how='left', left_index=True, right_index=True).reset_index(drop=True)
    # sepsis = pd.merge(sepsis, qstime[C_ONSET_TIME], how='left', left_index=True, right_index=True)

    onset_data.to_csv(os.path.join(out_dir, "sepsis_onset.csv"), index=False)