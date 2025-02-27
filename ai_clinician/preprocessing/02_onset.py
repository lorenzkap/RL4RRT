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
    parser = argparse.ArgumentParser(description=('Generates the '
        'onset.csv file in the data/intermediates directory.'))
    parser.add_argument('--data', dest='data_dir', type=str, default=None,
                        help='Directory in which raw and preprocessed data is stored (default is ../data/ directory)')
    parser.add_argument('--out', dest='output_dir', type=str, default=None,
                        help='Directory in which to output (default is data/intermediates directory)')

    args = parser.parse_args()
    data_dir = args.data_dir or os.path.join(PARENT_DIR, 'data')
    out_dir = args.output_dir or os.path.join(data_dir, 'intermediates')

    demog = load_intermediate_or_raw_csv(data_dir, "demog.csv")
    onset_data = demog[["subject_id", "icustayid", "intime"]]
    onset_data = onset_data.rename(columns={"intime": "onset_time"})
    onset_data.to_csv(os.path.join(out_dir, "onset.csv"), index=False)