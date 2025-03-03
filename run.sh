#!/bin/bash

command=$1

if [ "$command" = "extract" ]; then

  echo "EXTRACT DATA"
  echo "Using client secret: $2"
  echo "Using BigQuery project: $3"
  echo

  mkdir -p data
  python ai_clinician/data_extraction/extract.py $2 $3

  echo "DONE"

elif [ "$command" = "preprocess" ]; then

  echo "PREPROCESS DATA"
  echo "Using final data output directory name (in data/): $2"
  echo

  # echo "1/12 PREPROCESS RAW DATA"
  # python ai_clinician/preprocessing/01_preprocess_raw_data.py --no-bacterio || exit 1

  # echo "2/12 GENERATE ONSET"
  # python ai_clinician/preprocessing/02_onset.py || exit 1

  # echo "3/12 BUILD PATIENT STATES - COHORT"
  # python ai_clinician/preprocessing/03_build_patient_states.py data/intermediates/cohort --window-before 1 --window-after 241 || exit 1

  # echo "4/12 IMPUTE STATES - COHORT"
  # python ai_clinician/preprocessing/04_impute_states.py data/intermediates/cohort/patient_states.csv data/intermediates/cohort/patient_states_filled.csv --mask-file data/intermediates/cohort/state_imputation_mask.csv || exit 1

  # echo "5/12 BUILD STATES AND ACTIONS - COHORT"
  # python ai_clinician/preprocessing/05_build_states_and_actions.py data/intermediates/cohort/patient_states_filled.csv data/intermediates/cohort/qstime.csv data/intermediates/cohort/states_and_actions.csv --window-before 1 --window-after 241 --mapping-file data/intermediates/cohort/bin_mapping.csv || exit 1

  # echo "6/12 IMPUTE STATES AND ACTIONS - COHORT"
  # python ai_clinician/preprocessing/06_impute_states_actions.py data/intermediates/cohort/states_and_actions.csv data/intermediates/cohort/states_and_actions_filled.csv --mask-file data/intermediates/cohort/states_and_actions_mask.csv || exit 1

  # echo "7/12 BUILD AKI COHORT"
  # python ai_clinician/preprocessing/07_build_aki_cohort.py data/intermediates/cohort/states_and_actions_filled.csv data/intermediates/cohort/qstime.csv data/$2 || exit 1

  # echo "8/12 BUILD PATIENT STATES - MDP"
  # python ai_clinician/preprocessing/03_build_patient_states.py data/intermediates/mdp --window-before 25 --window-after 241 || exit 1

  # echo "9/12 IMPUTE STATES - MDP"
  # python ai_clinician/preprocessing/04_impute_states.py data/intermediates/mdp/patient_states.csv data/intermediates/mdp/patient_states_filled.csv --mask-file data/intermediates/mdp/state_imputation_mask.csv || exit 1

  # echo "10/12 BUILD STATES AND ACTIONS - MDP"
  # python ai_clinician/preprocessing/05_build_states_and_actions.py data/intermediates/mdp/patient_states_filled.csv data/intermediates/mdp/qstime.csv data/intermediates/mdp/states_and_actions.csv --window-before 25 --window-after 241 --mapping-file data/intermediates/mdp/bin_mapping.csv || exit 1

  # echo "11/12 IMPUTE STATES AND ACTIONS - MDP"
  # python ai_clinician/preprocessing/06_impute_states_actions.py data/intermediates/mdp/states_and_actions.csv data/intermediates/mdp/mimic_dataset.csv --mask-file data/intermediates/mdp/states_and_actions_mask.csv || exit 1

  echo "12/12 CORRECTING HEIGHT AND WEIGHT"
  python ai_clinician/preprocessing/08_height_weight.py data/intermediates/mdp/mimic_dataset.csv data/$2/mimic_dataset.csv || exit 1
  
  echo "DONE"

elif [ "$command" = "model" ]; then

  echo "BUILD MODELS"
  echo "Using MIMIC dataset directory (in data/): $2"
  echo "Using model output directory (in data/): $3"
  echo

  echo "GENERATE DATASETS"
  python ai_clinician/modeling/01_generate_datasets.py data/$2 data/$3 || exit 1

  echo "TRAIN MODELS"
  # python ai_clinician/modeling/02_train_models.py data/$3 --model-type DuelingDQN --n-models 2 || exit 1
  python ai_clinician/modeling/02_train_models.py data/$3 --n-models 100 || exit 1

  echo "DONE"
fi
