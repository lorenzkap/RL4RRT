import numpy as np
import pandas as pd
import tqdm
import argparse
import os
import pickle
import matplotlib.pyplot as plt

# Import the AI Clinician model and helpers
from ai_clinician.modeling.models.komorowski_model import *
from ai_clinician.modeling.models.common import *
from ai_clinician.modeling.columns import C_OUTCOME, C_ICUSTAYID, C_BLOC
from ai_clinician.preprocessing.utils import load_csv

tqdm.tqdm.pandas()

tqdm.tqdm.pandas()

def create_args(model_path):
    """
    Simulate command-line arguments.
    The only parameter that changes is the model path.
    """
    parser = argparse.ArgumentParser(description=(
        'Evaluates an AI Clinician model on the MIMIC-IV test set.'
    ))
    parser.add_argument('data', type=str,
                        help='Model data directory (should contain train and test directories)')
    parser.add_argument('model', type=str,
                        help='Path to pickle file containing the model')
    parser.add_argument('--out', dest='out_path', type=str, default=None,
                        help='Path to write out results (optional)')
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.99,
                        help='Discount factor for reward values')
    parser.add_argument('--soften-factor', dest='soften_factor', type=float, default=0.05,
                        help='Softening factor (proportion of random actions)')
    parser.add_argument('--num-iter-ql', dest='num_iter_ql', type=int, default=6,
                        help='Number of bootstraps for TD learning (physician policy)')
    parser.add_argument('--num-iter-wis', dest='num_iter_wis', type=int, default=500,
                        help='Number of bootstraps for WIS estimation (AI policy)')
    
    simulated_input = [
        '/home/lkapral/RRT_mimic_iv/data/model',  # Data directory
        model_path,                              # Model path (will be updated in the loop)
        '--gamma', '0.99',
        '--soften-factor', '0.01',
        '--num-iter-ql', '6',
        '--num-iter-wis', '500'
    ]
    return parser.parse_args(simulated_input)


def evaluate_model(model_path, data_dir, MIMICzs, MIMICraw, metadata, actions_RRT, reduced_features):
    """
    Loads the model at model_path, sets it up for evaluation,
    computes the states and policies, and returns a dictionary of statistics.
    """
    # Create simulated args (with the proper model path)
    args = create_args(model_path)
    
    # Load the model
    model = AIClinicianModel.load(args.model)
    assert model.metadata is not None, "Model missing metadata needed to generate actions"
    
    # Update the model for binary actions
    n_actions = 2
    model.n_actions = n_actions
    action_medians = np.array([0, 1])
    action_bins = np.array([0, 0.5, 1])
    model.metadata['actions']['action_bins'] = action_bins
    model.metadata['actions']['action_medians'] = action_medians

    # Use only the reduced features plus the RRT column (create copies)
    MIMICzs_model = MIMICzs[reduced_features + ['RRT']].copy()
    MIMICraw_model = MIMICraw[reduced_features + ['RRT']].copy()

    # Compute the states using the current model
    states = model.compute_states(MIMICzs_model.values)

    # Build complete record sequences (used for policy evaluation)
    records = build_complete_record_sequences(
        metadata,
        states,
        actions_RRT,
        model.absorbing_states,
        model.rewards
    )

    # Evaluate the physician's policy using TD learning
    test_bootql = evaluate_physician_policy_td(
        records,
        model.physician_policy,
        args.gamma,
        args.num_iter_ql,
        model.n_cluster_states
    )

    # Also compute the physician policy (if needed)
    physpol_test, transitionr_test, R_test = compute_physician_policy(
        records,
        model.n_states,
        model.n_actions,
        model.absorbing_states,
        reward_val=model.reward_val,
        transition_threshold=model.transition_threshold,
    )

    # Compute probabilities for policy evaluation
    phys_probs = model.compute_physician_probabilities(states=states, actions=actions_RRT)
    model_probs = model.compute_probabilities(states=states, actions=actions_RRT)

    # Evaluate the AI policy using Weighted Importance Sampling (WIS)
    test_bootwis, _, _ = evaluate_policy_wis(
        metadata,
        phys_probs,
        model_probs,
        model.rewards,
        args.gamma,
        args.num_iter_wis
    )

    # Prepare a dictionary with the key statistics
    model_stats = {}
    model_stats['test_bootql_mean'] = np.nanmean(test_bootql)
    model_stats['test_bootql_0.95'] = np.quantile(test_bootql, 0.95)
    model_stats['test_bootql_0.99'] = np.quantile(test_bootql, 0.99)
    model_stats['test_bootwis_mean'] = np.nanmean(test_bootwis)
    model_stats['test_bootwis_0.01'] = np.quantile(test_bootwis, 0.01)
    model_stats['test_bootwis_0.05'] = np.quantile(test_bootwis, 0.05)
    model_stats['test_bootwis_0.95'] = np.quantile(test_bootwis, 0.95)

    # (Optional) Print out some evaluation details
    print("\nEvaluation Results for model:", model_path)
    print("Physician Policy TD - Mean:", model_stats['test_bootql_mean'])
    print("AI Policy WIS - Mean:", model_stats['test_bootwis_mean'])

    return model_stats

def main():
    # Set up the data directory (assumed to be constant)
    data_dir = '/home/lkapral/RRT_mimic_iv/data/model'
    
    # Load the feature importance (for selecting features)
    fixed_num_features = 40
    feature_importance = pd.read_csv(os.path.join(data_dir, 'combined_feature_importances.csv'))
    weights = feature_importance.head(fixed_num_features)['Combined_Average'].values
    feature_weights = weights / np.linalg.norm(weights)
    reduced_features = feature_importance.head(fixed_num_features)['Feature'].tolist()

    # Load test data (MIMIC raw and standardized, plus metadata)
    AKHraw = pd.read_parquet('/home/lkapral/RRT_mimic_iv/data/model/AKH_preprocessed.parquet')
    from ai_clinician.modeling.normalization import DataNormalization
    #MIMIC_train = load_csv(os.path.join(data_dir, "train", "MIMICraw.csv"))
    normer = DataNormalization.load('/home/lkapral/RRT_mimic_iv/data/model/normalization.pkl')
    AKHzs = normer.transform(AKHraw)

    
    metadata = pd.DataFrame(AKHraw['bloc']).copy()
    metadata['icustayid'] = AKHraw['encounterId']
    metadata['outcome'] = AKHraw['hospital_mortality']
    # Define the RRT-related columns and create an 'action' column

    unique_icu_stays = metadata[C_ICUSTAYID].unique()
    
    # Create actions based on RRT
    print("Create actions")
    
    # Define RRT-related columns
    rrt_cols = [
        'Ultrafiltrate_Output',
        'Blood_Flow',
        'Hourly_Patient_Fluid_Removal',
        'Dialysate_Rate',
        'Hemodialysis_Output',  # Ensure the column name matches your DataFrame
        'Citrate',
        'Prefilter_Replacement_Rate',
        'Postfilter_Replacement_Rate'
    ]
    
    # Create 'action' column
    rrt_actions = (~AKHraw[rrt_cols].isna() & (AKHraw[rrt_cols] != 0)).any(axis=1)
    AKHraw['action'] = rrt_actions.astype(int)
    # Actions array
    AKHraw['action'].fillna(0, inplace=True)
    
    np.seterr(divide='ignore', invalid='ignore')
    
    AKHraw['RRT'] = AKHraw['action'].copy()
    AKHzs['RRT'] = AKHraw['RRT'].copy()
    actions_RRT = AKHraw['action'].values
    
    # Update the number of actions to 2 (binary actions)
    n_actions = 2

    
    AKHraw_full = AKHraw.copy()

    AKHzs = AKHzs[reduced_features+ ['RRT']]
    AKHraw = AKHraw[reduced_features + ['RRT']]
    feature_weights = np.append(feature_weights,1)

    # CSV file to which we will append the evaluation results
    csv_file = os.path.join(data_dir, "evaluation_results_muw.csv")
    
    # Create the CSV file with header if it does not exist
    if not os.path.exists(csv_file):
        header = ['penalty', 'model_index', 
                  'test_bootql_mean', 'test_bootql_0.95', 'test_bootql_0.99', 
                  'test_bootwis_mean', 'test_bootwis_0.01', 'test_bootwis_0.05', 'test_bootwis_0.95']
        pd.DataFrame(columns=header).to_csv(csv_file, index=False)
    
    # Define the penalty values (from 0.00 to 0.25 in 0.01 steps)
    penalty_values = np.linspace(0.0, 0.25, 26)
    
    # Loop over each penalty value and each of the five model indices (0-4)
    for penalty in penalty_values:
        # Format the penalty string to remove extra zeros.
        penalty_str = format(penalty, '.2f').rstrip('0').rstrip('.')
        if '.' not in penalty_str:
            penalty_str += '.0'
        for model_idx in range(5):
            # Build the model file path.
            # Example: /home/lkapral/RRT_mimic_iv/data/model/models_penal/0.17/model_params_40/top5/top5_model_0.pkl
            model_filename = f"top5_model_{model_idx}.pkl"
            model_path = os.path.join(data_dir, "models_penal", penalty_str, "model_params_40", "top5", model_filename)
            
            print(f"\nEvaluating model at penalty {penalty_str}, model index {model_idx}...")
            try:
                # Evaluate the model (returns a dictionary with evaluation stats)
                model_stats = evaluate_model(model_path, data_dir, AKHzs, AKHraw, metadata, actions_RRT, reduced_features)
            except Exception as e:
                print(f"Error evaluating model at penalty {penalty_str}, index {model_idx}: {e}")
                continue  # Skip to the next model if there is an error
            
            # Prepare a row with the results
            row = {
                'penalty': penalty_str,
                'model_index': model_idx,
                'test_bootql_mean': model_stats.get('test_bootql_mean', np.nan),
                'test_bootql_0.95': model_stats.get('test_bootql_0.95', np.nan),
                'test_bootql_0.99': model_stats.get('test_bootql_0.99', np.nan),
                'test_bootwis_mean': model_stats.get('test_bootwis_mean', np.nan),
                'test_bootwis_0.01': model_stats.get('test_bootwis_0.01', np.nan),
                'test_bootwis_0.05': model_stats.get('test_bootwis_0.05', np.nan),
                'test_bootwis_0.95': model_stats.get('test_bootwis_0.95', np.nan)
            }
            
            # Convert the row into a DataFrame and append it to the CSV.
            # (The CSV is updated every iteration.)
            df_row = pd.DataFrame([row])
            df_row.to_csv(csv_file, mode='a', header=False, index=False)
            print("Evaluation results saved to CSV.")

if __name__ == '__main__':
    main()
