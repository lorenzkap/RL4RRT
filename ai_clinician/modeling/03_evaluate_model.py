import numpy as np
import pandas as pd
import tqdm
import argparse
import os
import shutil
import pickle
from ai_clinician.modeling.models.komorowski_model import *
from ai_clinician.modeling.models.common import *
from ai_clinician.modeling.columns import C_OUTCOME
from ai_clinician.preprocessing.utils import load_csv
from ai_clinician.preprocessing.columns import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

tqdm.tqdm.pandas()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=('Evaluates an AI Clinician model on the MIMIC-IV test set.'))
    parser.add_argument('data', type=str,
                        help='Model data directory (should contain train and test directories)')
    parser.add_argument('model', type=str,
                        help='Path to pickle file containing the model')
    parser.add_argument('--out', dest='out_path', type=str, default=None,
                        help='Path to pickle file at which to write out results (optional)')
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.99,
                        help='Decay for reward values (default 0.99)')
    parser.add_argument('--soften-factor', dest='soften_factor', type=float, default=0.01,
                        help='Amount by which to soften factors (random actions will be chosen this proportion of the time)')
    parser.add_argument('--num-iter-ql', dest='num_iter_ql', type=int, default=6,
                        help='Number of bootstrappings to use for TD learning (physician policy)')
    parser.add_argument('--num-iter-wis', dest='num_iter_wis', type=int, default=750,
                        help='Number of bootstrappings to use for WIS estimation (AI policy)')
    args = parser.parse_args()
    
    data_dir = args.data
    model = AIClinicianModel.load(args.model)
    assert model.metadata is not None, "Model missing metadata needed to generate actions"

    n_cluster_states = model.n_cluster_states

    # Update the number of actions to 2 (binary actions)
    n_actions = 2
    model.n_actions = n_actions  # Update model's n_actions

    # Define action_medians and action_bins for binary actions
    action_medians = np.array([0, 1])
    action_bins = np.array([0, 0.5, 1])

    # Update model's action_bins and action_medians
    model.metadata['actions']['action_bins'] = action_bins
    model.metadata['actions']['action_medians'] = action_medians

    MIMICraw = load_csv(os.path.join(data_dir, "test", "MIMICraw.csv"))
    MIMICzs = load_csv(os.path.join(data_dir, "test", "MIMICzs.csv"))
    metadata = load_csv(os.path.join(data_dir, "test", "metadata.csv"))
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
    rrt_actions = (~MIMICraw[rrt_cols].isna() & (MIMICraw[rrt_cols] != 0)).any(axis=1)
    MIMICraw['action'] = rrt_actions.astype(int)

    # Actions array
    actions = MIMICraw['action'].values

    np.seterr(divide='ignore', invalid='ignore')
    
    blocs = metadata[C_BLOC].values
    stay_ids = metadata[C_ICUSTAYID].values
    outcomes = metadata[C_OUTCOME].values

    print("Evaluate on MIMIC test set")
    states = model.compute_states(MIMICzs.values)
    
    records = build_complete_record_sequences(
        metadata,
        states,
        actions,
        model.absorbing_states,
        model.rewards
    )
    
    test_bootql = evaluate_physician_policy_td(
        records,
        model.physician_policy,
        args.gamma,
        args.num_iter_ql,
        model.n_cluster_states
    )
    
    phys_probs = model.compute_physician_probabilities(states=states, actions=actions)
    model_probs = model.compute_probabilities(states=states, actions=actions)
    test_bootwis, _,  _ = evaluate_policy_wis(
        metadata,
        phys_probs,
        model_probs,
        model.rewards,
        args.gamma,
        args.num_iter_wis
    )

    model_stats = {}
    
    model_stats['test_bootql_0.95'] = np.quantile(test_bootql, 0.95)   # PHYSICIANS' 95% UB
    model_stats['test_bootql_mean'] = np.nanmean(test_bootql)
    model_stats['test_bootql_0.99'] = np.quantile(test_bootql, 0.99)
    model_stats['test_bootwis_mean'] = np.nanmean(test_bootwis)    
    model_stats['test_bootwis_0.01'] = np.quantile(test_bootwis, 0.01)  
    wis_95lb = np.quantile(test_bootwis, 0.05)  # AI 95% LB, we want this as high as possible
    model_stats['test_bootwis_0.05'] = wis_95lb
    model_stats['test_bootwis_0.95'] = np.quantile(test_bootwis, 0.95)
    print("Results:", model_stats)

    # Plotting
    fig = plt.figure(figsize=(20, 10))

    # Plot clinician's action distribution
    print('Drawing clinician policy bar...')
    actual_action_plot = fig.add_subplot(231)
    action_counts = np.bincount(actions, minlength=2)
    action_percentages = action_counts / len(actions)
    actual_action_plot.bar([0, 1], action_percentages, tick_label=['No RRT', 'RRT'])
    actual_action_plot.set_xlabel('Action')
    actual_action_plot.set_ylabel('Proportion')
    actual_action_plot.set_title('Clinicians\' policy')

    # Plot AI policy action distribution
    print('Drawing AI policy bar...')
    ai_action_plot = fig.add_subplot(232)
    optimal_actions = model.Q.argmax(axis=1)  # Ensure this works with binary actions
    optimal_action_counts = np.bincount(optimal_actions, minlength=2)
    optimal_action_percentages = optimal_action_counts / len(optimal_actions)
    ai_action_plot.bar([0, 1], optimal_action_percentages, tick_label=['No RRT', 'RRT'])
    ai_action_plot.set_xlabel('Action')
    ai_action_plot.set_ylabel('Proportion')
    ai_action_plot.set_title('AI policy')

    # Compare mortality rates between actions
    print('Comparing mortality rates between actions...')
    mortality_rates = []
    for action_value in [0, 1]:
        indices = np.where(actions == action_value)
        mortality = outcomes[indices]
        mortality_rate = np.mean(mortality)
        mortality_rates.append(mortality_rate)
    
    mortality_plot = fig.add_subplot(233)
    mortality_plot.bar([0, 1], mortality_rates, tick_label=['No RRT', 'RRT'])
    mortality_plot.set_xlabel('Action')
    mortality_plot.set_ylabel('Mortality Rate')
    mortality_plot.set_title('Mortality Rate by Action')

    plt.show()

    # Save figures
    output_dir = os.path.dirname(args.out_path) if args.out_path else data_dir
    fig_path = os.path.join(output_dir, "evaluation_results")
    os.makedirs(fig_path, exist_ok=True)
    
    fig.savefig(os.path.join(fig_path, "policy_evaluation.png"))
    plt.close(fig)

    # Save model stats if output path is provided
    if args.out_path is not None:
        with open(args.out_path, "wb") as file:
            pickle.dump(model_stats, file)
    print('Done.')
