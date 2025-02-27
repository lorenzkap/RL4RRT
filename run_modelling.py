import numpy as np
import pandas as pd
import tqdm
import argparse
import os
import shutil
from ai_clinician.preprocessing.utils import load_csv
from ai_clinician.preprocessing.columns import *
from ai_clinician.modeling.models.komorowski_model import AIClinicianModel
from ai_clinician.modeling.models.common import *
from ai_clinician.modeling.models.dqn import DuelingDQNModel
from ai_clinician.modeling.columns import C_OUTCOME
import pickle
from sklearn.model_selection import train_test_split

tqdm.tqdm.pandas()

pd.set_option('display.max_columns', None)

df = pd.read_csv('/home/lkapral/RRT_mimic_iv/data/mimic/mimic_dataset.csv')

import argparse
import os
import shutil
import pandas as pd

# Function to create and parse arguments
def create_args():
    parser = argparse.ArgumentParser(description='Simulate command-line argument parsing in Jupyter notebook.')
    parser.add_argument('data', type=str,
                        help='Model data directory (should contain train and test directories)')
    parser.add_argument('--worker-label', dest='worker_label', type=str, default='',
                        help='Label to suffix output files')
    parser.add_argument('--save', dest='save_behavior', type=str, default='best',
                        help='Models to save (best [default], all, none)')
    parser.add_argument('--val-size', dest='val_size', type=float, default=0.2,
                        help='Proportion of data to use for validation')
    parser.add_argument('--n-models', dest='n_models', type=int, default=500,
                        help='Number of models to build')
    parser.add_argument('--model-type', dest='model_type', type=str, default='AIClinician',
                        help='Model type to train (AIClinician or DuelingDQN)')
    parser.add_argument('--cluster-fraction', dest='cluster_fraction', type=float, default=0.25,
                        help='Fraction of patient states to sample for state clustering')
    parser.add_argument('--n-cluster-init', dest='n_cluster_init', type=int, default=32,
                        help='Number of cluster initializations to try in each replicate')
    parser.add_argument('--n-cluster-states', dest='n_cluster_states', type=int, default=500,
                        help='Number of states to define through clustering')
    parser.add_argument('--n-action-bins', dest='n_action_bins', type=int, default=5,
                        help='Number of action bins for fluids and vasopressors')
    parser.add_argument('--reward', dest='reward', type=int, default=100,
                        help='Value to assign as positive reward if discharged from hospital, or negative reward if died')
    parser.add_argument('--transition-threshold', dest='transition_threshold', type=int, default=5,
                        help='Prune state-action pairs with less than this number of occurrences in training data')
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.99,
                        help='Decay for reward values (default 0.99)')
    parser.add_argument('--soften-factor', dest='soften_factor', type=float, default=0.01,
                        help='Amount by which to soften factors (random actions will be chosen this proportion of the time)')
    parser.add_argument('--num-iter-ql', dest='num_iter_ql', type=int, default=6,
                        help='Number of bootstrappings to use for TD learning (physician policy)')
    parser.add_argument('--num-iter-wis', dest='num_iter_wis', type=int, default=700,
                        help='Number of bootstrappings to use for WIS estimation (AI policy)')
    parser.add_argument('--state-dim', dest='state_dim', type=int, default=256,
                        help='Dimension for learned state representation in DQN')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=128,
                        help='Number of units in hidden layer for DQN')

    # Simulate input arguments as if they were passed from the command line
    simulated_input = '--n-models 1500 --model-type AIClinician --n-action-bins 2 --val-size 0.2'.split()
    simulated_input.insert(0, '/home/lkapral/RRT_mimic_iv/data/model')
    return parser.parse_args(simulated_input)

# Create args object
args = create_args()

# Now, the rest of your script can use these args as if they were passed from the command line
data_dir = args.data
worker_label = args.worker_label
n_models = args.n_models
model_type = args.model_type
n_action_bins = args.n_action_bins
fixed_num_features = 40
# and so on for other arguments

# You can now use these variables in your code to set up directories, load data, etc.


# Loading CSV files
MIMICraw = pd.read_csv(os.path.join(data_dir, "train", "MIMICraw.csv"))
MIMICzs = pd.read_csv(os.path.join(data_dir, "train", "MIMICzs.csv"))
metadata = pd.read_csv(os.path.join(data_dir, "train", "metadata.csv"))
unique_icu_stays = metadata['icustayid'].unique()






feature_importance = pd.read_csv('/home/lkapral/RRT_mimic_iv/data/model/combined_feature_importances.csv')

weights = feature_importance.head(fixed_num_features)['Combined_Average'].values
feature_weights = weights / np.linalg.norm(weights)

reduced_features = feature_importance.head(fixed_num_features)['Feature'].tolist()



len(unique_icu_stays)

metadata.columns





MIMICraw

print("Create actions")

# List of RRT-related columns
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

# Create a boolean mask where any RRT column is not NaN and not zero
rrt_actions = (~MIMICraw[rrt_cols].isna() & (MIMICraw[rrt_cols] != 0)).any(axis=1)
MIMICraw['action'] = rrt_actions.astype(int)

# Update the number of actions
n_actions = 2  # Actions are now binary: 0 or 1

# Simplified fit_action_bins function for binary actions
def fit_action_bins_binary(actions):
    action_medians = np.array([0, 1])
    action_bins = np.array([0, 0.5, 1])
    all_actions = actions.values
    return all_actions, action_medians, action_bins

# Create all_actions, action_medians, action_bins
all_actions, action_medians, action_bins = fit_action_bins_binary(MIMICraw['action'])



model_type = args.model_type

np.seterr(divide='ignore', invalid='ignore')


MIMICraw['RRT'] = MIMICraw['action']
MIMICzs['RRT'] = MIMICraw['action']

MIMICzs = MIMICzs[reduced_features+ ['RRT']]
MIMICraw = MIMICraw[reduced_features + ['action']]
feature_weights = np.append(feature_weights,1)

MIMICraw['icustayid'] = metadata['icustayid']
len(MIMICraw[MIMICraw['action']>0]['icustayid'].unique())



MIMICraw['RRT'] = MIMICraw['action']
MIMICzs['RRT'] = MIMICraw['action']

# MIMICraw['icustayid'] = metadata['icustayid']

# MIMICraw['action'] = MIMICraw.groupby('icustayid')['action'].transform(
#     lambda x: x.ne(x.shift().fillna(x)).astype(int)
# )
# MIMICraw.drop(columns=['icustayid'], inplace=True)



print("Action distribution:")
print(MIMICraw['action'].value_counts())

print("Action medians:")
print(action_medians)

print("Action bins:")
print(action_bins)






args.n_models

def run_penalty_experiment(
    number_of_models,
    penal_amount, fixed_num_features,
    X_train, X_val, metadata_train, metadata_val, actions_train, actions_val,
    data_dir, args, model_type, AIClinicianModel, DuelingDQNModel,
    n_action_bins, action_bins, action_medians, 
    feature_weights, build_complete_record_sequences,
    evaluate_physician_policy_td, evaluate_policy_wis,
    train_ids, val_ids
):

    out_dir = os.path.join(data_dir, "models_penal", str(penal_amount))
    os.makedirs(out_dir, exist_ok=True)
    model_specs_dir = os.path.join(out_dir, f"model_params_{fixed_num_features}")
    os.makedirs(model_specs_dir, exist_ok=True)

    all_model_stats = []
    top_models = []  # Stores tuples of (wis_score, model, additional_vars)

    # Create directories for best/top5 models
    best_dir = os.path.join(model_specs_dir, 'best')
    top5_dir = os.path.join(model_specs_dir, 'top5')
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(top5_dir, exist_ok=True)
    
    # Loop over the number of models to train per penalty value
    for modl in range(number_of_models):
        print(f"Penalty: {penal_amount}, Model {modl} of {number_of_models}")

        base_model = AIClinicianModel(
            n_cluster_states=args.n_cluster_states,
            n_actions=n_actions,
            cluster_fit_fraction=args.cluster_fraction,
            n_cluster_init=args.n_cluster_init,
            gamma=args.gamma,
            reward_val=args.reward,
            transition_threshold=args.transition_threshold,
            penalise_action_amount=penal_amount
        )

        if model_type == 'DuelingDQN':
            model = DuelingDQNModel(
                state_dim=args.state_dim,
                n_actions=n_actions,
                hidden_dim=args.hidden_dim,
                gamma=args.gamma,
                reward_val=args.reward
            )
        else:
            model = base_model

        # Train the base model
        base_model.train(
            X_train.values,
            actions_train,
            metadata_train,
            X_val=X_val.values,
            actions_val=actions_val,
            metadata_val=metadata_val,
            feature_weights=feature_weights,
        )

        # If a specialized model is used (e.g. DuelingDQN), train it too
        if model != base_model:
            model.train(
                X_train.values,
                actions_train,
                metadata_train,
                X_val=X_val.values,
                actions_val=actions_val,
                metadata_val=metadata_val
            )

        ####### EVALUATE ON TRAIN SET #######
        states_train = base_model.compute_states(X_train.values)
        print("Evaluate on MIMIC training set")
        records_train = build_complete_record_sequences(
            metadata_train,
            states_train,
            actions_train,
            base_model.absorbing_states,
            base_model.rewards
        )

        train_bootql = evaluate_physician_policy_td(
            records_train,
            base_model.physician_policy,
            args.gamma,
            args.num_iter_ql,
            args.n_cluster_states
        )

        phys_probs_train = base_model.compute_physician_probabilities(states=states_train, actions=actions_train)
        model_probs_train = model.compute_probabilities(X=X_train.values, actions=actions_train)
        train_bootwis, _, _ = evaluate_policy_wis(
            metadata_train,
            phys_probs_train,
            model_probs_train,
            base_model.rewards,
            args.gamma,
            args.num_iter_wis
        )

        model_stats = {}
        model_stats['train_bootql_mean'] = np.nanmean(train_bootql)
        model_stats['train_bootql_0.99'] = np.quantile(train_bootql, 0.99)
        model_stats['train_bootql_0.95'] = np.quantile(train_bootql, 0.95)
        model_stats['train_bootwis_mean'] = np.nanmean(train_bootwis)
        model_stats['train_bootwis_0.05'] = np.quantile(train_bootwis, 0.05)
        model_stats['train_bootwis_0.95'] = np.quantile(train_bootwis, 0.95)

        ####### EVALUATE ON VALIDATION SET #######
        print("Evaluate on MIMIC validation set")
        states_val = base_model.compute_states(X_val.values)

        records_val = build_complete_record_sequences(
            metadata_val,
            states_val,
            actions_val,
            base_model.absorbing_states,
            base_model.rewards
        )

        val_bootql = evaluate_physician_policy_td(
            records_val,
            base_model.physician_policy,
            args.gamma,
            args.num_iter_ql,
            args.n_cluster_states
        )

        phys_probs_val = base_model.compute_physician_probabilities(states=states_val, actions=actions_val)
        phys_probs_all_val = base_model.compute_physician_probabilities(states=states_val)
        model_probs_val = model.compute_probabilities(X=X_val.values, actions=actions_val)
        model_probs_all_val = model.compute_probabilities(X=X_val.values)

        val_bootwis, _, _ = evaluate_policy_wis(
            metadata_val,
            phys_probs_val,
            model_probs_val,
            base_model.rewards,
            args.gamma,
            args.num_iter_wis
        )

        model_stats['val_bootql_0.95'] = np.quantile(val_bootql, 0.95)
        model_stats['val_bootql_mean'] = np.nanmean(val_bootql)
        model_stats['val_bootql_0.99'] = np.quantile(val_bootql, 0.99)
        model_stats['val_bootwis_mean'] = np.nanmean(val_bootwis)
        model_stats['val_bootwis_0.01'] = np.quantile(val_bootwis, 0.01)
        wis_95lb = np.quantile(val_bootwis, 0.05)
        model_stats['val_bootwis_0.05'] = wis_95lb
        model_stats['val_bootwis_0.95'] = np.quantile(val_bootwis, 0.95)
        print("95% LB: {:.2f}".format(wis_95lb))

        all_model_stats.append(model_stats)

 # Create temporary additional_vars structure
        current_additional_vars = {
            'states_val': states_val,
            'X_val': X_val.values,
            'actions_val': actions_val,
            'metadata_val': metadata_val,
            'phys_probs': base_model.compute_physician_probabilities(states=states_val),
            'model_probs': model.compute_probabilities(X=X_val.values),
            'rewards': base_model.rewards,
            'gamma': args.gamma
        }

        # Maintain top models in memory
        top_models.append((wis_95lb, model, current_additional_vars))
        top_models.sort(key=lambda x: x[0], reverse=True)  # Sort descending
        top_models = top_models[:5]  # Keep only top 5

        # Save/overwrite best model
        if top_models:
            # Save best model
            top_models[0][1].save(
                os.path.join(best_dir, 'best_model.pkl'),
                metadata={
                    'actions': {
                        'n_action_bins': n_action_bins,
                        'action_bins': action_bins,
                        'action_medians': action_medians
                    },
                    'split': {'train_ids': train_ids, 'val_ids': val_ids},
                    'eval_params': {
                        'num_iter_ql': args.num_iter_ql,
                        'num_iter_wis': args.num_iter_wis
                    },
                    'wis_score': top_models[0][0],
                    'rank': 0,
                    'model_num': modl
                }
            )
            # Save best additional vars
            with open(os.path.join(best_dir, 'additional_vars_best.pkl'), 'wb') as f:
                pickle.dump(top_models[0][2], f)

        # Save/overwrite top 5 models
        for idx, (score, model_obj, vars_obj) in enumerate(top_models):
            model_obj.save(
                os.path.join(top5_dir, f'top5_model_{idx}.pkl'),
                metadata={
                    'actions': {
                        'n_action_bins': n_action_bins,
                        'action_bins': action_bins,
                        'action_medians': action_medians
                    },
                    'split': {'train_ids': train_ids, 'val_ids': val_ids},
                    'eval_params': {
                        'num_iter_ql': args.num_iter_ql,
                        'num_iter_wis': args.num_iter_wis
                    },
                    'wis_score': score,
                    'rank': idx+1,
                    'model_num': modl
                }
            )
            with open(os.path.join(top5_dir, f'additional_vars_top5_{idx}.pkl'), 'wb') as f:
                pickle.dump(vars_obj, f)

        # Save statistics
        pd.DataFrame(all_model_stats).to_csv(
            os.path.join(out_dir, "model_stats.csv"),
            index=False,
            float_format='%.6f'
        )

    print(f'Penalty {penal_amount} complete. Best WIS 95% LB: {top_models[0][0]:.2f}')
    return penal_amount, all_model_stats


from joblib import Parallel, delayed
import multiprocessing
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

train_ids, val_ids = train_test_split(unique_icu_stays, test_size=args.val_size)
train_indexes = metadata[metadata[C_ICUSTAYID].isin(train_ids)].index
val_indexes = metadata[metadata[C_ICUSTAYID].isin(val_ids)].index

X_train = MIMICzs.iloc[train_indexes]
X_val = MIMICzs.iloc[val_indexes]
metadata_train = metadata.iloc[train_indexes]
metadata_val = metadata.iloc[val_indexes]
actions_train = all_actions[train_indexes]
actions_val = all_actions[val_indexes]


penal_amounts = [i/100. for i in range(-15, 35, 1)]
results = Parallel(n_jobs=len(penal_amounts)+1)(
    delayed(run_penalty_experiment)(
        args.n_models,
        penal_amount, 
        fixed_num_features,
        X_train, X_val, metadata_train, metadata_val, actions_train, actions_val,
        data_dir, args, model_type, AIClinicianModel, DuelingDQNModel,
        n_action_bins, action_bins, action_medians, 
        feature_weights, build_complete_record_sequences,
        evaluate_physician_policy_td, evaluate_policy_wis,
        train_ids, val_ids
    )
    for penal_amount in penal_amounts)
