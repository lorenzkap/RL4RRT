import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import MiniBatchKMeans, KMeans
from ai_clinician.preprocessing.columns import *
from ai_clinician.modeling.columns import *
from ai_clinician.modeling.models.MDPtoolbox import mdp_policy_iteration_with_Q
from ai_clinician.modeling.models.offpolicy import off_policy_q_learning
from ai_clinician.modeling.models.common import build_complete_record_sequences, compute_physician_policy
from ai_clinician.modeling.models.base_ import BaseModel

class AIClinicianModel(BaseModel):
    def __init__(self,
                 n_cluster_states=750,
                 n_actions=25,
                 cluster_fit_fraction=0.25,
                 n_cluster_init=32,
                 gamma=0.99,
                 reward_val=100,
                 transition_threshold=5,
                 soften_factor=0.01,
                 random_state=None,
                 metadata=None,
                 penalise_action_amount=0):
        super(AIClinicianModel, self).__init__()
        self.n_cluster_states = n_cluster_states
        self.n_actions = n_actions
        self.cluster_fit_fraction = cluster_fit_fraction
        self.n_cluster_init = n_cluster_init
        self.gamma = gamma
        self.random_state = random_state
        self.reward_val = reward_val
        self.transition_threshold = transition_threshold
        self.soften_factor = soften_factor
        self.metadata = metadata
        self.n_states = self.n_cluster_states + 2
        self.absorbing_states = [self.n_cluster_states + 1, self.n_cluster_states] # absorbing state numbers
        self.rewards = [self.reward_val, -self.reward_val]
        self.feature_weights = None
        self.penalise_action_amount = penalise_action_amount
        
        self.clusterer = None
        self.Q = None
        self.physician_policy = None
        self.transitionr = None
        self.R = None
    
    def _cluster_states_old(self, state_data):
        """
        Produces a clustering of the given state data, where each state is
        considered independent (even from the same patient).
        
        Returns: a clustering object that can be queried using a predict() function,
            and an array of clustering indexes ranging from 0 to n_clusters.
        """
        sample = state_data[np.random.choice(len(state_data),
                                            size=int(len(state_data) * self.cluster_fit_fraction),
                                            replace=False)]
        clusterer = MiniBatchKMeans(n_clusters=self.n_cluster_states,
                                    random_state=self.random_state,
                                    n_init=self.n_cluster_init,
                                    max_iter=30).fit(sample)
        return clusterer, clusterer.predict(state_data)
    


    def _cluster_states(self, state_data):
        """
        Produces a clustering of the given state data using K-Means with feature weights.
        Each state is considered independent (even from the same patient).

        Parameters:
            state_data (np.ndarray or pd.DataFrame): The data to be clustered, shape (n_samples, n_features).
            feature_weights (list or np.ndarray): A list or array of weights for each feature,
                                                 length should be equal to n_features.

        Returns:
            clusterer (KMeans): The fitted KMeans clustering object.
            cluster_indexes (np.ndarray): Array of clustering indexes ranging from 0 to n_clusters - 1.
        """
        # -----------------------
        # 1. Validate Inputs
        # -----------------------

        # Ensure state_data is a NumPy array
        state_data = np.asarray(state_data)

        # Scale each feature by its corresponding weight
        weighted_data = state_data * self.feature_weights

        # Determine the number of samples to use for fitting
        sample_size = int(len(weighted_data) * self.cluster_fit_fraction)

        if sample_size < self.n_cluster_states:
            raise ValueError(
                f"Sample size ({sample_size}) is smaller than the number of clusters ({self.n_cluster_states}). "
                "Increase cluster_fit_fraction or reduce n_cluster_states."
            )

        # Randomly select indices without replacement using a reproducible random state
        rng = np.random.default_rng(seed=self.random_state)
        sample_indices = rng.choice(len(weighted_data), size=sample_size, replace=False)

        # Extract the sampled data
        sample = weighted_data[sample_indices]

        # -----------------------
        # 4. Initialize and Fit KMeans
        # -----------------------

        # Initialize the KMeans clusterer with desired parameters
        clusterer = MiniBatchKMeans(
            n_clusters=self.n_cluster_states,
            n_init=self.n_cluster_init,
            max_iter=1000,  # Increased max_iter for better convergence
            random_state=self.random_state
        )

        # Fit KMeans on the sampled weighted data
        clusterer.fit(sample)

        # -----------------------
        # 5. Predict Cluster Assignments for All Data
        # -----------------------

        # Predict cluster assignments for the entire weighted dataset
        cluster_indexes = clusterer.predict(weighted_data)

        return clusterer, cluster_indexes


    
    def train(self, X_train, actions_train, metadata_train, X_val=None, actions_val=None, metadata_val=None, feature_weights=None, penalize_action=True):
        print("Clustering")
        if feature_weights is not None:
            # Convert feature_weights to a NumPy array
            feature_weights = np.array(feature_weights)

            # Validate feature_weights
            if feature_weights.ndim != 1:
                raise ValueError(f"feature_weights must be a 1-dimensional array, got shape {feature_weights.shape}.")
            if feature_weights.shape[0] != X_train.shape[1]:
                raise ValueError(
                    f"Length of feature_weights ({feature_weights.shape[0]}) does not match "
                    f"number of features in X_train ({X_train.shape[1]})."
                )
            self.feature_weights = feature_weights
        else:
            self.feature_weights = np.ones(X_train.shape[1])
        self.clusterer, states_train = self._cluster_states(X_train)
        
        # Create qldata3
        qldata3 = build_complete_record_sequences(
            metadata_train,
            states_train,
            actions_train,
            self.absorbing_states,
            self.rewards
        )
        
        
        
        ####### BUILD MODEL ########
        physpol, transitionr, R = compute_physician_policy(
            qldata3,
            self.n_states,
            self.n_actions,
            self.absorbing_states,
            reward_val=self.reward_val,
            transition_threshold=self.transition_threshold
        )
   
        if(penalize_action):
            penalise_states = self.find_rare_states(states_train, actions_train, max_state=self.n_states)
            R = self.reduce_reward_of_action(R, penalise_states, amount=self.penalise_action_amount)
            
        self.R = R
        self.physician_policy = physpol
        self.transitionr = transitionr
        print("Policy iteration")
        self.Q = mdp_policy_iteration_with_Q(
            np.swapaxes(transitionr, 0, 1),
            R,
            self.gamma,
            np.ones(self.n_states)
        )[-1]

    def find_rare_states(self, states_train, actions_train, threshold=5, max_state=751):
        """
        Identifies states with a number of action=1 occurrences less than a specified threshold.
        
        Parameters:
        - states_train (np.ndarray): 1D array of integer state values ranging from 0 to max_state.
        - actions_train (np.ndarray): 1D array of binary action values (0 or 1).
        - threshold (int): The minimum number of action=1 occurrences required to include a state.
                           States with counts below this threshold will be returned.
                           Default is 5.
        - max_state (int): The maximum possible state value. Default is 751.
        
        Returns:
        - List[int]: A list of state integers with action=1 counts less than the threshold.
        
        Raises:
        - ValueError: If inputs are invalid, e.g., mismatched lengths, invalid state or action values.
        """
        # Validate inputs
        if not isinstance(states_train, np.ndarray):
            raise ValueError("states_train must be a NumPy array.")
        if not isinstance(actions_train, np.ndarray):
            raise ValueError("actions_train must be a NumPy array.")
        if states_train.shape != actions_train.shape:
            raise ValueError("states_train and actions_train must have the same shape.")
        if not np.issubdtype(states_train.dtype, np.integer):
            raise ValueError("states_train must contain integer values.")
        if not np.all((states_train >= 0) & (states_train <= max_state)):
            raise ValueError(f"states_train must contain values between 0 and {max_state}.")
        if not np.array_equal(actions_train, actions_train.astype(bool)):
            raise ValueError("actions_train must contain only 0 and 1.")
        
        # Step 1: Create a boolean mask where actions_train == 1
        mask_action1 = actions_train == 1
        
        # Step 2: Extract states where action == 1
        states_action1 = states_train[mask_action1]
        
        # Step 3: Count the number of times each state has action 1
        # minlength=max_state+1 ensures that all states from 0 to max_state are accounted for
        state_counts = np.bincount(states_action1, minlength=max_state)
        
        # Step 4: Find states with counts lower than the threshold
        states_less_than_threshold = np.where(state_counts < threshold)[0]
        
        # Step 5: Convert the result to a list
        states_less_than_threshold_list = states_less_than_threshold.tolist()
        
        return states_less_than_threshold_list
    
    def reduce_reward_of_action(self, R, states, amount=0.1, action=1):
        
        R[states, action] = R[states, action] - amount*self.reward_val
        return R
        
        
    def set_clustering(self, X_train, actions_train, metadata_train, X_val=None, actions_val=None, metadata_val=None, feature_weights=None):
        
        print("Clustering")
        if feature_weights is not None:
            # Convert feature_weights to a NumPy array
            feature_weights = np.array(feature_weights)

            # Validate feature_weights
            if feature_weights.ndim != 1:
                raise ValueError(f"feature_weights must be a 1-dimensional array, got shape {feature_weights.shape}.")
            if feature_weights.shape[0] != X_train.shape[1]:
                raise ValueError(
                    f"Length of feature_weights ({feature_weights.shape[0]}) does not match "
                    f"number of features in X_train ({X_train.shape[1]})."
                )
            print("Feature_weights set up:", feature_weights)
            self.feature_weights = feature_weights
        else:
            self.feature_weights = np.ones(X_train.shape[1])
        self.clusterer, states_train = self._cluster_states(X_train)

        
        # Create qldata3
        qldata3 = build_complete_record_sequences(
            metadata_train,
            states_train,
            actions_train,
            self.absorbing_states,
            self.rewards
        )
        
        ####### BUILD MODEL ########
        physpol, transitionr, R = compute_physician_policy(
            qldata3,
            self.n_states,
            self.n_actions,
            self.absorbing_states,
            reward_val=self.reward_val,
            transition_threshold=self.transition_threshold
        )
        self.R = R
        self.physician_policy = physpol
        self.transitionr = transitionr


    
    def compute_states(self, X,):
        X = np.asarray(X)
        # Scale each feature by its corresponding weight
        weighted_data = X * self.feature_weights
        return self.clusterer.predict(weighted_data)
        
    def compute_Q(self, X=None, states=None, actions=None):
        assert X is not None or states is not None, "At least one of states or X must not be None"
        if states is None:
            states = self.compute_states(X)

        if actions is not None:
            assert len(actions) == len(states)
            # Add a column of zeros at the end for the '-1' action (no action recorded)
            return np.hstack([self.Q, np.zeros(self.Q.shape[0]).reshape(-1, 1)])[states, actions]
        return self.Q[states]
    
    def compute_V(self, X):
        raise NotImplementedError

    def compute_probabilities(self, X=None, states=None, actions=None):
        assert X is not None or states is not None, "At least one of states or X must not be None"
        Q_vals = self.compute_Q(X=X, states=states)
        optimal_actions = np.argmax(Q_vals, axis=1).reshape(-1, 1)
        probs = np.where(np.arange(self.n_actions)[np.newaxis,:] == optimal_actions,
                         1 - self.soften_factor,
                         self.soften_factor / (self.n_actions - 1))
        if actions is not None:
            # Add a column of zeros at the end for the '-1' action (no action recorded)
            return np.hstack([probs, np.zeros(probs.shape[0]).reshape(-1, 1)])[np.arange(len(actions)), actions]
        return probs
    
    def compute_physician_probabilities(self, X=None, states=None, actions=None, soften=True):
        """
        Returns the probabilities for each state and action according to the
        physician policy, which is learned using the same state clustering model
        as the AI Clinician.
        
        If actions is provided, any values where the action is -1 will result in
        a zero probability.
        """
        assert X is not None or states is not None, "At least one of states or X must not be None"
        

        if soften:
            soft_physpol = self.physician_policy.copy() # behavior policy = clinicians'

            for i in range(self.n_cluster_states):
                ii = soft_physpol[i,:] == 0
                z = self.soften_factor / ii.sum()
                coef = soft_physpol[i,~ii].sum()
                soft_physpol[i, ii] = z
                soft_physpol[i, ~ii] = soft_physpol[i,~ii] * (1 - self.soften_factor / coef)
        else:
            soft_physpol = self.physician_policy

        if states is None:
            states = self.compute_states(X)
        probs = soft_physpol[states]
        if actions is not None:
            # Add a column of zeros at the end for the '-1' action (no action recorded)
            return np.hstack([probs, np.zeros(probs.shape[0]).reshape(-1, 1)])[np.arange(len(actions)), actions]
        return probs
        
    def save(self, filepath, metadata=None):
        """
        Saves the model as a pickle to the given filepath.
        """
        model_data = {
            'model_type': 'AIClinicianModel',
            'Qon': self.Q,
            'physician_policy': self.physician_policy,
            'T': self.transitionr,
            'R': self.R,
            'clusterer': self.clusterer,
            'feature_weights': self.feature_weights,
            'params': {
                'n_cluster_states': self.n_cluster_states,
                'n_actions': self.n_actions,
                'cluster_fit_fraction': self.cluster_fit_fraction,
                'n_cluster_init': self.n_cluster_init,
                'gamma': self.gamma,
                'random_state': self.random_state,
                'reward_val': self.reward_val,
                'transition_threshold': self.transition_threshold,
                'soften_factor': self.soften_factor
            }
        }
        if metadata is not None:
            model_data['metadata'] = metadata 
        with open(filepath, 'wb') as file:
            pickle.dump(model_data, file)
    
    @classmethod
    def load(cls, filepath):
        """
        Loads a model from a pickle file at the given filepath.
        """
        with open(filepath, 'rb') as file:
            model_data = pickle.load(file)
        assert model_data['model_type'] == 'AIClinicianModel', 'Invalid model type for AIClinicianModel'
        model = cls(**model_data['params'])
        model.metadata = model_data.get('metadata', None)
        model.clusterer = model_data['clusterer']
        model.physician_policy = model_data['physician_policy']
        model.Q = model_data['Qon']
        model.transitionr = model_data['T']
        model.R = model_data['R']
        model.feature_weights = model_data['feature_weights']
        print('Loaded_feature_weights', model_data['feature_weights'])
        return model


def evaluate_policy(qldata3, predicted_actions, physpol, n_cluster_states, soften_factor=0.01, gamma=0.99, num_iter_ql=6, num_iter_wis=750):
    """
    Evaluates a policy using two off-policy evaluation methods on the given set
    of patient trajectories.
    
    Parameters:
        qldata3: A dataframe containing complete record sequences of each
            patient
        predicted_actions: A vector of length S (number of states) containing
            the actions predicted by the policy to be evaluated.
        physpol: The actual physician policy, expressed as a matrix (S, A) of
            action probabilities given each state.
            
    Returns: Bootstrapped CIs for TD-learning and WIS.
    """

    qldata3 = build_record_sequences_with_policies(
        qldata3,
        predicted_actions,
        physpol,
        n_cluster_states,
        soften_factor=soften_factor
    )

    bootql = offpolicy_eval_tdlearning(qldata3, physpol, gamma, num_iter_ql, n_cluster_states)
    bootwis, _, _ = offpolicy_eval_wis(qldata3, gamma, num_iter_wis)

    return bootql, bootwis