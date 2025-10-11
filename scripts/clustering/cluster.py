import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
import seaborn as sns
import os
from itertools import combinations
from kmodes.kprototypes import KPrototypes
import gower
import prince 
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import scipy.stats as stats
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import re 

class ClusterEvaluator:
    """
    A class for performing, evaluating, and comparing clustering algorithms,
    including K-Means, Hierarchical, GMM, and K-Prototypes.
    """
    def __init__(self, X, X_pre_rd, k_range, gamma_values=None, random_state=42, linkage='ward', covariance_type='full'):
        """
        Initializes the ClusterEvaluator.

        Args:
            X (pd.DataFrame or np.ndarray): The dataset for numerical clustering, 
                                            often after dimensionality reduction.
            X_pre_rd (pd.DataFrame or np.ndarray): The original dataset before any 
                                                   dimensionality reduction, used for K-Prototypes.
            k_range (range or list): A range or list of integers for the number of clusters (k) to test.
            gamma_values (list, optional): A list of gamma values to test for the K-Prototypes model. 
                                           Defaults to [0.5].
            random_state (int, optional): The random state for reproducibility. Defaults to 42.
            linkage (str, optional): The linkage method for Hierarchical Clustering. Defaults to 'ward'.
            covariance_type (str, optional): The covariance type for GMM. Defaults to 'full'.
        """
        if isinstance(X, pd.DataFrame):
            self.X_df = X.copy()
        else:
            self.X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        self.X = self.X_df.values
        self._original_X_df = self.X_df.copy()

        if isinstance(X_pre_rd, pd.DataFrame):
            self.X_pre_rd_df = X_pre_rd.copy()
        else:
            self.X_pre_rd_df = pd.DataFrame(X_pre_rd, columns=[f'feature_{i}' for i in range(X_pre_rd.shape[1])])

        self.X_pre_rd = self.X_pre_rd_df.values
        self._original_pre_rd_df = self.X_pre_rd_df.copy()

        self.X_rd_df = self.X_df.copy()

        self.k_range = k_range
        # Store the list of gamma values to test. Default to a common value if None.
        self.gamma_values = gamma_values or [0.5]
        self.random_state = random_state
        self.linkage = linkage
        self.covariance_type = covariance_type
        
        self.results = {'kmeans': pd.DataFrame(), 'hierarchical': pd.DataFrame(), 'gmm': pd.DataFrame(), 'kprototypes': pd.DataFrame()}
        self.final_models = {'kmeans': None, 'hierarchical': None, 'gmm': None, 'kprototypes': None}

    # ==============================================================================
    # DATA PREPARATION
    # ==============================================================================
    def stratified_sample(self, strata_col, sample_frac, random_state=42):
        """
        Performs stratified sampling on the active (RD) dataset and creates a
        corresponding sample from the original (pre-RD) dataset.
        """
        if strata_col not in self._original_X_df.columns:
            print(f"Error: Stratification column '{strata_col}' not found.")
            return
        if not hasattr(self, '_original_pre_rd_df'):
            print("Error: Original pre-RD dataframe not found.")
            return

        print(f"Performing stratified sampling on '{strata_col}' with a {sample_frac:.2%} fraction...")
        
        df_sampled_rd = self._original_X_df.groupby(strata_col, group_keys=False).apply(
            lambda x: x.sample(frac=sample_frac, random_state=random_state), 
            include_groups=False
        )
        sampled_indices = df_sampled_rd.index
        df_sampled_original = self._original_pre_rd_df.loc[sampled_indices]

        original_rows = len(self._original_X_df)
        self.X_df = df_sampled_rd
        self.X = self.X_df.values
        self.X_df_original_sampled = df_sampled_original
        new_rows = len(self.X_df)
        
        print(f"Sampling complete. Active dataset reduced from {original_rows} to {new_rows} rows.")
        return self.X_df
    
    def scale_features(self, columns_to_scale=None, dataset_to_scale='rd', strata_col_to_exclude=None):
        """
        Scales numerical feature columns in the specified dataset(s), ensuring the
        stratification column is never scaled.

        Args:
            columns_to_scale (list, optional): Specific columns to scale. If None, all numerical
                                               columns in the target dataset(s) are scaled.
            dataset_to_scale (str, optional): Which dataset to scale. Options are:
                                              'rd' (reduced-dimension data, default),
                                              'pre_rd' (original pre-reduction data), 'both'.
            strata_col_to_exclude (str, optional): The name of the stratification column to
                                                   exclude from scaling.
        """
        # A helper function to remove the stratification column from a list of columns
        def _exclude_strata_col(cols_list, strata_col):
            if strata_col and strata_col in cols_list:
                print(f"ⓘ Excluding stratification column '{strata_col}' from scaling.")
                cols_list.remove(strata_col)
            return cols_list

        if dataset_to_scale in ['rd', 'both']:
            print("\n--- Scaling Reduced-Dimension (RD) Dataset ---")
            
            # Determine the initial list of columns to scale
            if columns_to_scale is None:
                cols_for_rd = self.X_df.select_dtypes(include=np.number).columns.tolist()
            else:
                # Make a copy to avoid modifying the original list
                cols_for_rd = list(columns_to_scale)
            
            # Exclude the stratification column before scaling
            final_cols_for_rd = _exclude_strata_col(cols_for_rd, strata_col_to_exclude)
            
            scaled_rd_df = self._perform_scaling(self.X_df, 'scaler_rd', final_cols_for_rd)
            if scaled_rd_df is not None:
                self.X_df = scaled_rd_df
                self.X = self.X_df.values
                print("✅ RD dataset scaling complete. self.X_df and self.X have been updated.")

        if dataset_to_scale in ['pre_rd', 'both']:
            print("\n--- Scaling Original Pre-Reduction (pre-RD) Dataset ---")
            
            # Determine the initial list of columns to scale
            numeric_cols_pre_rd = self.X_pre_rd_df.select_dtypes(include=np.number).columns.tolist()
            if columns_to_scale is None:
                cols_for_pre_rd = numeric_cols_pre_rd
            else:
                cols_for_pre_rd = [col for col in columns_to_scale if col in numeric_cols_pre_rd]
            
            # Exclude the stratification column before scaling
            final_cols_for_pre_rd = _exclude_strata_col(cols_for_pre_rd, strata_col_to_exclude)

            scaled_pre_rd_df = self._perform_scaling(self.X_pre_rd_df, 'scaler_pre_rd', final_cols_for_pre_rd)
            if scaled_pre_rd_df is not None:
                self.X_pre_rd_df = scaled_pre_rd_df
                print("✅ Pre-RD dataset scaling complete. self.X_pre_rd_df has been updated.")

    def _perform_scaling(self, df_to_scale, scaler_attr_name, columns_to_scale):
        """Helper function to perform StandardScaler on a dataframe's numeric columns."""
        scaler = StandardScaler()
        # Store the specific scaler instance (e.g., self.scaler_rd or self.scaler_pre_rd)
        setattr(self, scaler_attr_name, scaler)

        df_copy = df_to_scale.copy()
        
        if not columns_to_scale:
            print("No valid numeric columns specified or found to scale.")
            return df_copy # Return original df if no columns to scale

        # Final check for existence in the dataframe
        actual_cols_to_scale = [col for col in columns_to_scale if col in df_copy.columns]
        missing_cols = set(columns_to_scale) - set(actual_cols_to_scale)
        if missing_cols:
            print(f"Warning: Columns not found and will be ignored: {list(missing_cols)}")

        if not actual_cols_to_scale:
            print("No valid columns left to scale.")
            return df_copy

        print(f"Scaling {len(actual_cols_to_scale)} features: {actual_cols_to_scale}...")
        df_copy[actual_cols_to_scale] = scaler.fit_transform(df_copy[actual_cols_to_scale])
        
        return df_copy

    # ==============================================================================
    # CLUSTERING ALGORITHMS
    # ==============================================================================
    def run_and_evaluate(self, algorithms=['kmeans', 'hierarchical', 'gmm', 'kprototypes'], n_samples=1, strata_col=None, sample_frac=None, kproto_features=None):
        """
        Runs and evaluates specified clustering algorithms. Can perform cross-sampling.
        
        Args:
            # ... other args
            kproto_features (list, optional): A list of column names to use exclusively for the K-Prototypes model.
                                            If None, all features from X_pre_rd are used. Defaults to None.
        """
        if n_samples > 1:
            if strata_col is None or sample_frac is None:
                raise ValueError("For n_samples > 1, 'strata_col' and 'sample_frac' must be provided.")
            self._run_cross_sample_evaluation(algorithms, n_samples, strata_col, sample_frac, kproto_features)
        else:
            self._run_single_evaluation(algorithms, kproto_features)

    def _run_single_evaluation(self, algorithms, kproto_features=None):
        """Runs evaluation on the single, current dataset."""
        if 'kmeans' in algorithms:
            self.results['kmeans'] = self._evaluate_kmeans()
        if 'hierarchical' in algorithms:
            self.results['hierarchical'] = self._evaluate_hierarchical()
        if 'gmm' in algorithms:
            self.results['gmm'] = self._evaluate_gmm()
        if 'kprototypes' in algorithms:
            # Pass the features list here
            self.results['kprototypes'] = self._evaluate_kprototypes(features_to_use=kproto_features)
        
        print("\nEvaluation complete for all specified algorithms.")

    def _run_cross_sample_evaluation(self, algorithms, n_samples, strata_col, sample_frac, kproto_features=None):
        """Runs evaluation across multiple stratified samples and aggregates the results."""
        all_results = {algo: [] for algo in algorithms}

        print(f"\nRunning cross-sample evaluation with {n_samples} samples...")
        for i in range(n_samples):
            print(f"  Iteration {i + 1}/{n_samples}...")
            # Sample the reduced-dimension data to get indices
            sample_df_rd = self._original_X_df.groupby(strata_col, group_keys=False).apply(
                lambda x: x.sample(frac=sample_frac, random_state=self.random_state + i),
                include_groups=False
            )
            X_sample_rd = sample_df_rd.values
            
            # Get corresponding pre-reduction data for K-Prototypes
            sample_df_pre_rd = self._original_pre_rd_df.loc[sample_df_rd.index]
            
            if 'kmeans' in algorithms:
                all_results['kmeans'].append(self._evaluate_kmeans(X_data=X_sample_rd))
            if 'hierarchical' in algorithms:
                all_results['hierarchical'].append(self._evaluate_hierarchical(X_data=X_sample_rd))
            if 'gmm' in algorithms:
                all_results['gmm'].append(self._evaluate_gmm(X_data=X_sample_rd))
            if 'kprototypes' in algorithms:
                all_results['kprototypes'].append(self._evaluate_kprototypes(X_data_df=sample_df_pre_rd, features_to_use=kproto_features))
        
        print("\nAggregating results...")
        for algo, results_list in all_results.items():
            if results_list:
                concatenated_df = pd.concat(results_list)
                # Group by index (which can be a MultiIndex for k-prototypes)
                if all(name is None for name in concatenated_df.index.names):
                    # This handles unnamed indexes (e.g., from KMeans, GMM). Group by the index values directly.
                    mean_df = concatenated_df.groupby(concatenated_df.index).mean()
                    std_df = concatenated_df.groupby(concatenated_df.index).std()
                else:
                    # This handles named indexes (e.g., from KPrototypes). Group by level names to preserve them.
                    mean_df = concatenated_df.groupby(level=concatenated_df.index.names).mean()
                    std_df = concatenated_df.groupby(level=concatenated_df.index.names).std()
                final_df = mean_df.join(std_df, lsuffix='_mean', rsuffix='_std')
                self.results[algo] = final_df
        
        print("Cross-sample evaluation complete.")

    def _evaluate_kprototypes(self, X_data_df=None, features_to_use=None):
        """Evaluates K-Prototypes for different gamma values using Gower distance for metrics."""
        if X_data_df is None:
            X_data_df = self._original_pre_rd_df.loc[self.X_df.index]

        if features_to_use:
            missing_features = set(features_to_use) - set(X_data_df.columns)
            if missing_features:
                print(f"⚠️ Warning: The following features for K-Prototypes were not found and will be ignored: {list(missing_features)}")
            
            valid_features = [f for f in features_to_use if f in X_data_df.columns]

            if not valid_features:
                print("⛔ Error: No valid features found for K-Prototypes. Aborting evaluation.")
                return pd.DataFrame()

            X_data_df = X_data_df[valid_features]
            print(f"📊 Running K-Prototypes on a subset of {len(valid_features)} features.")

        categorical_features_indices = [i for i, col in enumerate(X_data_df.columns) 
                                        if X_data_df[col].dtype == 'object' or X_data_df[col].dtype.name == 'category']
        
        print("  Calculating Gower distance matrix on original data for metrics...")
        gower_dist_matrix = gower.gower_matrix(X_data_df)

        print("  Scaling numerical features for model training...")
        X_scaled_for_model = X_data_df.copy()
        numerical_cols = X_scaled_for_model.select_dtypes(include=np.number).columns.tolist()
        
        if numerical_cols:
            scaler = StandardScaler()
            X_scaled_for_model[numerical_cols] = scaler.fit_transform(X_scaled_for_model[numerical_cols])
        
        X_data_np_scaled = X_scaled_for_model.values

        if not self.gamma_values:
            print("Warning: No gamma values provided for K-Prototypes evaluation.")
            return pd.DataFrame()

        print(f"Running K-Prototypes for k in {list(self.k_range)} and gamma in {self.gamma_values}...")
        results_list = []
        
        for gamma in self.gamma_values:
            print(f"  Testing gamma = {gamma}...")
            for k in self.k_range:
                model = KPrototypes(n_clusters=k, init='Cao', gamma=gamma, 
                                    random_state=self.random_state, n_init=5, verbose=0)
                
                labels = model.fit_predict(X_data_np_scaled, categorical=categorical_features_indices)
                
                if len(np.unique(labels)) > 1:
                    twss_cost = model.cost_
                    sil_score = silhouette_score(gower_dist_matrix, labels, metric='precomputed')
                    dbi_score = davies_bouldin_score(gower_dist_matrix, labels)
                else:
                    twss_cost, sil_score, dbi_score = np.nan, -1, np.nan

                results_list.append({
                    'gamma': gamma,
                    'k': k,
                    'TWSS (Cost)': twss_cost,
                    'Silhouette Score (Gower)': sil_score,
                    'Davies-Bouldin Score (Gower)': dbi_score 
                })

        if not results_list:
            return pd.DataFrame()
            
        results_df = pd.DataFrame(results_list)
        return results_df.set_index(['gamma', 'k'])

    def _evaluate_kmeans(self, X_data=None):
        if X_data is None: X_data = self.X
        print(f"Running K-Means for k in {list(self.k_range)}...")
        results_k = {}
        for k in self.k_range:
            model = KMeans(n_clusters=k, n_init='auto', random_state=self.random_state)
            labels = model.fit_predict(X_data)
            results_k[k] = {
                'TWSS': model.inertia_,
                'Silhouette Score': silhouette_score(X_data, labels),
                'DBI Score': davies_bouldin_score(X_data, labels)
            }
        return pd.DataFrame.from_dict(results_k, orient='index')

    def _calculate_twss(self, X, labels):
        """Calculates the total within-cluster sum of squares."""
        twss = 0
        unique_labels = np.unique(labels)
        for label in unique_labels:
            # This line now works correctly because 'X' is the 2D data
            # and 'labels' is the 1D array for filtering.
            cluster_points = X[labels == label]
            if len(cluster_points) > 0:
                centroid = cluster_points.mean(axis=0)
                twss += np.sum((cluster_points - centroid) ** 2)
        return twss

    def _calculate_dunn_index(self, distance_matrix, labels):
        """
        Calculates the Dunn Index for a given clustering.
        The Dunn Index is the ratio of the minimum inter-cluster distance to the
        maximum intra-cluster distance. A higher Dunn Index indicates better clustering.

        Args:
            distance_matrix (np.ndarray): A square matrix of pairwise distances.
            labels (np.ndarray): The cluster labels for each data point.

        Returns:
            float: The Dunn Index score. Returns 0 if it cannot be computed.
        """
        unique_labels = np.unique(labels)
        # The index cannot be computed with less than 2 clusters.
        if len(unique_labels) < 2:
            return 0.0

        # --- Calculate maximum intra-cluster distance (diameter) ---
        max_intra_cluster_dist = 0.0
        for label in unique_labels:
            # Get indices of points in the current cluster
            cluster_indices = np.where(labels == label)[0]
            # Skip if cluster has only one point
            if len(cluster_indices) < 2:
                continue
            # Extract the sub-matrix of distances for the current cluster
            cluster_distances = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
            # The diameter is the maximum distance within the cluster
            diameter = np.max(cluster_distances)
            if diameter > max_intra_cluster_dist:
                max_intra_cluster_dist = diameter

        # Avoid division by zero if all clusters are single points or distances are zero.
        if max_intra_cluster_dist == 0:
            return 0.0

        # --- Calculate minimum inter-cluster distance ---
        min_inter_cluster_dist = np.inf
        # Iterate through all unique pairs of clusters
        for label1, label2 in combinations(unique_labels, 2):
            indices1 = np.where(labels == label1)[0]
            indices2 = np.where(labels == label2)[0]
            
            # Extract the sub-matrix of distances between the two clusters
            inter_cluster_distances = distance_matrix[np.ix_(indices1, indices2)]
            
            # Find the minimum distance in this sub-matrix
            min_dist = np.min(inter_cluster_distances)
            if min_dist < min_inter_cluster_dist:
                min_inter_cluster_dist = min_dist
                
        return min_inter_cluster_dist / max_intra_cluster_dist

    def _evaluate_hierarchical(self, X_data=None):
        if X_data is None: X_data = self.X
        print(f"Running Hierarchical Clustering (linkage='{self.linkage}') for k in {list(self.k_range)}...")
        results_h = {}
        for k in self.k_range:
            model = AgglomerativeClustering(n_clusters=k, linkage=self.linkage)
            labels = model.fit_predict(X_data)
            results_h[k] = {
                'TWSS': self._calculate_twss(labels, X_data),
                'Silhouette Score': silhouette_score(X_data, labels),
                'DBI Score': davies_bouldin_score(X_data, labels)
            }
        return pd.DataFrame.from_dict(results_h, orient='index')
        
    def _evaluate_gmm(self, X_data=None):
        if X_data is None: X_data = self.X
        print(f"Running GMM (covariance_type='{self.covariance_type}') for k in {list(self.k_range)}...")
        results_g = {}
        for k in self.k_range:
            model = GaussianMixture(n_components=k, covariance_type=self.covariance_type, random_state=self.random_state, init_params='k-means++')
            labels = model.fit_predict(X_data)
            results_g[k] = {
                'TWSS': self._calculate_twss(labels, X_data),
                'Silhouette Score': silhouette_score(X_data, labels),
                'DBI Score': davies_bouldin_score(X_data, labels),
                'BIC': model.bic(X_data),
                'AIC': model.aic(X_data)
            }
        return pd.DataFrame.from_dict(results_g, orient='index')

    def run_final_models(self, hyperparams):
        """
        Trains final models using a dictionary of specified hyperparameters.

        Args:
            hyperparams (dict): A dictionary where keys are algorithm names and
                                values are another dictionary of parameters.
                                Example:
                                {
                                    'kmeans': {'n_clusters': 3},
                                    'hierarchical': {'n_clusters': 3, 'linkage': 'ward'},
                                    'gmm': {'n_components': 3, 'covariance_type': 'full'},
                                    'kprototypes': {'n_clusters': 3, 'gamma': 0.25, 'features': ['col1', 'col2']}
                                }
        """
        # --- Helper to safely get parameters or use class defaults ---
        def _get_param(algo_name, param_name, default_value):
            return hyperparams.get(algo_name, {}).get(param_name, default_value)

        # --- Get data for numerical models ---
        feature_df_rd = self.X_df[[col for col in self.X_df.columns if not col.startswith('cluster')]]
        
        # --- K-MEANS ---
        if 'kmeans' in hyperparams:
            k = _get_param('kmeans', 'n_clusters', None)
            if k is None:
                print("\nSkipping K-Means: 'n_clusters' not provided in hyperparams.")
            else:
                print(f"\nRunning final K-Means model with k={k}...")
                model = KMeans(n_clusters=k, n_init='auto', random_state=self.random_state)
                self.X_df['cluster_kmeans'] = model.fit_predict(feature_df_rd)
                self.final_models['kmeans'] = model

        # --- HIERARCHICAL ---
        if 'hierarchical' in hyperparams:
            k = _get_param('hierarchical', 'n_clusters', None)
            linkage = _get_param('hierarchical', 'linkage', self.linkage)
            if k is None:
                print("\nSkipping Hierarchical: 'n_clusters' not provided in hyperparams.")
            else:
                print(f"\nRunning final Hierarchical model with k={k} and linkage='{linkage}'...")
                model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
                self.X_df['cluster_hierarchical'] = model.fit_predict(feature_df_rd)
                self.final_models['hierarchical'] = model
            
        # --- GMM ---
        if 'gmm' in hyperparams:
            k = _get_param('gmm', 'n_components', None)
            covariance_type = _get_param('gmm', 'covariance_type', self.covariance_type)
            if k is None:
                print("\nSkipping GMM: 'n_components' not provided in hyperparams.")
            else:
                print(f"\nRunning final GMM with k={k} and covariance_type='{covariance_type}'...")
                model = GaussianMixture(n_components=k, covariance_type=covariance_type, random_state=self.random_state, init_params='k-means++')
                self.X_df['cluster_gmm'] = model.fit_predict(feature_df_rd)
                self.final_models['gmm'] = model

        # --- K-PROTOTYPES ---
        if 'kprototypes' in hyperparams:
            k = _get_param('kprototypes', 'n_clusters', None)
            gamma = _get_param('kprototypes', 'gamma', self.gamma_values[0])
            features = _get_param('kprototypes', 'features', None)

            if k is None:
                print("\nSkipping K-Prototypes: 'n_clusters' not provided in hyperparams.")
            else:
                print(f"\nRunning final K-Prototypes model with k={k}, gamma={gamma}...")
                data_pre_rd = self.X_pre_rd_df.loc[self.X_df.index]
                
                if features:
                    valid_features = [f for f in features if f in data_pre_rd.columns]
                    data_pre_rd = data_pre_rd[valid_features]
                    print(f"--> Using a subset of {len(valid_features)} features.")
                
                categorical_indices = [i for i, col in enumerate(data_pre_rd.columns) if data_pre_rd[col].dtype in ['object', 'category']]
                
                model = KPrototypes(n_clusters=k, init='Cao', gamma=gamma, random_state=self.random_state, n_init=10)
                self.X_df['cluster_kprototypes'] = model.fit_predict(data_pre_rd.values, categorical=categorical_indices)
                self.final_models['kprototypes'] = model
            
        print("\n✅ Final models are trained and cluster labels are added to the DataFrame.")
        return self.X_df

    # ==============================================================================
    # VISUALIZATION & METRICS
    # ==============================================================================
    def plot_numeric_evaluation_metrics(self):
        """Plots evaluation metrics for the purely numeric clustering algorithms."""
        numeric_results = {k: v for k, v in self.results.items() if k != 'kprototypes' and not v.empty}
        if not numeric_results:
            print("No results found for numeric algorithms (K-Means, Hierarchical, GMM).")
            return

        metrics = ['TWSS', 'Silhouette Score', 'DBI Score']
        fig, axs = plt.subplots(1, 3, figsize=(20, 6))
        
        for ax, metric in zip(axs, metrics):
            for algo_name, df in numeric_results.items():
                label = {'kmeans': 'K-Means', 'hierarchical': f'Hierarchical ({self.linkage})', 'gmm': f'GMM ({self.covariance_type})'}.get(algo_name)
                self._plot_metric(ax, df, metric, label)

            ax.set_title(metric)
            ax.grid(True)
            ax.legend()
        
        fig.suptitle('Comparação de métricas de validade interna (espaço reduzido)', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_kprototypes_evaluation_metrics(self):
        """Plots evaluation metrics for K-Prototypes, comparing different gamma values."""
        kproto_results = self.results.get('kprototypes')
        if kproto_results is None or kproto_results.empty:
            print("No results found for K-Prototypes. Please run `run_and_evaluate()` with 'kprototypes'.")
            return

        # The new metrics to be plotted. TWSS (lower is better), Silhouette (higher), Davies-Bouldin Score (lower).
        metrics = ['TWSS (Cost)', 'Silhouette Score (Gower)', 'Davies-Bouldin Score (Gower)']
        fig, axs = plt.subplots(1, 3, figsize=(22, 6))
        
        # Check if the results are from a cross-sample run (mean/std columns exist)
        is_cross_sample = any('_mean' in col for col in kproto_results.columns)

        for ax, metric in zip(axs, metrics):
            # The base name of the metric (without _mean or _std)
            base_metric = metric
            
            # In a cross-sample run, we plot the mean and std dev
            if is_cross_sample:
                mean_col = f'{base_metric}_mean'
                std_col = f'{base_metric}_std'
                
                # Check if the required columns exist
                if mean_col not in kproto_results.columns:
                    print(f"Warning: Mean column '{mean_col}' not found for K-Prototypes. Skipping plot for this metric.")
                    continue

                # Group by gamma value to plot separate lines
                # The index is (gamma, k), so level 0 is gamma
                for gamma, group_df in kproto_results.groupby(level='gamma'):
                    k_values = group_df.index.get_level_values('k')
                    means = group_df[mean_col]
                    stds = group_df[std_col].fillna(0) if std_col in group_df else 0
                    
                    ax.errorbar(k_values, means, yerr=stds, fmt='-', capsize=4, markersize=5, label=f'gamma={gamma}')
            
            # In a single run, we just plot the direct score
            else:
                if base_metric not in kproto_results.columns:
                    print(f"Warning: Metric column '{base_metric}' not found for K-Prototypes. Skipping plot.")
                    continue

                # Group by gamma value to plot a line for each
                for gamma, group_df in kproto_results.groupby(level='gamma'):
                    # Extract k values from the second level of the MultiIndex
                    k_values = group_df.index.get_level_values('k')
                    scores = group_df[base_metric]
                    ax.plot(k_values, scores, '-', markersize=5, label=f'gamma={gamma}')

            ax.set_title(base_metric)
            ax.set_xlabel('Número de clusters (k)')
            ax.set_ylabel('Score')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
        fig.suptitle('Comparação de métricas de validade interna (espaço original)', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def _plot_metric(self, ax, df, metric, label):
        """Helper function to plot a single metric, handling cross-sample error bars."""
        mean_col, std_col = f'{metric}_mean', f'{metric}_std'
        if mean_col in df.columns and std_col in df.columns:
            means, stds = df[mean_col], df[std_col].fillna(0)
            ax.errorbar(df.index, means, yerr=stds, fmt='-', capsize=4, markersize=4, label=label)
        elif metric in df.columns:
            ax.plot(df.index, df[metric], '-', markersize=4, label=label)
        ax.set_xlabel('Número de clusters (k)')
        ax.set_ylabel('Score')

    def plot_feature_distributions(self, palette='viridis'):
        """
        Creates violin plots to compare the distribution of each feature across clusters
        for each of the final trained models.
        """
        # 1. Check if any final models have been trained
        cluster_cols = [col for col in self.X_df.columns if col.startswith('cluster_')]
        if not cluster_cols:
            print("Please run `run_final_models()` before plotting feature distributions.")
            return

        # 2. Identify the feature columns to plot
        feature_columns = [col for col in self.X_df.columns if not col.startswith('cluster_')]
        
        print("\n--- Generating Feature Distribution Plots ---")
        
        # 3. Loop through each feature and create a dedicated plot
        for feature in feature_columns:
            # Create a new figure for each feature
            num_plots = len(cluster_cols)
            fig, axs = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6), sharey=True)
            if num_plots == 1:
                axs = [axs] # Make it iterable
            fig.suptitle(f'Distribution of "{feature}" Across Clusters', fontsize=16)

            plot_idx = 0
            if 'cluster_kmeans' in self.X_df.columns:
                sns.violinplot(ax=axs[plot_idx], data=self.X_df, x='cluster_kmeans', y=feature, hue='cluster_kmeans', palette=palette, legend=False)
                axs[plot_idx].set_title(f'K-Means (k={self.final_models["kmeans"].n_clusters})')
                plot_idx += 1

            if 'cluster_hierarchical' in self.X_df.columns:
                sns.violinplot(ax=axs[plot_idx], data=self.X_df, x='cluster_hierarchical', y=feature, hue='cluster_hierarchical', palette=palette, legend=False)
                linkage_method = self.final_models["hierarchical"].linkage
                axs[plot_idx].set_title(f'Hierarchical ({linkage_method}) (k={self.final_models["hierarchical"].n_clusters})')
                plot_idx += 1

            if 'cluster_gmm' in self.X_df.columns:
                sns.violinplot(ax=axs[plot_idx], data=self.X_df, x='cluster_gmm', y=feature, hue='cluster_gmm', palette=palette, legend=False)
                cov_type = self.final_models["gmm"].covariance_type
                axs[plot_idx].set_title(f'GMM ({cov_type}) (k={self.final_models["gmm"].n_components})')
                plot_idx += 1
            
            if 'cluster_kprototypes' in self.X_df.columns:
                # K-Prototypes involves categorical data, which isn't suitable for violin plots.
                # A count plot is more appropriate for showing distributions.
                print(f"Note: Violin plot for '{feature}' is only applicable if it is a numerical feature.")


            # Apply common labels
            for ax in axs:
                ax.set_xlabel('Cluster ID')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

    def plot_cluster_sizes(self):
        """
        Creates bar charts comparing the size (number of points) of each cluster
        for each of the final trained algorithms.
        """
        if all(model is None for model in self.final_models.values()):
            print("Please run `run_final_models()` before plotting cluster sizes.")
            return

        # Identify which algorithms have been run
        algorithms_run = [algo for algo, model in self.final_models.items() if model is not None]
        if not algorithms_run:
            print("No final models have been run. Please run `run_final_models()` first.")
            return

        # Create subplots based on the number of models run
        num_models = len(algorithms_run)
        fig, axs = plt.subplots(1, num_models, figsize=(5 * num_models, 6), sharey=True)
        if num_models == 1:
            axs = [axs] # Make it iterable if there's only one plot
        fig.suptitle('Comparação dos tamanhos dos clusters', fontsize=16)

        # Define properties for each algorithm
        algo_properties = {
            'kmeans': {'palette': 'viridis'},
            'hierarchical': {'palette': 'plasma'},
            'gmm': {'palette': 'magma'},
            'kprototypes': {'palette': 'cividis'}
        }

        for i, algorithm in enumerate(algorithms_run):
            ax = axs[i]
            cluster_col = f'cluster_{algorithm}'
            
            # Get model-specific details
            model = self.final_models[algorithm]
            if algorithm in ['kmeans', 'hierarchical']:
                k = model.n_clusters
                title = f'{algorithm.capitalize()} (k={k})'
            elif algorithm == 'gmm':
                k = model.n_components
                title = f'GMM (k={k})'
            elif algorithm == 'kprototypes':
                k = model.n_clusters
                title = f'K-Prototypes (k={k})'

            # Calculate and plot sizes
            sizes = self.X_df[cluster_col].value_counts().sort_index()
            sns.barplot(ax=ax, x=sizes.index, y=sizes.values, palette=algo_properties[algorithm]['palette'], hue=sizes.index, legend=False)
            ax.set_title(title)
            ax.set_xlabel('Número do cluster')
            if i == 0:
                ax.set_ylabel('Tamanho dos clusters')

            ax.bar_label(ax.containers[0])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_cluster_matrix(self, algorithm, palette='viridis'):
        """
        Creates a pair plot for numeric algorithms or a 2D FAMD scatter plot for K-Prototypes.

        Args:
            algorithm (str): The algorithm to plot ('kmeans', 'hierarchical', 'gmm', or 'kprototypes').
            palette (str, optional): The color palette to use for the plots.
        """
        cluster_col = f'cluster_{algorithm}'
        model = self.final_models.get(algorithm)

        if model is None or cluster_col not in self.X_df.columns:
            print(f"Please run `run_final_models()` for '{algorithm}' before plotting.")
            return

        print(f"\n--- Generating Cluster Visualization for {algorithm.upper()} ---")

        if algorithm == 'kprototypes':
            # --- FAMD Scatter Plot for K-Prototypes ---
            data_to_plot = self._original_pre_rd_df.loc[self.X_df.index]
            
            print("  Running FAMD to get principal components for visualization...")
            try:
                famd = prince.FAMD(n_components=2, n_iter=3, random_state=self.random_state)
                components = famd.fit_transform(data_to_plot)
                components.columns = ['Component 1', 'Component 2']
            except Exception as e:
                print(f"  Could not perform FAMD. Error: {e}")
                print("  Please ensure the 'prince' library is installed (`pip install prince`).")
                return

            components[cluster_col] = self.X_df[cluster_col].values
            
            plt.figure(figsize=(10, 8))
            sns.scatterplot(data=components, x='Component 1', y='Component 2', hue=cluster_col, palette=palette)
            
            k = self.final_models[algorithm].n_clusters
            plt.title(f'Clusters do modelo k-prototypes (k={k})', fontsize=16)
            plt.xlabel('Componente 1 (FAMD)')
            plt.ylabel('Componente 2 (FAMD)')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(title='Cluster')
            plt.show()

        else:
            # --- Pair Plot for Numeric Algorithms (Existing Logic) ---
            feature_columns = [col for col in self.X_df.columns if not col.startswith('cluster')]
            plot_df_features = self.X_df[feature_columns + [cluster_col]]

            if len(feature_columns) < 2:
                print("At least two numeric features are required for a pair plot.")
                return
            
            g = sns.pairplot(plot_df_features, hue=cluster_col, palette=palette, corner=True, vars=feature_columns)
            
            if algorithm in ['kmeans', 'hierarchical']:
                title = f'{algorithm.capitalize()} clusters (k={model.n_clusters})'
            elif algorithm == 'gmm':
                title = f'GMM clusters (k={model.n_components})'
            
            g.fig.suptitle(title, y=1.02, fontsize=16)
            plt.show()

    # ==============================================================================
    # HIERARCHICAL-SPECIFIC METHODS
    # ==============================================================================
    def evaluate_linkage_methods(self, linkage_methods=['ward', 'complete', 'average', 'single'], n_samples=1, strata_col=None, sample_frac=None):
        """
        Evaluates and plots different linkage methods for Hierarchical Clustering.
        Can perform cross-sampling to evaluate stability.

        Args:
            linkage_methods (list): A list of linkage methods to test.
            n_samples (int): If 1, runs on the current self.X data. If > 1, performs
                             cross-sampling and aggregates results.
            strata_col (str): The column to stratify on. Required if n_samples > 1.
            sample_frac (float): The fraction of data to sample. Required if n_samples > 1.
            
        Returns:
            tuple: A tuple containing:
                - dict: A dictionary with aggregated metric scores for each linkage method.
                - pd.DataFrame: A DataFrame with the size of each cluster for every run.
        """
        if n_samples > 1 and (strata_col is None or sample_frac is None):
            raise ValueError("For n_samples > 1, 'strata_col' and 'sample_frac' must be provided.")

        print(f"\n--- Evaluating Linkage Methods for k in {list(self.k_range)} ---")
        
        final_aggregated_results = {}
        cluster_sizes_data = [] # NEW: List to store cluster size information

        if n_samples > 1:
            print(f"Running cross-sample evaluation with {n_samples} samples...")
            all_run_results = {linkage: [] for linkage in linkage_methods}

            for i in range(n_samples):
                print(f"  Iteration {i + 1}/{n_samples}...")
                sample_df = self._original_X_df.groupby(strata_col, group_keys=False).apply(
                    lambda x: x.sample(frac=sample_frac, random_state=self.random_state + i),
                    include_groups=False
                )
                X_sample = sample_df.values

                for linkage in linkage_methods:
                    current_results = {}
                    for k in self.k_range:
                        if linkage == 'ward' and X_sample.shape[1] < 2:
                            continue # Ward linkage requires multi-dimensional data
                        model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
                        labels = model.fit_predict(X_sample)
                        
                        if len(set(labels)) > 1:
                            twss = self._calculate_twss(X_sample, labels) 
                            silhouette = silhouette_score(X_sample, labels)
                            dbi = davies_bouldin_score(X_sample, labels)
                            twss = self._calculate_twss(X_sample, labels) 
                        else:
                            silhouette, dbi, twss = -1, -1, -1

                        # NEW: Add TWSS to the results
                        current_results[k] = {'TWSS': twss, 'Silhouette Score': silhouette, 'DBI Score': dbi}

                        # NEW: Capture cluster sizes
                        unique_labels, counts = np.unique(labels, return_counts=True)
                        for label, size in zip(unique_labels, counts):
                            cluster_sizes_data.append({
                                'Linkage': linkage,
                                'k': k,
                                'Sample': i + 1,
                                'Cluster': label,
                                'Size': size
                            })
                    all_run_results[linkage].append(pd.DataFrame.from_dict(current_results, orient='index'))
            
            print("\nAggregating results...")
            for linkage, results_list in all_run_results.items():
                if results_list:
                    concatenated_df = pd.concat(results_list)
                    mean_df = concatenated_df.groupby(concatenated_df.index).mean()
                    std_df = concatenated_df.groupby(concatenated_df.index).std()
                    final_aggregated_results[linkage] = mean_df.join(std_df, lsuffix='_mean', rsuffix='_std')
        else:
            for linkage in linkage_methods:
                print(f"Testing linkage: '{linkage}'...")
                current_results = {}
                for k in self.k_range:
                    model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
                    labels = model.fit_predict(self.X)
                    
                    if len(set(labels)) > 1:
                        twss = self._calculate_twss(self.X, labels) 
                        silhouette = silhouette_score(self.X, labels)
                        dbi = davies_bouldin_score(self.X, labels)
                    else:
                        silhouette, dbi, twss = -1, -1, -1
                    
                    current_results[k] = {'TWSS': twss, 'Silhouette Score': silhouette, 'DBI Score': dbi}
                    
                    unique_labels, counts = np.unique(labels, return_counts=True)
                    for label, size in zip(unique_labels, counts):
                        cluster_sizes_data.append({
                            'Linkage': linkage,
                            'k': k,
                            'Sample': 1,
                            'Cluster': label,
                            'Size': size
                        })
                final_aggregated_results[linkage] = pd.DataFrame.from_dict(current_results, orient='index')
        
        # Convert cluster sizes to a DataFrame
        cluster_sizes_df = pd.DataFrame(cluster_sizes_data)

        # --- Plotting ---
        fig, axs = plt.subplots(1, 3, figsize=(24, 7), sharex=True)
        fig.suptitle('Hierarchical Linkage Method Comparison', fontsize=16)
        
        # NEW: Added 'TWSS' to the list of metrics to plot
        metrics = ['TWSS', 'Silhouette Score', 'DBI Score']
        
        for ax, metric in zip(axs, metrics):
            for linkage_name, results_df in final_aggregated_results.items():
                if not results_df.empty:
                    self._plot_metric(ax, results_df, metric, linkage_name)
            
            ax.set_title(metric)
            ax.set_xlabel('Number of Clusters (k)')
            ax.grid(True)
            ax.legend()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # Return both the aggregated results and the cluster size DataFrame
        return final_aggregated_results, cluster_sizes_df

    def plot_dendrogram(self, linkage=None, k_cut=None):
        linkage = linkage or self.linkage
        print(f"\n--- Generating Dendrogram with '{linkage}' linkage ---")
        Z = sch.linkage(self.X, method=linkage)
        plt.figure(figsize=(15, 7))
        if k_cut is not None and k_cut > 1:
            plt.title(f'Dendrograma com {k_cut} clusters (ligação: {linkage})')
            distances = sorted(Z[:, 2], reverse=True)
            threshold = distances[k_cut - 2]
            sch.dendrogram(Z, color_threshold=threshold)
            plt.axhline(y=threshold, c='grey', lw=1.5, linestyle='--')
        else:
            plt.title(f'Dendrograma do modelo hierárquico aglomerativo (ligação: {linkage})')
            sch.dendrogram(Z)
        plt.xlabel('Observações')
        plt.ylabel('Distância')
        plt.grid(axis='y')
        plt.show()
    
    # ==============================================================================
    # GMM-SPECIFIC METHODS
    # ==============================================================================
    def evaluate_gmm_criteria(self, covariance_types=['full', 'tied', 'diag', 'spherical'], n_samples=1, strata_col=None, sample_frac=None):
        """
        Evaluates GMM performance using BIC and ICL for different covariance types.
        Can perform cross-sampling to evaluate stability.
        
        Args:
            covariance_types (list): GMM covariance types to test.
            n_samples (int): If 1, runs on the current self.X data. If > 1, performs
                             cross-sampling and aggregates results.
            strata_col (str): The column to stratify on. Required if n_samples > 1.
            sample_frac (float): The fraction of data to sample. Required if n_samples > 1.
        """
        if n_samples > 1 and (strata_col is None or sample_frac is None):
            raise ValueError("For n_samples > 1, 'strata_col' and 'sample_frac' must be provided.")

        print(f"\n--- Evaluating GMM Information Criteria for k in {list(self.k_range)} ---")
        
        final_aggregated_results = {}

        if n_samples > 1:
            print(f"Running cross-sample evaluation with {n_samples} samples...")
            all_run_results = {cov_type: [] for cov_type in covariance_types}

            for i in range(n_samples):
                print(f"  Iteration {i + 1}/{n_samples}...")
                sample_df = self._original_X_df.groupby(strata_col, group_keys=False).apply(
                    lambda x: x.sample(frac=sample_frac, random_state=self.random_state + i),
                    include_groups=False
                )
                X_sample = sample_df.values

                for cov_type in covariance_types:
                    current_results = {}
                    for k in self.k_range:
                        model = GaussianMixture(n_components=k, covariance_type=cov_type, random_state=self.random_state, init_params='k-means++')
                        labels = model.fit_predict(X_sample)
                        
                        bic = model.bic(X_sample)
                        probs = model.predict_proba(X_sample)
                        entropy = -np.sum(probs * np.log(probs + 1e-9))
                        icl = bic + entropy
                        
                        current_results[k] = {'BIC': bic, 'ICL': icl}
                    all_run_results[cov_type].append(pd.DataFrame.from_dict(current_results, orient='index'))
            
            print("\nAggregating results...")
            for cov_type, results_list in all_run_results.items():
                if results_list:
                    concatenated_df = pd.concat(results_list)
                    mean_df = concatenated_df.groupby(concatenated_df.index).mean()
                    std_df = concatenated_df.groupby(concatenated_df.index).std()
                    final_aggregated_results[cov_type] = mean_df.join(std_df, lsuffix='_mean', rsuffix='_std')
        else:
            for cov_type in covariance_types:
                print(f"Testing covariance type: '{cov_type}'...")
                current_results = {}
                for k in self.k_range:
                    model = GaussianMixture(n_components=k, covariance_type=cov_type, random_state=self.random_state, init_params='k-means++')
                    labels = model.fit_predict(self.X)
                    
                    bic = model.bic(self.X)
                    probs = model.predict_proba(self.X)
                    entropy = -np.sum(probs * np.log(probs + 1e-9))
                    icl = bic + entropy
                    
                    current_results[k] = {'BIC': bic, 'ICL': icl}
                final_aggregated_results[cov_type] = pd.DataFrame.from_dict(current_results, orient='index')

        # --- Plotting Results ---
        metrics_to_plot = ['BIC', 'ICL']
        metric_properties = {
            'BIC': 'Score (Lower is Better)',
            'ICL': 'Score (Lower is Better)'
        }
        
        fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
        axs = axs.flatten()
        
        fig.suptitle('GMM Hyperparameter Evaluation', fontsize=16)

        for i, metric in enumerate(metrics_to_plot):
            ax = axs[i]
            for cov_type, results_df in final_aggregated_results.items():
                self._plot_metric(ax, results_df, metric, cov_type)
            ax.set_title(metric)
            ax.set_ylabel(metric_properties[metric])
            ax.set_xlabel('Number of Components (k)')
            ax.grid(True)
            ax.legend()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        
        return final_aggregated_results

    def plot_gmm_posterior_probabilities(self, palette='viridis'):
        """
        Plots the distribution of posterior probabilities for GMM.

        For each data point assigned to a cluster, this plot shows the distribution
        of the probability (i.e., the model's confidence) for that specific assignment.

        High, tight distributions near 1.0 indicate well-separated and confident clusters.
        Wide distributions or those centered lower suggest cluster overlap or uncertainty.

        Args:
            palette (str, optional): The color palette for the violin plot.
        """
        # 1. Check if the final GMM has been trained
        if self.final_models.get('gmm') is None or 'cluster_gmm' not in self.X_df.columns:
            print("Please run `run_final_models()` with 'gmm' before plotting probabilities.")
            return

        print("\n--- Generating GMM Posterior Probability Plots ---")

        # 2. Get the posterior probabilities for all points and all clusters
        gmm_model = self.final_models['gmm']
        # Use the feature data the model was trained on
        feature_cols = gmm_model.feature_names_in_
        posterior_probs = gmm_model.predict_proba(self.X_df[feature_cols])

        # 3. Create a DataFrame for plotting
        # Get the assigned cluster labels
        assigned_clusters = self.X_df['cluster_gmm']
        
        # For each point, get the probability of it belonging to its *assigned* cluster
        # This uses numpy's advanced indexing to pick the right probability from each row
        confidence_scores = posterior_probs[np.arange(len(posterior_probs)), assigned_clusters]

        plot_df = pd.DataFrame({
            'Assigned Cluster': assigned_clusters,
            'Posterior Probability': confidence_scores
        })

        # 4. Create the violin plot
        plt.figure(figsize=(12, 7))
        sns.violinplot(data=plot_df, x='Assigned Cluster', y='Posterior Probability', 
            hue='Assigned Cluster', palette=palette, legend=False)

        plt.title(f'Distribuição das probabilidades posteriores do modelo GMM (k={gmm_model.n_components})', fontsize=16)
        plt.xlabel('Cluster')
        plt.ylabel("Probabilidades posteriores das atribuições de cluster")
        plt.ylim(0, 1.05) # Probabilities are between 0 and 1
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()

        summary_df = plot_df.groupby('Assigned Cluster')['Posterior Probability'].agg(['mean', 'std'])

        return summary_df

    # ==============================================================================
    # EXTERNAL VALIDITY
    # ==============================================================================
    def cross_predict_and_evaluate(self, target_evaluator, name_source='Model A', name_target='Model B'):
        """
        Uses this evaluator's final models to predict cluster assignments on another 
        evaluator's dataset and calculates the Adjusted Rand Index (ARI).

        This method assesses the external validity and robustness of the clustering 
        solutions. A high ARI indicates that the structure found by the source model
        is similar to the structure found by the target model.

        It correctly handles different data sources:
        - K-Means/GMM: Use the reduced-dimension data (self.X_df).
        - Hierarchical: Uses a KNN surrogate model on the reduced-dimension data.
        - K-Prototypes: Use the original, pre-reduction data (self.X_pre_rd_df).
        
        Args:
            target_evaluator (ClusterEvaluator): Another trained instance of the class.
            name_source (str): A descriptive name for this (the source) instance.
            name_target (str): A descriptive name for the target instance.

        Returns:
            pd.Series: A pandas Series containing the ARI scores for each comparable model.
        """
        print(f"\n--- Cross-Model Evaluation: '{name_source}' Model -> '{name_target}' Data ---")
        ari_scores = {}

        # Iterate through all models
        for model_type in ['kmeans', 'gmm', 'hierarchical', 'kprototypes']:
            source_model = self.final_models.get(model_type)
            target_model = target_evaluator.final_models.get(model_type)

            # 1. Check if both source and target models have been trained
            if source_model is None or target_model is None:
                print(f"\n[INFO] Skipping {model_type.upper()}: One or both models not trained.")
                continue

            print(f"\n--- Evaluating {model_type.upper()} ---")

            try:
                # 2. Logic for NUMERICAL models (K-Means, GMM, Hierarchical)
                if model_type in ['kmeans', 'gmm', 'hierarchical']:
                    # These models use the potentially scaled, reduced-dimension data
                    # Get feature names by excluding any existing cluster columns
                    source_features = self.X_df.drop(columns=[c for c in self.X_df.columns if 'cluster' in c], errors='ignore').columns.tolist()
                    target_df = target_evaluator.X_df
                    data_for_prediction = target_df
                
                # 3. Logic for MIXED-DATA model (K-Prototypes)
                elif model_type == 'kprototypes':
                    # This model uses the active, potentially scaled pre-RD data
                    source_data_pre_rd = self.X_pre_rd_df.loc[self.X_df.index]
                    target_data_pre_rd = target_evaluator.X_pre_rd_df.loc[target_evaluator.X_df.index]
                    
                    source_features = source_data_pre_rd.columns.tolist()
                    data_for_prediction = target_data_pre_rd
                
                # 4. Align features and get target labels
                if not set(source_features).issubset(data_for_prediction.columns):
                    missing_cols = set(source_features) - set(data_for_prediction.columns)
                    print(f"[ERROR] Target data is missing features required by the source {model_type.upper()} model.")
                    print(f"        Missing features: {list(missing_cols)}")
                    continue
                
                aligned_target_data = data_for_prediction[source_features]
                true_target_labels = target_evaluator.X_df[f'cluster_{model_type}']

                # 5. Predict labels on the aligned target data using the source model
                print(f"Predicting labels on '{name_target}' data using '{name_source}' model...")

                if model_type in ['kmeans', 'gmm']:
                    predicted_labels = source_model.predict(aligned_target_data)
                
                elif model_type == 'hierarchical':
                    print("  (Using KNN surrogate model for Hierarchical clustering)")
                    source_data_for_knn = self.X_df[source_features]
                    source_labels_for_knn = self.X_df['cluster_hierarchical']
                    
                    # Train a KNN model on the source data and its hierarchical cluster labels
                    knn_surrogate = KNeighborsClassifier(n_neighbors=5)
                    knn_surrogate.fit(source_data_for_knn, source_labels_for_knn)
                    
                    # Predict labels on the target data
                    predicted_labels = knn_surrogate.predict(aligned_target_data)

                elif model_type == 'kprototypes':
                    cat_cols = [col for col in source_data_pre_rd.columns if source_data_pre_rd[col].dtype in ['object', 'category']]
                    categorical_indices = [aligned_target_data.columns.get_loc(col) for col in cat_cols if col in aligned_target_data.columns]
                    predicted_labels = source_model.predict(aligned_target_data.values, categorical=categorical_indices)

                # 6. Calculate and store the Adjusted Rand Score
                ari = adjusted_rand_score(true_target_labels, predicted_labels)
                ari_scores[model_type] = ari
                print(f"✅ Adjusted Rand Score (ARI) for {model_type.upper()}: {ari:.4f}")

            except Exception as e:
                print(f"[ERROR] An unexpected error occurred while evaluating {model_type.upper()}: {e}")

        return pd.Series(ari_scores, name='ARI Scores')

    def evaluate_cross_model_stability(self, target_evaluator, strata_col, sample_frac, hyperparams, n_iterations=50, name_source='Source', name_target='Target'):
        """
        Iteratively evaluates the stability of cluster structure between two datasets.
        """
        # --- Input Validation and Setup ---
        algos_to_run = [algo for algo in hyperparams if algo in ['kmeans', 'gmm', 'hierarchical', 'kprototypes']]
        if not algos_to_run:
            print("[ERROR] No valid algorithms specified in hyperparams dictionary.")
            return

        print(f"\n--- Evaluating Cross-Model Stability ({n_iterations} iterations) ---")
        print(f"'{name_source}' -> '{name_target}'")
        ari_scores = {algo: [] for algo in algos_to_run}

        # --- Iteration Loop ---
        for i in range(n_iterations):
            # --- Sample Data ---
            source_sample_rd = self._original_X_df.groupby(strata_col, group_keys=False).apply(
                lambda x: x.sample(frac=sample_frac, random_state=self.random_state + i), include_groups=False)
            source_sample_pre_rd = self._original_pre_rd_df.loc[source_sample_rd.index]

            target_sample_rd = target_evaluator._original_X_df.groupby(strata_col, group_keys=False).apply(
                lambda x: x.sample(frac=sample_frac, random_state=self.random_state + i + n_iterations), include_groups=False)
            target_sample_pre_rd = target_evaluator._original_pre_rd_df.loc[target_sample_rd.index]
            
            # --- Identify Common Features and Apply Scaling ---
            kproto_features = hyperparams.get('kprototypes', {}).get('features')
            
            # Determine common features based on whether a subset is specified for k-prototypes
            if kproto_features:
                common_cols_pre_rd = [f for f in kproto_features if f in source_sample_pre_rd.columns and f in target_sample_pre_rd.columns]
            else:
                common_cols_pre_rd = source_sample_pre_rd.columns.intersection(target_sample_pre_rd.columns).tolist()

            common_cols_rd = source_sample_rd.columns.intersection(target_sample_rd.columns).tolist()
            
            if (i + 1) % 10 == 0:
                print(f"  Iteration {i + 1}/{n_iterations}...")

            if not common_cols_rd or not common_cols_pre_rd:
                print(f"[Warning] No common features found in iteration {i+1}. Skipping.")
                continue

            # --- Scaling Helper ---
            def _apply_scaler(df, scaler):
                if scaler:
                    cols_to_transform = [col for col in scaler.feature_names_in_ if col in df.columns]
                    if cols_to_transform:
                        df[cols_to_transform] = scaler.transform(df[cols_to_transform])
                return df

            source_sample_rd = _apply_scaler(source_sample_rd, getattr(self, 'scaler_rd', None))
            source_sample_pre_rd = _apply_scaler(source_sample_pre_rd, getattr(self, 'scaler_pre_rd', None))
            target_sample_rd = _apply_scaler(target_sample_rd, getattr(target_evaluator, 'scaler_rd', None))
            target_sample_pre_rd = _apply_scaler(target_sample_pre_rd, getattr(target_evaluator, 'scaler_pre_rd', None))
            
            # --- Filter to Common Features ---
            source_rd_common = source_sample_rd[common_cols_rd]
            target_rd_common = target_sample_rd[common_cols_rd]
            source_pre_rd_common = source_sample_pre_rd[common_cols_pre_rd]
            target_pre_rd_common = target_sample_pre_rd[common_cols_pre_rd]
            
            # --- Iterate through algorithms for this sample ---
            for algo in algos_to_run:
                try:
                    params = hyperparams[algo]
                    
                    # --- Train on Source ---
                    if algo == 'kmeans':
                        model_source = KMeans(n_clusters=params['n_clusters'], n_init='auto', random_state=self.random_state).fit(source_rd_common.values)
                    elif algo == 'gmm':
                        model_source = GaussianMixture(n_components=params['n_components'], covariance_type=params.get('covariance_type', self.covariance_type), random_state=self.random_state).fit(source_rd_common.values)
                    elif algo == 'hierarchical':
                        model_source = AgglomerativeClustering(n_clusters=params['n_clusters'], linkage=params.get('linkage', self.linkage))
                        source_labels = model_source.fit_predict(source_rd_common.values)
                    elif algo == 'kprototypes':
                        cat_indices_source = [i for i, c in enumerate(source_pre_rd_common.columns) if source_pre_rd_common[c].dtype in ['object', 'category']]
                        model_source = KPrototypes(n_clusters=params['n_clusters'], gamma=params.get('gamma', self.gamma_values[0]), init='Cao', random_state=self.random_state, n_init=5, verbose=0)
                        model_source.fit(source_pre_rd_common.values, categorical=cat_indices_source)

                    # --- Train on Target to get 'true' labels ---
                    if algo == 'kmeans':
                        labels_target_true = KMeans(n_clusters=params['n_clusters'], n_init='auto', random_state=self.random_state).fit_predict(target_rd_common.values)
                    elif algo == 'gmm':
                        labels_target_true = GaussianMixture(n_components=params['n_components'], covariance_type=params.get('covariance_type', target_evaluator.covariance_type), random_state=target_evaluator.random_state).fit_predict(target_rd_common.values)
                    elif algo == 'hierarchical':
                        labels_target_true = AgglomerativeClustering(n_clusters=params['n_clusters'], linkage=params.get('linkage', target_evaluator.linkage)).fit_predict(target_rd_common.values)
                    elif algo == 'kprototypes':
                        cat_indices_target = [i for i, c in enumerate(target_pre_rd_common.columns) if target_pre_rd_common[c].dtype in ['object', 'category']]
                        model_target = KPrototypes(n_clusters=params['n_clusters'], gamma=params.get('gamma', self.gamma_values[0]), init='Cao', random_state=target_evaluator.random_state, n_init=5, verbose=0)
                        labels_target_true = model_target.fit_predict(target_pre_rd_common.values, categorical=cat_indices_target)

                    # --- Predict from Source onto Target ---
                    if algo in ['kmeans', 'gmm']:
                        labels_predicted = model_source.predict(target_rd_common.values)
                    elif algo == 'hierarchical':
                        knn = KNeighborsClassifier(n_neighbors=5).fit(source_rd_common.values, source_labels)
                        labels_predicted = knn.predict(target_rd_common.values)
                    elif algo == 'kprototypes':
                        labels_predicted = model_source.predict(target_pre_rd_common.values, categorical=cat_indices_target)
                        
                    ari_scores[algo].append(adjusted_rand_score(labels_target_true, labels_predicted))

                except Exception as e:
                    print(f"[Warning] Could not complete iteration {i+1} for {algo.upper()}: {e}")

        # --- Report Results ---
        print("\nCross-model stability evaluation complete.")
        for name, scores in ari_scores.items():
            if scores:
                print(f"{name.capitalize()} Stability: Mean ARI = {np.mean(scores):.4f}, Std Dev = {np.std(scores):.4f}")
        
        # --- MODIFIED PLOTTING SECTION ---
        if any(ari_scores.values()):
            # Prepare data for the bar chart
            plot_labels = []
            mean_scores = []
            std_devs = []

            # Extract stats for each algorithm
            for algo, scores in ari_scores.items():
                if scores:
                    k = hyperparams[algo].get('n_clusters') or hyperparams[algo].get('n_components')
                    plot_labels.append(f"{algo.capitalize()} (k={k})")
                    mean_scores.append(np.mean(scores))
                    std_devs.append(np.std(scores))

            # Create the bar chart 📊
            plt.figure(figsize=(12, 7))
            colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(plot_labels)))
            bars = plt.bar(plot_labels, mean_scores, yerr=std_devs, color=colors, capsize=5, alpha=0.85, edgecolor='black')

            # Customize the plot
            plt.title(f'Cross-Model Stability: {name_source} -> {name_target}', fontsize=16, fontweight='bold')
            plt.xlabel('Clustering Algorithm', fontsize=12)
            plt.ylabel('Mean Adjusted Rand Score (ARI)', fontsize=12)
            
            # Set y-axis limit to be slightly above the highest bar/error bar
            if mean_scores:
                max_val = max([m + s for m, s in zip(mean_scores, std_devs)])
                plt.ylim(0, max_val * 1.15)
            else:
                plt.ylim(0, 1)

            plt.xticks(rotation=15, ha='right')
            plt.grid(True, linestyle='--', alpha=0.6, axis='y')
            plt.tight_layout()
            plt.show()

        return ari_scores
    
    def plot_cross_prediction_matrix(self, target_evaluator, algorithm, name_source='Source', name_target='Target', palette='viridis'):
        """
        Creates a pair plot for the original and predicted cluster labels in a cross-prediction scenario.
        This is a single-shot visualization using the primary final_models.
        """
        if algorithm not in ['kmeans', 'gmm', 'kprototypes', 'hierarchical']:
            print("[ERROR] Invalid algorithm specified.")
            return

        source_model = self.final_models.get(algorithm)
        if source_model is None or target_evaluator.final_models.get(algorithm) is None:
            print(f"[ERROR] Final models must be trained for both source and target for {algorithm.upper()}.")
            return

        target_df_with_labels = target_evaluator.X_df.copy()
        original_cluster_col = f'cluster_{algorithm}'
        
        # --- Prepare source and target data ---
        if algorithm == 'kprototypes':
            source_data = self.X_pre_rd_df.loc[self.X_df.index]
            target_data = target_evaluator.X_pre_rd_df.loc[target_evaluator.X_df.index]
            source_scaler = getattr(self, 'scaler_pre_rd', None)
        else:
            source_data = self.X_df[[c for c in self.X_df.columns if not c.startswith('cluster')]]
            target_data = target_evaluator.X_df[[c for c in target_evaluator.X_df.columns if not c.startswith('cluster')]]
            source_scaler = getattr(self, 'scaler_rd', None)

        common_features = source_data.columns.intersection(target_data.columns).tolist()
        if not common_features:
            print("[ERROR] No common features found for visualization.")
            return

        print(f"Visualizing based on {len(common_features)} common features...")
        source_data_common = source_data[common_features].copy()
        target_data_common = target_data[common_features].copy()

        # <<< FIX: Apply the source scaler to the target data before prediction >>>
        if source_scaler:
            # Find which of the scaler's features are present in the target dataframe
            numeric_cols_to_transform = [col for col in source_scaler.feature_names_in_ if col in target_data_common.columns]
            if numeric_cols_to_transform:
                print(f"Applying source model's scaler to {len(numeric_cols_to_transform)} target features...")
                target_data_common[numeric_cols_to_transform] = source_scaler.transform(target_data_common[numeric_cols_to_transform])

        # --- Get Predicted Labels ---
        predicted_cluster_col = 'predicted_cluster'
        if algorithm == 'hierarchical':
            source_numeric = pd.get_dummies(source_data_common)
            target_numeric = pd.get_dummies(target_data_common)
            common_numeric_features = source_numeric.columns.union(target_numeric.columns)
            source_final = source_numeric.reindex(columns=common_numeric_features, fill_value=0)
            target_final = target_numeric.reindex(columns=common_numeric_features, fill_value=0)
            
            knn = KNeighborsClassifier(n_neighbors=5)
            # Use original (pre-dummified) source data for fitting KNN
            source_labels_for_knn = self.X_df.loc[source_data_common.index, original_cluster_col]
            knn.fit(source_final, source_labels_for_knn)
            predicted_labels = knn.predict(target_final)
            
        elif algorithm == 'kprototypes':
            cat_cols = source_data_common.select_dtypes(include=['object', 'category']).columns
            for col in cat_cols:
                all_categories = pd.concat([source_data_common[col], target_data_common[col]]).astype('category').cat.categories
                target_data_common[col] = pd.Categorical(target_data_common[col], categories=all_categories)
            
            cat_indices = [i for i, col in enumerate(target_data_common.columns) if target_data_common[col].dtype.name == 'category']
            predicted_labels = source_model.predict(target_data_common.values, categorical=cat_indices)

        else: # K-Means and GMM
            predicted_labels = source_model.predict(target_data_common)

        target_df_with_labels[predicted_cluster_col] = predicted_labels
        
        # --- Plotting ---
        print(f"\n--- Generating Cross-Prediction Matrix Plots for {algorithm.upper()} ---")
        
        if algorithm == 'kprototypes':
            # Use the original, unscaled target data for FAMD visualization
            famd_data = target_evaluator.X_pre_rd_df.loc[target_evaluator.X_df.index, common_features]
            famd = prince.FAMD(n_components=2, n_iter=3, random_state=self.random_state)
            components = famd.fit_transform(famd_data)
            components.columns = ['Component 1', 'Component 2']
            components['Original Cluster'] = target_df_with_labels[original_cluster_col].values
            components['Predicted Cluster'] = target_df_with_labels[predicted_cluster_col].values
            
            fig, axs = plt.subplots(1, 2, figsize=(18, 8), sharex=True, sharey=True)
            sns.scatterplot(ax=axs[0], data=components, x='Component 1', y='Component 2', hue='Original Cluster', palette=palette)
            axs[0].set_title(f"Original Clusters ({name_target})")
            sns.scatterplot(ax=axs[1], data=components, x='Component 1', y='Component 2', hue='Predicted Cluster', palette=palette)
            axs[1].set_title(f"Clusters Predicted by {name_source}")
            fig.suptitle(f"Cross-Prediction Visualization for {algorithm.upper()} (on FAMD Components)", fontsize=16)
            plt.show()

        else:
            # Use original, unscaled data for plotting
            plot_data = target_evaluator.X_df.loc[target_evaluator.X_df.index, common_features]
            plot_data[original_cluster_col] = target_df_with_labels[original_cluster_col]
            plot_data[predicted_cluster_col] = target_df_with_labels[predicted_cluster_col]

            print(f"Plotting original clusters for '{name_target}'...")
            g_original = sns.pairplot(plot_data, vars=common_features, hue=original_cluster_col, palette=palette, corner=True)
            g_original.fig.suptitle(f"Original Clusters ({name_target})", y=1.02, fontsize=16)
            plt.show()
            
            print(f"Plotting clusters predicted by '{name_source}'...")
            g_predicted = sns.pairplot(plot_data, vars=common_features, hue=predicted_cluster_col, palette=palette, corner=True)
            g_predicted.fig.suptitle(f"Clusters Predicted by {name_source}", y=1.02, fontsize=16)
            plt.show()

    # ==============================================================================
    # STABILITY AND SCALABILITY
    # ==============================================================================
    def evaluate_cross_sample_stability(self, strata_col, sample_frac, hyperparams, n_iterations=50):
        """
        Evaluates model stability by training on new stratified samples and predicting
        on the original sample.
        """
        algos_to_run = [algo for algo in hyperparams if algo in ['kmeans', 'gmm', 'hierarchical', 'kprototypes']]
        if not algos_to_run:
            print("[ERROR] No valid algorithms specified in hyperparams dictionary.")
            return

        print(f"\n--- Evaluating Cross-Sample Stability ({n_iterations} iterations) for {', '.join(algos_to_run)} ---")
        ari_scores = {algo: [] for algo in algos_to_run}
        baselines = {}
        
        print("Calculating baseline clusterings on the current stratified sample...")
        baseline_data_rd = self.X_df[[col for col in self.X_df.columns if not col.startswith('cluster')]].copy()
        baseline_data_pre_rd = self._original_pre_rd_df.loc[self.X_df.index].copy()
        
        if hasattr(self, 'scaler_pre_rd'):
            print("  Applying existing pre-RD scaling to K-Prototypes baseline data.")
            num_cols_pre_rd = [col for col in self.scaler_pre_rd.feature_names_in_ if col in baseline_data_pre_rd.columns]
            if num_cols_pre_rd:
                baseline_data_pre_rd[num_cols_pre_rd] = self.scaler_pre_rd.transform(baseline_data_pre_rd[num_cols_pre_rd])
        
        # --- Establish Baseline Clusterings ---
        if 'kmeans' in algos_to_run:
            k = hyperparams['kmeans']['n_clusters']
            model = KMeans(n_clusters=k, n_init='auto', random_state=self.random_state)
            baselines['kmeans'] = model.fit_predict(baseline_data_rd.values)
        
        if 'gmm' in algos_to_run:
            params = hyperparams['gmm']
            model = GaussianMixture(n_components=params['n_components'], covariance_type=params.get('covariance_type', self.covariance_type), random_state=self.random_state, init_params='k-means++')
            baselines['gmm'] = model.fit_predict(baseline_data_rd.values)
        
        if 'hierarchical' in algos_to_run:
            params = hyperparams['hierarchical']
            model = AgglomerativeClustering(n_clusters=params['n_clusters'], linkage=params.get('linkage', self.linkage))
            baselines['hierarchical'] = model.fit_predict(baseline_data_rd.values)

        if 'kprototypes' in algos_to_run:
            params = hyperparams['kprototypes']
            k, gamma, features = params['n_clusters'], params.get('gamma', self.gamma_values[0]), params.get('features')
            
            data_for_baseline = baseline_data_pre_rd[features] if features else baseline_data_pre_rd
            
            cat_indices = [i for i, col in enumerate(data_for_baseline.columns) if data_for_baseline[col].dtype in ['object', 'category']]
            model = KPrototypes(n_clusters=k, gamma=gamma, init='Cao', random_state=self.random_state, n_init=10, verbose=0)
            baselines['kprototypes'] = model.fit_predict(data_for_baseline.values, categorical=cat_indices)

        # --- Iteration Loop ---
        print(f"Beginning {n_iterations} cross-sample validation iterations...")
        for i in range(n_iterations):
            if (i + 1) % 10 == 0:
                print(f"  Iteration {i + 1}/{n_iterations}...")

            new_sample_df_rd = self._original_X_df.groupby(strata_col, group_keys=False).apply(
                lambda x: x.sample(frac=sample_frac, random_state=self.random_state + i),
                include_groups=False
            )
            new_sample_df_pre_rd = self._original_pre_rd_df.loc[new_sample_df_rd.index]

            X_new_sample_rd = new_sample_df_rd.copy()
            if hasattr(self, 'scaler_rd'):
                X_new_sample_rd[self.scaler_rd.feature_names_in_] = self.scaler_rd.transform(X_new_sample_rd[self.scaler_rd.feature_names_in_])

            # --- K-MEANS ---
            if 'kmeans' in algos_to_run:
                model_new = KMeans(n_clusters=hyperparams['kmeans']['n_clusters'], n_init='auto', random_state=self.random_state)
                model_new.fit(X_new_sample_rd.values)
                predicted_labels = model_new.predict(baseline_data_rd.values)
                ari_scores['kmeans'].append(adjusted_rand_score(baselines['kmeans'], predicted_labels))

            # --- GMM ---
            if 'gmm' in algos_to_run:
                params = hyperparams['gmm']
                model_new = GaussianMixture(n_components=params['n_components'], covariance_type=params.get('covariance_type', self.covariance_type), random_state=self.random_state, init_params='k-means++')
                model_new.fit(X_new_sample_rd.values)
                predicted_labels = model_new.predict(baseline_data_rd.values)
                ari_scores['gmm'].append(adjusted_rand_score(baselines['gmm'], predicted_labels))

            # --- HIERARCHICAL ---
            if 'hierarchical' in algos_to_run:
                params = hyperparams['hierarchical']
                cluster_model = AgglomerativeClustering(n_clusters=params['n_clusters'], linkage=params.get('linkage', self.linkage))
                new_labels = cluster_model.fit_predict(X_new_sample_rd.values)
                knn_surrogate = KNeighborsClassifier(n_neighbors=5).fit(X_new_sample_rd.values, new_labels)
                predicted_labels = knn_surrogate.predict(baseline_data_rd.values)
                ari_scores['hierarchical'].append(adjusted_rand_score(baselines['hierarchical'], predicted_labels))
                
            # --- K-PROTOTYPES ---
            if 'kprototypes' in algos_to_run:
                params = hyperparams['kprototypes']
                k, gamma, features = params['n_clusters'], params.get('gamma', self.gamma_values[0]), params.get('features')
                
                new_sample_for_kproto = new_sample_df_pre_rd.copy()
                if hasattr(self, 'scaler_pre_rd'):
                    num_cols_in_sample = [col for col in self.scaler_pre_rd.feature_names_in_ if col in new_sample_for_kproto.columns]
                    if num_cols_in_sample:
                        new_sample_for_kproto[num_cols_in_sample] = self.scaler_pre_rd.transform(new_sample_for_kproto[num_cols_in_sample])
                
                data_for_fitting = new_sample_for_kproto[features] if features else new_sample_for_kproto
                data_for_predicting = baseline_data_pre_rd[features] if features else baseline_data_pre_rd

                cat_indices = [i for i, col in enumerate(data_for_fitting.columns) if data_for_fitting[col].dtype in ['object', 'category']]
                model_new = KPrototypes(n_clusters=k, gamma=gamma, init='Cao', random_state=self.random_state, n_init=5, verbose=0)
                model_new.fit(data_for_fitting.values, categorical=cat_indices)
                predicted_labels = model_new.predict(data_for_predicting.values, categorical=cat_indices)
                ari_scores['kprototypes'].append(adjusted_rand_score(baselines['kprototypes'], predicted_labels))

        # --- Reporting ---
        print("\nCross-sample stability evaluation complete.")
        for name, scores in ari_scores.items():
            if scores:
                k = hyperparams[name].get('n_clusters') or hyperparams[name].get('n_components')
                print(f"{name.capitalize()} (k={k}) Stability: Mean ARI = {np.mean(scores):.4f}, Std Dev = {np.std(scores):.4f}")
        
        # --- MODIFIED PLOTTING SECTION ---
        if any(ari_scores.values()):
            # Prepare data for the bar chart
            plot_labels = []
            mean_scores = []
            std_devs = []

            # Extract stats for each algorithm
            for algo, scores in ari_scores.items():
                if scores:
                    k = hyperparams[algo].get('n_clusters') or hyperparams[algo].get('n_components')
                    plot_labels.append(f"{algo.capitalize()} (k={k})")
                    mean_scores.append(np.mean(scores))
                    std_devs.append(np.std(scores))

            # Create the bar chart 📊
            plt.figure(figsize=(12, 7))
            colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(plot_labels)))
            bars = plt.bar(plot_labels, mean_scores, yerr=std_devs, color=colors, capsize=5, alpha=0.85, edgecolor='black')

            # Customize the plot
            plt.title('Cross-Sample Model Stability Comparison', fontsize=16, fontweight='bold')
            plt.xlabel('Clustering Algorithm', fontsize=12)
            plt.ylabel('Mean Adjusted Rand Score (ARI) vs. Baseline', fontsize=12)
            
            # Set y-axis limit to be slightly above the highest bar/error bar
            if mean_scores:
                max_val = max([m + s for m, s in zip(mean_scores, std_devs)])
                plt.ylim(0, max_val * 1.15)
            else:
                plt.ylim(0, 1)

            plt.xticks(rotation=15, ha='right')
            plt.grid(True, linestyle='--', alpha=0.6, axis='y')
            plt.tight_layout()
            plt.show()
        
        return ari_scores

    def evaluate_stability_by_seed(self, hyperparams, random_seeds):
        """
        Evaluates sensitivity to initialization seeds for KMeans, GMM, and K-Prototypes.
        """
        if not isinstance(hyperparams, dict) or not hyperparams:
            print("[ERROR] Please provide a non-empty dictionary for hyperparams.")
            return
        if not isinstance(random_seeds, list) or len(random_seeds) < 2:
            raise ValueError("Please provide a list with at least two random seeds.")

        print(f"\n--- Evaluating Stability by Seed ---")
        
        # This dictionary will now store the list of ARI scores for each algorithm
        ari_scores_by_algo = {}
        baseline_seed = random_seeds[0]
        other_seeds = random_seeds[1:]

        # --- K-MEANS ---
        if 'kmeans' in hyperparams:
            k = hyperparams['kmeans']['n_clusters']
            baseline_km = KMeans(n_clusters=k, n_init='auto', random_state=baseline_seed).fit_predict(self.X)
            scores = [adjusted_rand_score(baseline_km, KMeans(n_clusters=k, n_init='auto', random_state=seed).fit_predict(self.X)) for seed in other_seeds]
            ari_scores_by_algo['kmeans'] = scores
            print(f"K-Means (k={k}) Stability: Mean ARI = {np.mean(scores):.4f}, Std Dev = {np.std(scores):.4f}")
        
        # --- GMM ---
        if 'gmm' in hyperparams:
            params = hyperparams['gmm']
            k, cov_type = params['n_components'], params.get('covariance_type', self.covariance_type)
            baseline_gmm = GaussianMixture(n_components=k, covariance_type=cov_type, random_state=baseline_seed, init_params='k-means++').fit_predict(self.X)
            scores = [adjusted_rand_score(baseline_gmm, GaussianMixture(n_components=k, covariance_type=cov_type, random_state=seed, init_params='k-means++').fit_predict(self.X)) for seed in other_seeds]
            ari_scores_by_algo['gmm'] = scores
            print(f"GMM ({cov_type}, k={k}) Stability: Mean ARI = {np.mean(scores):.4f}, Std Dev = {np.std(scores):.4f}")
        
        # --- K-PROTOTYPES ---
        if 'kprototypes' in hyperparams:
            params = hyperparams['kprototypes']
            k, gamma, features = params['n_clusters'], params.get('gamma', self.gamma_values[0]), params.get('features')
            
            data_for_kproto = self._original_pre_rd_df.loc[self.X_df.index].copy()
            if hasattr(self, 'scaler_pre_rd'):
                num_cols = [col for col in self.scaler_pre_rd.feature_names_in_ if col in data_for_kproto.columns]
                if num_cols:
                    data_for_kproto[num_cols] = self.scaler_pre_rd.transform(data_for_kproto[num_cols])
            
            data_to_use = data_for_kproto[features] if features else data_for_kproto
            cat_indices = [i for i, col in enumerate(data_to_use.columns) if data_to_use[col].dtype in ['object', 'category']]
            
            model_baseline = KPrototypes(n_clusters=k, gamma=gamma, init='Cao', random_state=baseline_seed, n_init=1, verbose=0)
            baseline_kp = model_baseline.fit_predict(data_to_use.values, categorical=cat_indices)

            scores = []
            for seed in other_seeds:
                model_new = KPrototypes(n_clusters=k, gamma=gamma, init='Cao', random_state=seed, n_init=1, verbose=0)
                labels_new = model_new.fit_predict(data_to_use.values, categorical=cat_indices)
                scores.append(adjusted_rand_score(baseline_kp, labels_new))
                
            ari_scores_by_algo['kprototypes'] = scores
            print(f"K-Prototypes (k={k}, n_init=1) Stability: Mean ARI = {np.mean(scores):.4f}, Std Dev = {np.std(scores):.4f}")
            
        # --- HIERARCHICAL ---
        if 'hierarchical' in hyperparams:
            # Hierarchical is deterministic, so its score is always 1.0 with 0 std dev.
            # We add it as a list of one item to be consistent for plotting.
            ari_scores_by_algo['hierarchical'] = [1.0] 
            print(f"Hierarchical Stability: Mean ARI = 1.0000, Std Dev = 0.0000 (Deterministic)")

        # --- PLOTTING SECTION ---
        if ari_scores_by_algo:
            # Prepare data for the bar chart
            plot_labels = []
            mean_scores = []
            std_devs = []

            # Extract stats for each algorithm
            for algo, scores in ari_scores_by_algo.items():
                if scores:
                    k = hyperparams[algo].get('n_clusters') or hyperparams[algo].get('n_components')
                    plot_labels.append(f"{algo.capitalize()} (k={k})")
                    mean_scores.append(np.mean(scores))
                    std_devs.append(np.std(scores))

            # Create the bar chart 📊
            plt.figure(figsize=(10, 7))
            colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(plot_labels)))
            bars = plt.bar(plot_labels, mean_scores, yerr=std_devs, color=colors, capsize=5, alpha=0.85, edgecolor='black')

            # Customize the plot
            plt.title('Initialization Seed Stability', fontsize=16, fontweight='bold')
            plt.xlabel('Clustering Algorithm', fontsize=12)
            plt.ylabel('Mean Adjusted Rand Score (ARI) vs. Baseline Seed', fontsize=12)
            
            # Set y-axis limit
            if mean_scores:
                max_val = max([m + s for m, s in zip(mean_scores, std_devs)])
                plt.ylim(0, max_val * 1.15 if max_val < 1 else 1.15)
            else:
                plt.ylim(0, 1)

            plt.xticks(rotation=15, ha='right')
            plt.grid(True, linestyle='--', alpha=0.6, axis='y')
            plt.tight_layout()
            plt.show()

        return ari_scores_by_algo

    def evaluate_processing_time(self, hyperparams, max_sample_size, n_steps=10, n_iterations=5):
        """
        Evaluates and plots the processing time (scalability) for algorithms
        by sampling from the full dataset up to a specified maximum sample size.
        """
        if not isinstance(hyperparams, dict) or not hyperparams:
            print("[ERROR] Please provide a non-empty dictionary for hyperparams.")
            return

        # --- 1. VALIDATE INPUTS AND SET UP SAMPLING ---
        # Ensure the required full datasets exist
        if not hasattr(self, 'X_rd_df') or not hasattr(self, '_original_pre_rd_df'):
            print("[ERROR] Full datasets 'self.X_rd_df' or 'self._original_pre_rd_df' not found.")
            return
            
        total_available_samples = self.X_rd_df.shape[0]
        
        # Check if the requested max_sample_size is feasible
        if max_sample_size > total_available_samples:
            print(f"[Warning] max_sample_size ({max_sample_size}) is larger than the total available samples ({total_available_samples}).")
            print(f"Clamping max_sample_size to {total_available_samples}.")
            max_sample_size = total_available_samples

        print(f"\n--- Evaluating Processing Time (up to n={max_sample_size}, {n_iterations} iterations per step) ---")
        
        # Generate the sample sizes for each step of the evaluation
        min_samples = max(100, int(max_sample_size / n_steps)) # Ensure a meaningful minimum
        sample_sizes = sorted(list(set(np.linspace(min_samples, max_sample_size, n_steps, dtype=int))))
        
        times_all_iterations = {name: [[] for _ in sample_sizes] for name in hyperparams}

        # --- 2. ITERATE THROUGH SAMPLE SIZES AND TIME ALGORITHMS ---
        for i, n in enumerate(sample_sizes):
            print(f"Timing for n={n} samples (running {n_iterations} iterations)...")
            for iter_num in range(n_iterations):
                # Randomly select indices from the full dataset range
                indices = np.random.choice(total_available_samples, n, replace=False)
                
                # Sample the dimensionality-reduced data for numerical algorithms
                sample_X_rd = self.X_rd_df.iloc[indices].values
                
                # --- Time K-Means ---
                if 'kmeans' in hyperparams:
                    start = time.perf_counter()
                    KMeans(n_clusters=hyperparams['kmeans']['n_clusters'], n_init='auto', random_state=self.random_state + iter_num).fit(sample_X_rd)
                    times_all_iterations['kmeans'][i].append(time.perf_counter() - start)
                
                # --- Time Hierarchical ---
                if 'hierarchical' in hyperparams:
                    params = hyperparams['hierarchical']
                    start = time.perf_counter()
                    AgglomerativeClustering(n_clusters=params['n_clusters'], linkage=params.get('linkage', self.linkage)).fit(sample_X_rd)
                    times_all_iterations['hierarchical'][i].append(time.perf_counter() - start)

                # --- Time GMM ---
                if 'gmm' in hyperparams:
                    params = hyperparams['gmm']
                    start = time.perf_counter()
                    GaussianMixture(n_components=params['n_components'], covariance_type=params.get('covariance_type', self.covariance_type), random_state=self.random_state + iter_num, init_params='k-means++').fit(sample_X_rd)
                    times_all_iterations['gmm'][i].append(time.perf_counter() - start)
                
                # --- Time K-Prototypes ---
                if 'kprototypes' in hyperparams:
                    params = hyperparams['kprototypes']
                    features = params.get('features')
                    # Sample the pre-reduction data using the same indices
                    sample_X_pre_rd_df = self._original_pre_rd_df.iloc[indices]
                    
                    data_to_use = sample_X_pre_rd_df[features] if features else sample_X_pre_rd_df
                    cat_indices = [idx for idx, col in enumerate(data_to_use.columns) if data_to_use[col].dtype in ['object', 'category']]
                    
                    start = time.perf_counter()
                    KPrototypes(n_clusters=params['n_clusters'], gamma=params.get('gamma', self.gamma_values[0]), init='Cao', random_state=self.random_state + iter_num, n_init=1, verbose=0).fit(data_to_use.values, categorical=cat_indices)
                    times_all_iterations['kprototypes'][i].append(time.perf_counter() - start)

        # --- 3. CALCULATE STATS AND PLOT RESULTS ---
        mean_times = {name: [np.mean(step_times) for step_times in times_all_iterations[name]] for name in hyperparams if any(times_all_iterations[name])}
        std_times = {name: [np.std(step_times) for step_times in times_all_iterations[name]] for name in hyperparams if any(times_all_iterations[name])}

        plt.figure(figsize=(12, 7))
        ax = plt.gca()

        for algo_name in mean_times:
            label = algo_name.capitalize()
            # Add extra details to labels for clarity
            if algo_name == 'hierarchical':
                linkage = hyperparams[algo_name].get('linkage', self.linkage)
                label = f'Hierarchical ({linkage})'
            elif algo_name == 'gmm':
                cov_type = hyperparams[algo_name].get('covariance_type', self.covariance_type)
                label = f'GMM ({cov_type})'
            ax.errorbar(sample_sizes, mean_times[algo_name], yerr=std_times[algo_name], fmt='-o', capsize=4, label=label, alpha=0.8)

        ax.set_xlabel('Number of Samples (n)', fontsize=12)
        ax.set_ylabel('Processing Time (seconds)', fontsize=12)
        ax.set_title('Algorithm Processing Time vs. Sample Size', fontsize=16, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        plt.tight_layout()
        plt.show()

        # --- 4. RETURN RESULTS AS A DATAFRAME ---
        timing_df = pd.DataFrame({'Number of Samples': sample_sizes})
        for name in mean_times:
            timing_df[f'{name.capitalize()} Mean Time (s)'] = mean_times[name]
            timing_df[f'{name.capitalize()} Std Dev Time (s)'] = std_times[name]
        
        return timing_df.set_index('Number of Samples')

    def evaluate_perturbation_stability(self, hyperparams, n_iterations=50, noise_level=0.1):
        """
        Evaluates clustering stability by adding Gaussian noise to the dataset.
        """
        if not hyperparams:
            print("Error: The provided hyperparams dictionary is empty.")
            return

        print(f"\n--- Evaluating Noise Perturbation Stability (Noise Level: {noise_level}) ---")
        rng = np.random.default_rng(self.random_state)
        
        # Noise perturbation is only well-defined for fully numeric algorithms
        algos_to_run = [algo for algo in hyperparams if algo in ['kmeans', 'gmm', 'hierarchical']]
        if not algos_to_run:
            print("No applicable numerical algorithms found in hyperparams for perturbation test.")
            return
            
        ari_scores = {algo: [] for algo in algos_to_run}
        baselines = {}

        print("Calculating baseline clusterings on original data...")
        if 'kmeans' in algos_to_run:
            k = hyperparams['kmeans']['n_clusters']
            baselines['kmeans'] = KMeans(n_clusters=k, n_init='auto', random_state=self.random_state).fit_predict(self.X)
        
        if 'hierarchical' in algos_to_run:
            params = hyperparams['hierarchical']
            baselines['hierarchical'] = AgglomerativeClustering(n_clusters=params['n_clusters'], linkage=params.get('linkage', self.linkage)).fit_predict(self.X)
        
        if 'gmm' in algos_to_run:
            params = hyperparams['gmm']
            baselines['gmm'] = GaussianMixture(n_components=params['n_components'], covariance_type=params.get('covariance_type', self.covariance_type), init_params='k-means++', random_state=self.random_state).fit_predict(self.X)

        print("Baselines calculated. Starting noise iterations...")
        for i in range(n_iterations):
            if (i + 1) % 10 == 0:
                print(f"  Iteration {i + 1}/{n_iterations}...")
            
            noise = rng.normal(loc=0.0, scale=noise_level, size=self.X.shape)
            X_perturbed = self.X + noise

            if 'kmeans' in ari_scores:
                k = hyperparams['kmeans']['n_clusters']
                perturbed_labels = KMeans(n_clusters=k, n_init='auto', random_state=self.random_state + i).fit_predict(X_perturbed)
                ari_scores['kmeans'].append(adjusted_rand_score(baselines['kmeans'], perturbed_labels))
            
            if 'hierarchical' in ari_scores:
                params = hyperparams['hierarchical']
                perturbed_labels = AgglomerativeClustering(n_clusters=params['n_clusters'], linkage=params.get('linkage', self.linkage)).fit_predict(X_perturbed)
                ari_scores['hierarchical'].append(adjusted_rand_score(baselines['hierarchical'], perturbed_labels))
            
            if 'gmm' in ari_scores:
                params = hyperparams['gmm']
                perturbed_labels = GaussianMixture(n_components=params['n_components'], covariance_type=params.get('covariance_type', self.covariance_type), random_state=self.random_state + i, init_params='k-means++').fit_predict(X_perturbed)
                ari_scores['gmm'].append(adjusted_rand_score(baselines['gmm'], perturbed_labels))

        print("\nNoise evaluation complete.")
        for name, scores in ari_scores.items():
            if scores:
                k = hyperparams[name].get('n_clusters') or hyperparams[name].get('n_components')
                print(f"{name.capitalize()} (k={k}) Stability: Mean ARI = {np.mean(scores):.4f}, Std Dev = {np.std(scores):.4f}")
        
        # --- MODIFIED PLOTTING SECTION ---
        if any(ari_scores.values()):
            # Prepare data for the bar chart
            plot_labels = []
            mean_scores = []
            std_devs = []

            # Extract stats for each algorithm
            for algo, scores in ari_scores.items():
                if scores:
                    k = hyperparams[algo].get('n_clusters') or hyperparams[algo].get('n_components')
                    plot_labels.append(f"{algo.capitalize()} (k={k})")
                    mean_scores.append(np.mean(scores))
                    std_devs.append(np.std(scores))

            # Create the bar chart 📊
            plt.figure(figsize=(10, 7))
            colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(plot_labels)))
            bars = plt.bar(plot_labels, mean_scores, yerr=std_devs, color=colors, capsize=5, alpha=0.85, edgecolor='black')

            # Customize the plot
            plt.title(f'Noise Perturbation Stability (Noise Level: {noise_level})', fontsize=16, fontweight='bold')
            plt.xlabel('Clustering Algorithm', fontsize=12)
            plt.ylabel('Mean Adjusted Rand Score (ARI) vs. Baseline', fontsize=12)
            
            # Set y-axis limit to be slightly above the highest bar/error bar
            if mean_scores:
                max_val = max([m + s for m, s in zip(mean_scores, std_devs)])
                plt.ylim(0, max_val * 1.15)
            else:
                plt.ylim(0, 1)

            plt.xticks(rotation=15, ha='right')
            plt.grid(True, linestyle='--', alpha=0.6, axis='y')
            plt.tight_layout()
            plt.show()
        
        return ari_scores

    # ==============================================================================
    # VARIABLE IMPORTANCE 
    # ==============================================================================
    def plot_decision_tree_importance(self, max_depth=3):
        """
        Trains a Decision Tree Classifier to predict cluster labels based on the
        original (pre-reduction) features and visualizes the resulting tree.

        This method now automatically handles categorical features using one-hot encoding.

        Args:
            max_depth (int, optional): The maximum depth of the decision tree.
                                       A smaller number (e.g., 3 or 4) creates a more
                                       interpretable visualization. Defaults to 3.
        """
        # 1. Check if any final models have been trained
        if all(model is None for model in self.final_models.values()):
            print("Please run `run_final_models()` before analyzing variable importance.")
            return

        # 2. Get the original features corresponding to the data that was clustered.
        try:
            # Use .copy() to avoid potential SettingWithCopyWarning later
            X_original_features = self._original_pre_rd_df.loc[self.X_df.index].copy()
        except Exception as e:
            print(f"Error retrieving original features: {e}")
            print("Ensure that `self._original_pre_rd_df` contains the original data and its index aligns.")
            return

        print(f"\n--- Generating Decision Tree Visualizations (max_depth={max_depth}) ---")

        # 3. Automatically detect and preprocess categorical features
        
        # Identify categorical columns by data type
        categorical_cols = X_original_features.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) > 0:
            print(f"Found categorical features: {list(categorical_cols)}. Applying One-Hot Encoding...")
            # Apply one-hot encoding to create a purely numeric DataFrame
            X_processed = pd.get_dummies(X_original_features, columns=categorical_cols, drop_first=False)
            print(f"Original feature count: {len(X_original_features.columns)}. New feature count after encoding: {len(X_processed.columns)}.")
        else:
            print("No categorical features found. Using original features for the tree.")
            X_processed = X_original_features

        # 4. Iterate through each of the trained models
        # <<< MODIFIED >>> Added 'kprototypes' to the loop
        for algorithm in ['kmeans', 'hierarchical', 'gmm', 'kprototypes']:
            cluster_col = f'cluster_{algorithm}'
            model = self.final_models.get(algorithm)

            if model is None or cluster_col not in self.X_df.columns:
                print(f"Skipping Decision Tree for {algorithm.upper()} (model not run).")
                continue

            print(f"Training and plotting tree for {algorithm.upper()} clusters...")
            
            # --- Prepare Data ---
            y_labels = self.X_df[cluster_col]
            # Use the columns from the PROCESSED DataFrame
            feature_names = X_processed.columns.tolist() 
            class_names = [f'Cluster {i}' for i in sorted(y_labels.unique())]

            # --- Train Decision Tree Classifier ---
            tree_classifier = DecisionTreeClassifier(
                max_depth=max_depth, 
                random_state=self.random_state
            )
            # Fit the tree using the PROCESSED data
            tree_classifier.fit(X_processed, y_labels)

            # --- Plot the Tree ---
            plt.figure(figsize=(20, 10))
            plot_tree(
                tree_classifier,
                feature_names=feature_names, # Pass the new, encoded feature names
                class_names=class_names,
                filled=True,
                rounded=True,
                fontsize=10,
                precision=2
            )
            
            plt.title(f'Decision Tree for Explaining {algorithm.upper()} Clusters', fontsize=16)
            plt.show()

    def _calculate_permutation_importance(self, algorithm, n_repeats=10, score_metric='accuracy'):
        """Calculates and returns permutation importance as a DataFrame."""
        cluster_col = f'cluster_{algorithm}'
        if cluster_col not in self.X_df.columns:
            return None

        print(f"\n--- Calculating Permutation Importance for {algorithm.upper()} ---")
        
        # ... (All the calculation steps from your original function go here) ...
        # 1. Prepare data
        X_original_features = self._original_pre_rd_df.loc[self.X_df.index]
        y_labels = self.X_df[cluster_col]
        X_processed = pd.get_dummies(X_original_features, drop_first=True)
        
        # 2. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_labels, test_size=0.3, random_state=self.random_state, stratify=y_labels
        )

        # 3. Train surrogate model
        surrogate_model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        surrogate_model.fit(X_train, y_train)

        # 4. Calculate importance
        result = permutation_importance(
            surrogate_model, X_test, y_test, n_repeats=n_repeats,
            random_state=self.random_state, scoring=score_metric, n_jobs=-1
        )

        importances = pd.DataFrame({
            'feature': X_processed.columns.tolist(),
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        return importances

    def plot_permutation_importance(self, n_repeats=10, score_metric='accuracy'):
        """Calculates and PLOTS feature importance for all algorithms."""
        # Loop through algorithms as before
        for algorithm in ['kmeans', 'hierarchical', 'gmm', 'kprototypes']:
            importances_df = self._calculate_permutation_importance(algorithm, n_repeats, score_metric)
            
            if importances_df is None:
                continue

            # 5. Plot the results (the plotting part remains here)
            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance_mean', y='feature', data=importances_df, palette='viridis', hue='feature', dodge=False)
            plt.xlabel(f'Mean Importance ({score_metric.capitalize()} Drop)')
            plt.ylabel('Feature')
            plt.title(f'Permutation Feature Importance for {algorithm.upper()} Clusters')
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()

    def calculate_cluster_statistics(self, algorithm, p_value_threshold=0.05):
        """
        Calculates statistics to check for significant differences between clusters.

        - For continuous variables: Performs a one-way ANOVA across all clusters.
        - For categorical variables: Performs a Chi-square test of independence across all clusters.

        Args:
            algorithm (str): The algorithm to evaluate ('kmeans', 'hierarchical', 'gmm', or 'kprototypes'). # <<< UPDATED >>>
            p_value_threshold (float, optional): The alpha level for significance testing. Defaults to 0.05.

        Returns:
            tuple: A tuple containing two DataFrames:
                   - (anova_results, chi2_results)
        """
        cluster_col = f'cluster_{algorithm}'
        if cluster_col not in self.X_df.columns:
            print(f"Error: Final model for '{algorithm}' has not been run. Please run `run_final_models()` first.")
            return None, None

        print(f"\n--- Calculating Cluster Statistics for {algorithm.upper()} ---")
        
        # 1. Merge cluster labels with the original feature set for interpretation
        df_full = self._original_pre_rd_df.loc[self.X_df.index].copy()
        df_full[cluster_col] = self.X_df[cluster_col]

        # 2. Identify variable types
        continuous_vars = df_full.select_dtypes(include=np.number).columns.tolist()
        categorical_vars = df_full.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove cluster column from list of variables to test
        if cluster_col in continuous_vars:
            continuous_vars.remove(cluster_col)

        # --- 3. ANOVA for Continuous Variables ---
        anova_results = []
        if continuous_vars:
            cluster_ids = sorted(df_full[cluster_col].unique())
            # ANOVA requires at least 2 groups (clusters) to compare
            if len(cluster_ids) > 1:
                for var in continuous_vars:
                    # Create a list where each element is the data for one cluster
                    groups = [df_full[df_full[cluster_col] == c][var].dropna() for c in cluster_ids]
                    
                    # Ensure all groups have sufficient data to perform the test
                    if all(len(g) > 1 for g in groups):
                        f_stat, p_val = stats.f_oneway(*groups)
                        anova_results.append({
                            'Variable': var,
                            'F-statistic': f_stat,
                            'P-value': p_val,
                            'Significant': p_val < p_value_threshold
                        })
            
            if anova_results:
                anova_results = pd.DataFrame(anova_results).set_index('Variable')
            else:
                anova_results = pd.DataFrame(columns=['F-statistic', 'P-value', 'Significant'])
                
        else:
            anova_results = pd.DataFrame(columns=['F-statistic', 'P-value', 'Significant'])
        
        # --- 4. Chi-Square Tests for Categorical Variables (Unchanged) ---
        chi2_results = []
        if categorical_vars:
            for var in categorical_vars:
                contingency_table = pd.crosstab(df_full[var], df_full[cluster_col])
                if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                    chi2, p_val, _, _ = stats.chi2_contingency(contingency_table)
                    chi2_results.append({
                        'Variable': var,
                        'Chi-Square': chi2,
                        'P-value': p_val,
                        'Significant': p_val < p_value_threshold
                    })
            if chi2_results:
                chi2_results = pd.DataFrame(chi2_results).set_index('Variable')
            else:
                chi2_results = pd.DataFrame(columns=['Chi-Square', 'P-value', 'Significant'])
        else:
            chi2_results = pd.DataFrame(columns=['Chi-Square', 'P-value', 'Significant'])
            
        print("Statistical analysis complete.")
        return anova_results, chi2_results

    def calculate_effect_sizes(self, algorithm):
        """
        Calculates effect sizes to measure the magnitude of differences between clusters.

        - For continuous variables: Calculates Cohen's d for each pair of clusters.
        - For categorical variables: Calculates Cramér's V across all clusters.

        Args:
            algorithm (str): The algorithm to evaluate ('kmeans', 'hierarchical', 'gmm', or 'kprototypes'). # <<< UPDATED >>>

        Returns:
            tuple: A tuple containing two DataFrames:
                   - (cohens_d_results, cramers_v_results)
        """
        cluster_col = f'cluster_{algorithm}'
        if cluster_col not in self.X_df.columns:
            print(f"Error: Final model for '{algorithm}' has not been run. Please run `run_final_models()` first.")
            return None, None

        print(f"\n--- Calculating Effect Sizes for {algorithm.upper()} ---")
        
        df_full = self._original_pre_rd_df.loc[self.X_df.index].copy()
        df_full[cluster_col] = self.X_df[cluster_col]

        continuous_vars = df_full.select_dtypes(include=np.number).columns.tolist()
        categorical_vars = df_full.select_dtypes(include=['object', 'category']).columns.tolist()
        if cluster_col in continuous_vars:
            continuous_vars.remove(cluster_col)

        # --- Cohen's d for Continuous Variables ---
        cohens_d_results = []
        if continuous_vars:
            cluster_ids = sorted(df_full[cluster_col].unique())
            for var in continuous_vars:
                for c1, c2 in combinations(cluster_ids, 2):
                    group1 = df_full[df_full[cluster_col] == c1][var].dropna()
                    group2 = df_full[df_full[cluster_col] == c2][var].dropna()
                    
                    n1, n2 = len(group1), len(group2)
                    if n1 > 1 and n2 > 1:
                        mean1, mean2 = group1.mean(), group2.mean()
                        std1, std2 = group1.std(), group2.std()
                        
                        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
                        if pooled_std > 0:
                            d = (mean1 - mean2) / pooled_std
                        else:
                            d = 0.0
                            
                        cohens_d_results.append({
                            'Variable': var,
                            'Comparison': f'Cluster {c1} vs {c2}',
                            "Cohen's d": d
                        })
            cohens_d_results = pd.DataFrame(cohens_d_results).set_index(['Variable', 'Comparison'])
        else:
            cohens_d_results = pd.DataFrame(columns=["Cohen's d"])

        # --- Cramér's V for Categorical Variables ---
        cramers_v_results = []
        if categorical_vars:
            for var in categorical_vars:
                contingency_table = pd.crosstab(df_full[var], df_full[cluster_col])
                if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                    chi2, _, _, _ = stats.chi2_contingency(contingency_table)
                    n = contingency_table.sum().sum()
                    phi2 = chi2 / n
                    r, k = contingency_table.shape
                    v = np.sqrt(phi2 / min(k - 1, r - 1))
                    cramers_v_results.append({'Variable': var, "Cramér's V": v})
            cramers_v_results = pd.DataFrame(cramers_v_results).set_index('Variable')
        else:
            cramers_v_results = pd.DataFrame(columns=["Cramér's V"])

        print("Effect size calculation complete.")
        return cohens_d_results, cramers_v_results
    
    def export_full_analysis_to_excel(self, filepath, n_repeats=10, p_value_threshold=0.05):
        """
        Runs all analyses and saves the results into a single, multi-sheet Excel file.
        
        Args:
            filepath (str): Path to save the output Excel file (e.g., 'full_analysis.xlsx').
            n_repeats (int, optional): n_repeats for permutation importance. Defaults to 10.
            p_value_threshold (float, optional): Alpha level for significance tests. Defaults to 0.05.
        """
        # Find all algorithms that have been run
        cluster_cols = [col for col in self.X_df.columns if col.startswith('cluster_')]
        algorithms = [col.replace('cluster_', '') for col in cluster_cols]
        if not algorithms:
            print("Error: No cluster results found to analyze.")
            return

        print(f"Found completed algorithms: {', '.join(algorithms)}")

        # Use ExcelWriter to save multiple sheets to one file
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            print(f"\n--- Exporting all analysis results to '{filepath}' ---")
            
            # Loop through each algorithm and run all analyses
            for alg in algorithms:
                print(f"\nProcessing analysis for {alg.upper()}...")

                # 1. Permutation Importance
                perm_importance_df = self._calculate_permutation_importance(alg, n_repeats)
                if perm_importance_df is not None:
                    perm_importance_df.to_excel(writer, sheet_name=f'{alg.upper()} - Permutation Imp.', index=False)

                # 2. Cluster Statistics (ANOVA / Chi-Square)
                anova_df, chi2_df = self.calculate_cluster_statistics(alg, p_value_threshold)
                if anova_df is not None and not anova_df.empty:
                    anova_df.to_excel(writer, sheet_name=f'{alg.upper()} - ANOVA')
                if chi2_df is not None and not chi2_df.empty:
                    chi2_df.to_excel(writer, sheet_name=f'{alg.upper()} - Chi-Square')

                # 3. Effect Sizes (Cohen's d / Cramér's V)
                cohens_d_df, cramers_v_df = self.calculate_effect_sizes(alg)
                if cohens_d_df is not None and not cohens_d_df.empty:
                    cohens_d_df.to_excel(writer, sheet_name=f'{alg.upper()} - Cohen\'s d')
                if cramers_v_df is not None and not cramers_v_df.empty:
                    cramers_v_df.to_excel(writer, sheet_name=f'{alg.upper()} - Cramer\'s V')
        
        print("\n✅ All analyses have been successfully exported.")

    def generate_cluster_profiles(self, algorithm, categorical_mode='percentage'):
        """
        Generates descriptive profile tables for each cluster and optionally saves them to Excel.

        Args:
            algorithm (str): The algorithm to evaluate ('kmeans', 'hierarchical', 'gmm', 'kprototypes').
            filepath (str, optional): The path to the Excel file to save results. 
                                      If None, results are not saved. Defaults to None.
            categorical_mode (str, optional): How to display categorical data.
                                              'percentage' (default) or 'count'.

        Returns:
            tuple: A tuple containing two DataFrames:
                   - (numeric_profile, categorical_profile)
        """
        cluster_col = f'cluster_{algorithm}'
        if cluster_col not in self.X_df.columns:
            print(f"Error: Cluster labels for '{algorithm}' not found. Run the final model first.")
            return None, None

        print(f"\n--- Generating Cluster Profiles for {algorithm.upper()} ---")

        # 1. Prepare the full DataFrame (no changes here)
        df_full = self._original_pre_rd_df.loc[self.X_df.index].copy()
        df_full[cluster_col] = self.X_df[cluster_col]

        # 2. Identify variable types (no changes here)
        continuous_vars = df_full.select_dtypes(include=np.number).columns.tolist()
        categorical_vars = df_full.select_dtypes(include=['object', 'category']).columns.tolist()
        if cluster_col in continuous_vars:
            continuous_vars.remove(cluster_col)

        # 3. Profile Numeric Variables (no changes here)
        if continuous_vars:
            numeric_profile = df_full.groupby(cluster_col)[continuous_vars].agg(['mean', 'std'])
        else:
            numeric_profile = pd.DataFrame()

        # 4. Profile Categorical Variables (no changes here)
        if categorical_vars:
            all_cat_profiles = []
            for var in categorical_vars:
                normalize = True if categorical_mode == 'percentage' else False
                profile = pd.crosstab(df_full[cluster_col], df_full[var], normalize='index')
                if normalize:
                    profile = (profile * 100).round(2)
                all_cat_profiles.append(profile)
            categorical_profile = pd.concat(all_cat_profiles, keys=categorical_vars, axis=0)
        else:
            categorical_profile = pd.DataFrame()
        
        print("Profile generation complete.")
        return numeric_profile, categorical_profile
    
    def export_all_profiles_to_excel(self, filepath, categorical_mode='percentage'):
        """
        Generates profiles for all completed algorithms and saves them to a single Excel file.

        Each algorithm will have two sheets in the file: one for numeric profiles
        and one for categorical profiles.

        Args:
            filepath (str): The path to the Excel file to save results (e.g., 'cluster_comparison.xlsx').
            categorical_mode (str, optional): 'percentage' (default) or 'count'.
        """
        # Step 1: Automatically find all algorithms that have been run
        cluster_cols = [col for col in self.X_df.columns if col.startswith('cluster_')]
        if not cluster_cols:
            print("Error: No cluster results found. Please run your models first.")
            return

        algorithms_to_run = [re.sub('^cluster_', '', col) for col in cluster_cols]
        print(f"Found {len(algorithms_to_run)} completed algorithms: {', '.join(algorithms_to_run)}")

        # Step 2: Open an Excel writer to save all sheets in one file
        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                print(f"--- Exporting all profiles to '{filepath}' ---")
                
                # Step 3: Loop through each algorithm
                for algorithm in algorithms_to_run:
                    print(f"  -> Generating profile for {algorithm.upper()}...")
                    
                    # Get the profile DataFrames using the other method
                    numeric_profile, categorical_profile = self.generate_cluster_profiles(
                        algorithm, 
                        categorical_mode=categorical_mode
                    )

                    # Write both DataFrames to the Excel file with clear sheet names
                    if numeric_profile is not None and not numeric_profile.empty:
                        sheet_name_num = f'{algorithm.upper()} - Numeric'
                        numeric_profile.to_excel(writer, sheet_name=sheet_name_num)

                    if categorical_profile is not None and not categorical_profile.empty:
                        sheet_name_cat = f'{algorithm.upper()} - Categorical'
                        categorical_profile.to_excel(writer, sheet_name=sheet_name_cat)

            print(f"✅ All profiles successfully saved to '{filepath}'")
        except Exception as e:
            print(f"Error: Could not save file. {e}")