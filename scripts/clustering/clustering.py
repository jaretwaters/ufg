# ==============================================================================
# Importing Required Libraries
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from itertools import combinations, product
import warnings

# --- Caching & Parallelization ---
from joblib import Memory, Parallel, delayed

# Suppress joblib warnings for clearer output
warnings.filterwarnings("ignore", category=UserWarning, module='joblib')

# --- Clustering & Preprocessing ---
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from kmodes.kprototypes import KPrototypes
import gower
import scipy.cluster.hierarchy as sch
import scipy.stats as stats

# --- Dimensionality Reduction for Visualization ---
from prince import FAMD 

# --- Evaluation Metrics ---
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
from validclust import dunn

# --- Variable Importance (Surrogate Models) ---
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


# ==============================================================================
# CLASS 1: FeatureSelector (for pre-processing before K-Prototypes)
# ==============================================================================
class FeatureSelector:
    """
    A standalone class for analyzing, visualizing, and selecting features from a 
    DataFrame with mixed numerical and categorical data.
    """
    def __init__(self, df, random_state=42):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")
        self.df = df.copy()
        self.random_state = random_state
        self.numerical_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(exclude=np.number).columns.tolist()
        print("--- FeatureSelector Initialized ---")
    
    def _calculate_normalized_entropy(self, series):
        series = series.dropna()
        if series.empty or len(series.unique()) <= 1: return 0.0
        value_counts = series.value_counts()
        probs = value_counts / len(series)
        entropy = -np.sum(probs * np.log2(probs))
        if len(probs) <= 1: return 0.0
        return entropy / np.log2(len(probs))

    def analyze_features(self):
        print("\n--- Starting Feature Analysis ---")
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 14))
        means = self.df[self.numerical_cols].mean().abs()
        stds = self.df[self.numerical_cols].std()
        cv = (stds / means.replace(0, np.nan)).fillna(0).sort_values(ascending=False)
        sns.barplot(x=cv.values, y=cv.index, ax=axes[0, 0], hue=cv.index, palette='viridis', legend=False)
        axes[0, 0].set_title('Coefficient of Variation (Numerical)')
        corr_matrix = self.df[self.numerical_cols].corr()
        sns.heatmap(corr_matrix, ax=axes[0, 1], cmap='coolwarm', annot=False)
        axes[0, 1].set_title('Correlation Matrix (Numerical)')
        entropies = {col: self._calculate_normalized_entropy(self.df[col]) for col in self.categorical_cols}
        ent_series = pd.Series(entropies).sort_values(ascending=False)
        sns.barplot(x=ent_series.values, y=ent_series.index, ax=axes[1, 0], hue=ent_series.index, palette='plasma', legend=False)
        axes[1, 0].set_title('Normalized Entropy (Categorical)')
        n = len(self.categorical_cols)
        nmi_matrix = pd.DataFrame(np.eye(n), index=self.categorical_cols, columns=self.categorical_cols)
        for i in range(n):
            for j in range(i + 1, n):
                nmi_score = normalized_mutual_info_score(self.df[self.categorical_cols[i]], self.df[self.categorical_cols[j]])
                nmi_matrix.iloc[i, j] = nmi_matrix.iloc[j, i] = nmi_score
        sns.heatmap(nmi_matrix, ax=axes[1, 1], cmap='magma', annot=False)
        axes[1, 1].set_title('Normalized Mutual Information (Categorical)')
        plt.suptitle('Feature Analysis Dashboard', fontsize=20)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.show()

    def select_features(self, cv_threshold=None, correlation_threshold=None, entropy_threshold=None, mutual_info_threshold=None):
        print("\n--- Starting Feature Selection Process ---")
        df_selected = self.df.copy()
        cols_to_drop = set()
        if cv_threshold is not None:
            means = df_selected[self.numerical_cols].mean().abs()
            stds = df_selected[self.numerical_cols].std()
            cv = stds / means.replace(0, np.nan)
            cols_to_drop.update(cv[cv < cv_threshold].index)
        if correlation_threshold is not None:
            corr_matrix = df_selected[self.numerical_cols].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
            cols_to_drop.update(to_drop)
        if entropy_threshold is not None:
            for col in self.categorical_cols:
                if self._calculate_normalized_entropy(df_selected[col]) < entropy_threshold:
                    cols_to_drop.add(col)
        if mutual_info_threshold is not None:
            cat_cols = [c for c in self.categorical_cols if c in df_selected.columns]
            for col1, col2 in combinations(cat_cols, 2):
                if normalized_mutual_info_score(df_selected[col1], df_selected[col2]) > mutual_info_threshold:
                    ent1 = self._calculate_normalized_entropy(df_selected[col1])
                    ent2 = self._calculate_normalized_entropy(df_selected[col2])
                    cols_to_drop.add(col1 if ent1 < ent2 else col2)
        
        df_selected.drop(columns=list(cols_to_drop), inplace=True, errors='ignore')
        print(f"Dropped {len(cols_to_drop)} columns: {list(cols_to_drop)}")
        return df_selected

# ==============================================================================
# CLASS 2: ClusteringComparator (Main Unified Class)
# ==============================================================================
class ClusteringComparator:
    """
    A unified class for performing, evaluating, and comparing multiple clustering
    algorithms. Optimized with parallel processing and caching.
    """
    def __init__(self, X_original, X_reduced, k_range, 
                linkage='ward', covariance_type='full', random_state=42,
                n_jobs=-1, cache_dir='__joblib_cache__'):
        if not X_original.index.equals(X_reduced.index):
            raise ValueError("Indices of X_original and X_reduced must be aligned.")
        self._master_original_df = X_original.copy()
        self._master_reduced_df = X_reduced.copy()
        self.X_original_df = X_original.copy()
        self.X_reduced_df = X_reduced.copy()
        self.k_range = k_range
        self.random_state = random_state
        self.linkage = linkage
        self.covariance_type = covariance_type
        self.n_jobs = n_jobs
        self.numerical_cols_kproto = self.X_original_df.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols_kproto = self.X_original_df.select_dtypes(exclude=np.number).columns.tolist()
        self.results = {'kmeans': pd.DataFrame(), 'hierarchical': pd.DataFrame(), 
                        'gmm': pd.DataFrame(), 'kprototypes': pd.DataFrame()}
        self.final_models = {'kmeans': None, 'hierarchical': None, 'gmm': None, 'kprototypes': None, 'hierarchical_surrogate': None}
        
        # Caching Setup
        if not os.path.exists(cache_dir): os.makedirs(cache_dir)
        self.memory = Memory(cache_dir, verbose=0)
        self._cached_gower_matrix = self.memory.cache(gower.gower_matrix)

        print("--- ClusteringComparator Initialized ---")

    # ==============================================================================
    # SECTION 1: DATA PREPARATION & HELPERS
    # ==============================================================================
    def stratified_sample(self, strata_col, sample_frac):
        if strata_col not in self._master_original_df.columns:
            raise ValueError(f"Stratification column '{strata_col}' not found.")
        print(f"Performing stratified sampling on '{strata_col}' with a {sample_frac:.2%} fraction...")
        sampled_indices = self._master_original_df.groupby(strata_col, group_keys=False).apply(
            lambda x: x.sample(frac=sample_frac, random_state=self.random_state)
        ).index
        self.X_original_df = self._master_original_df.loc[sampled_indices].copy()
        self.X_reduced_df = self._master_reduced_df.loc[sampled_indices].copy()
        print(f"Sampling complete. Active datasets reduced to {len(self.X_original_df)} rows.")
        return self.X_original_df, self.X_reduced_df
        
    def scale_features(self, scale_original_numeric=False, scale_reduced=True):
        if scale_original_numeric:
            self.scaler_kproto = StandardScaler()
            self.X_original_df[self.numerical_cols_kproto] = self.scaler_kproto.fit_transform(self.X_original_df[self.numerical_cols_kproto])
        if scale_reduced:
            self.scaler_reduced = StandardScaler()
            self.X_reduced_df[self.X_reduced_df.columns] = self.scaler_reduced.fit_transform(self.X_reduced_df)
        print("Scaling complete.")

    def _train_surrogate_model(self, data, labels, n_neighbors=5):
        print(f"Training KNN surrogate model (k={n_neighbors}) for prediction...")
        surrogate = KNeighborsClassifier(n_neighbors=n_neighbors)
        surrogate.fit(data, labels)
        return surrogate

    def _calculate_twss(self, X, labels):
        twss = 0
        for label in np.unique(labels):
            cluster_points = X[labels == label]
            if len(cluster_points) > 0:
                centroid = cluster_points.mean(axis=0)
                twss += np.sum((cluster_points - centroid) ** 2)
        return twss

    def _calculate_icl(self, X, gmm_model):
            """Calculates the Integrated Completed Likelihood (ICL) for a fitted GMM."""
            bic = gmm_model.bic(X)
            # ICL is approximated as BIC minus twice the entropy of the posterior probabilities
            posterior_probs = gmm_model.predict_proba(X)
            # Add a small epsilon to prevent log(0) for numerically stable calculation
            entropy = -np.sum(posterior_probs * np.log(posterior_probs + 1e-9))
            return bic - (2 * entropy)

    # ==============================================================================
    # SECTION 2: CLUSTERING & EVALUATION (INTERNAL VALIDITY)
    # ==============================================================================
    def run_and_evaluate(self, algorithms=['kmeans', 'hierarchical', 'gmm', 'kprototypes'],
                        cross_sample_numeric=0, cross_sample_kproto=0, strata_col=None):
        """
        Runs and evaluates specified clustering algorithms with optional stratified cross-sampling.
        """
        # --- 1. Input Validation ---
        if strata_col and strata_col not in self._master_original_df.columns:
            raise ValueError(f"Stratification column '{strata_col}' not found in the master dataframe.")

        numeric_algos = [a for a in algorithms if a in ['kmeans', 'hierarchical', 'gmm']]
        
        # --- 2. Evaluation for Numeric-Space Models (KMeans, Hierarchical, GMM) ---
        if numeric_algos:
            if cross_sample_numeric <= 0:
                # --- Single Run Evaluation ---
                print("--- Running Single Evaluation for Numeric Models ---")
                if 'kmeans' in numeric_algos:
                    self.results['kmeans'] = self._evaluate_kmeans()
                if 'hierarchical' in numeric_algos:
                    self.results['hierarchical'] = self._evaluate_hierarchical()
                if 'gmm' in numeric_algos:
                    self.results['gmm'] = self._evaluate_gmm()
            else:
                # --- Cross-Sampled Evaluation ---
                print(f"--- Running Cross-Sample Evaluation for Numeric Models ({cross_sample_numeric} iterations) ---")
                if strata_col:
                    print(f"--- Using STRATIFIED sampling on column: '{strata_col}' ---")
                
                results_list = []
                for i in range(cross_sample_numeric):
                    print(f"  Iteration {i + 1}/{cross_sample_numeric}...")
                    
                    if strata_col:
                        iter_df = self._master_original_df.groupby(strata_col, group_keys=False).apply(
                            lambda x: x.sample(n=len(x), replace=True, random_state=self.random_state + i))
                        iter_X_redu = self._master_reduced_df.loc[iter_df.index]
                        if strata_col in iter_X_redu.columns:
                            iter_X_redu = iter_X_redu.drop(columns=[strata_col])
                    else:
                        indices = np.random.choice(len(self._master_reduced_df), size=len(self.X_reduced_df), replace=True)
                        iter_X_redu = self._master_reduced_df.iloc[indices]
                    
                    # Call new individual evaluation methods
                    if 'kmeans' in numeric_algos:
                        results_list.append(self._evaluate_kmeans(data=iter_X_redu).assign(algorithm='kmeans'))
                    if 'hierarchical' in numeric_algos:
                        results_list.append(self._evaluate_hierarchical(data=iter_X_redu).assign(algorithm='hierarchical'))
                    if 'gmm' in numeric_algos:
                        results_list.append(self._evaluate_gmm(data=iter_X_redu).assign(algorithm='gmm'))

                concat_df = pd.concat(results_list)
                grouped = concat_df.groupby(['algorithm', 'k']).agg(['mean', 'sem'])
                grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
                for algo in numeric_algos:
                    self.results[algo] = grouped.loc[algo]

        # --- 3. Evaluation for K-Prototypes (logic remains the same) ---
        if 'kprototypes' in algorithms:
            if cross_sample_kproto <= 0:
                print("--- Running Single Evaluation for K-Prototypes ---")
                self.results['kprototypes'] = self._evaluate_kprototypes()
            else:
                print(f"--- Running Cross-Sample Evaluation for K-Prototypes ({cross_sample_kproto} iterations) ---")
                if strata_col:
                    print(f"--- Using STRATIFIED sampling on column: '{strata_col}' ---")

                results_list = []
                for i in range(cross_sample_kproto):
                    print(f"  Iteration {i + 1}/{cross_sample_kproto}...")
                    
                    if strata_col:
                        iter_X_orig_sampled = self._master_original_df.groupby(strata_col, group_keys=False).apply(
                            lambda x: x.sample(n=len(x), replace=True, random_state=self.random_state + i))
                        iter_X_orig = iter_X_orig_sampled.drop(columns=[strata_col])
                    else:
                        indices = np.random.choice(len(self._master_original_df), size=len(self.X_original_df), replace=True)
                        iter_X_orig = self._master_original_df.iloc[indices]
                        
                    results_list.append(self._evaluate_kprototypes(data=iter_X_orig))

                concat_df = pd.concat(results_list)
                self.results['kprototypes'] = concat_df.groupby(['k']).agg(['mean', 'sem'])
                self.results['kprototypes'].columns = ['_'.join(col).strip() for col in self.results['kprototypes'].columns.values]

        print("\n✅ Evaluation complete for all specified algorithms.")

    def _evaluate_kmeans(self, data=None):
        """
        Evaluates K-Means clustering across the specified k_range using parallel processing.
        """
        data = self.X_reduced_df if data is None else data
        X = data.values

        def _process_k(k):
            model = KMeans(n_clusters=k, n_init='auto', random_state=self.random_state).fit(X)
            labels = model.labels_
            if len(np.unique(labels)) > 1:
                sil = silhouette_score(X, labels)
                dbi = davies_bouldin_score(X, labels)
            else:
                sil, dbi = np.nan, np.nan # Safeguard for single-cluster result
            return k, {'TWSS': model.inertia_, 'Silhouette': sil, 'DBI': dbi}

        # Run evaluation for each k in parallel
        k_results = Parallel(n_jobs=self.n_jobs)(delayed(_process_k)(k) for k in self.k_range)
        
        # Reconstruct and return the results DataFrame
        results_df = pd.DataFrame([res for _, res in sorted(k_results)], index=self.k_range)
        results_df.index.name = 'k'
        return results_df

    def _evaluate_hierarchical(self, data=None):
        """
        Evaluates Hierarchical clustering across the specified k_range using parallel processing.
        """
        data = self.X_reduced_df if data is None else data
        X = data.values

        def _process_k(k):
            labels = AgglomerativeClustering(n_clusters=k, linkage=self.linkage).fit_predict(X)
            if len(np.unique(labels)) > 1:
                twss = self._calculate_twss(X, labels)
                sil = silhouette_score(X, labels)
                dbi = davies_bouldin_score(X, labels)
            else:
                twss, sil, dbi = np.nan, np.nan, np.nan # Safeguard
            return k, {'TWSS': twss, 'Silhouette': sil, 'DBI': dbi}
        
        # Run evaluation for each k in parallel
        k_results = Parallel(n_jobs=self.n_jobs)(delayed(_process_k)(k) for k in self.k_range)
        
        # Reconstruct and return the results DataFrame
        results_df = pd.DataFrame([res for _, res in sorted(k_results)], index=self.k_range)
        results_df.index.name = 'k'
        return results_df
    
    def _evaluate_gmm(self, data=None):
        """
        Evaluates GMM clustering across the specified k_range using parallel processing.
        """
        data = self.X_reduced_df if data is None else data
        X = data.values

        def _process_k(k):
            model = GaussianMixture(n_components=k, covariance_type=self.covariance_type, random_state=self.random_state).fit(X)
            labels = model.predict(X)
            if len(np.unique(labels)) > 1:
                twss = self._calculate_twss(X, labels)
                sil = silhouette_score(X, labels)
                dbi = davies_bouldin_score(X, labels)
            else:
                twss, sil, dbi = np.nan, np.nan, np.nan # Safeguard
            return k, {'TWSS': twss, 'Silhouette': sil, 'DBI': dbi}

        # Run evaluation for each k in parallel
        k_results = Parallel(n_jobs=self.n_jobs)(delayed(_process_k)(k) for k in self.k_range)
        
        # Reconstruct and return the results DataFrame
        results_df = pd.DataFrame([res for _, res in sorted(k_results)], index=self.k_range)
        results_df.index.name = 'k'
        return results_df

    def _evaluate_kprototypes(self, data=None):
        """Simplified helper for single/default gamma evaluation."""
        data = self.X_original_df if data is None else data
        cat_indices = [data.columns.get_loc(c) for c in self.categorical_cols_kproto if c in data.columns]
        gower_mat = self._cached_gower_matrix(data.to_numpy(dtype=object))
        
        def _process_run(k):
            # Uses default gamma
            model = KPrototypes(n_clusters=k, gamma=None, init='Cao', n_init=10, random_state=self.random_state, n_jobs=1)
            labels = model.fit_predict(data.values, categorical=cat_indices)
            return {
                'k': k, 'Cost': model.cost_,
                'Silhouette': silhouette_score(gower_mat, labels, metric='precomputed') if len(np.unique(labels)) > 1 else -1,
                'Dunn': dunn(gower_mat, labels) if len(np.unique(labels)) > 1 else -1
            }
        
        results_list = Parallel(n_jobs=self.n_jobs)(delayed(_process_run)(k) for k in self.k_range)
        return pd.DataFrame(results_list)

    def run_final_models(self, optimal_k_dict):
        print("\n--- Training Final Models ---")
        if 'kmeans' in optimal_k_dict:
            k = optimal_k_dict['kmeans']
            model = KMeans(n_clusters=k, n_init='auto', random_state=self.random_state)
            self.X_reduced_df['cluster_kmeans'] = model.fit_predict(self.X_reduced_df)
            self.final_models['kmeans'] = model
        if 'hierarchical' in optimal_k_dict:
            k = optimal_k_dict['hierarchical']
            model = AgglomerativeClustering(n_clusters=k, linkage=self.linkage)
            labels = model.fit_predict(self.X_reduced_df)
            self.X_reduced_df['cluster_hierarchical'] = labels
            self.final_models['hierarchical'] = model
            features_for_surrogate = self.X_reduced_df.drop(columns=[c for c in self.X_reduced_df if c.startswith('cluster')])
            self.final_models['hierarchical_surrogate'] = self._train_surrogate_model(features_for_surrogate, labels)
        if 'gmm' in optimal_k_dict:
            k = optimal_k_dict['gmm']
            model = GaussianMixture(n_components=k, covariance_type=self.covariance_type, random_state=self.random_state)
            self.X_reduced_df['cluster_gmm'] = model.fit_predict(self.X_reduced_df)
            self.final_models['gmm'] = model
        # Sync all labels to the original dataframe
        for col in self.X_reduced_df.columns:
            if col.startswith('cluster_'): self.X_original_df[col] = self.X_reduced_df[col]
        print("\n✅ Final models trained and synced.")
        return self.X_original_df

    # ==============================================================================
    # SECTION 4: ALGORITHM-SPECIFIC METHODS
    # ==============================================================================
    def evaluate_linkage_methods(self, linkage_methods=['ward', 'complete', 'average'], n_iterations=0, strata_col=None):
        print(f"\n--- Evaluating Hierarchical Linkage Methods ---")
        
        if n_iterations > 0:
            print(f"Running cross-sample evaluation with {n_iterations} iterations...")
            if strata_col and strata_col not in self._master_original_df.columns:
                raise ValueError(f"Stratification column '{strata_col}' not found.")
            
            iteration_results = []
            for i in range(n_iterations):
                if strata_col:
                    iter_df = self._master_original_df.groupby(strata_col, group_keys=False).apply(
                        lambda x: x.sample(n=len(x), replace=True, random_state=self.random_state + i)
                    )
                    iter_X_redu = self._master_reduced_df.loc[iter_df.index]
                    # Ensure strata_col is not in the reduced features if it exists
                    if strata_col in iter_X_redu.columns:
                        iter_X_redu = iter_X_redu.drop(columns=[strata_col])
                else:
                    indices = np.random.choice(len(self._master_reduced_df), size=len(self.X_reduced_df), replace=True)
                    iter_X_redu = self._master_reduced_df.iloc[indices]
                
                X = iter_X_redu.values
                for linkage in linkage_methods:
                    def _process_linkage_k(k):
                        labels = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(X)
                        if len(set(labels)) > 1:
                            return {'k': k, 'TWSS': self._calculate_twss(X, labels), 'Silhouette': silhouette_score(X, labels), 'DBI': davies_bouldin_score(X, labels)}
                        return {'k': k, 'TWSS': float('inf'), 'Silhouette': -1, 'DBI': float('inf')}
                    
                    results_list = Parallel(n_jobs=self.n_jobs)(delayed(_process_linkage_k)(k) for k in self.k_range)
                    df = pd.DataFrame(results_list).assign(linkage=linkage, iteration=i)
                    iteration_results.append(df)
            
            all_results_df = pd.concat(iteration_results)
            grouped = all_results_df.groupby(['linkage', 'k']).agg(
                TWSS_mean=('TWSS', 'mean'), TWSS_sem=('TWSS', 'sem'),
                Silhouette_mean=('Silhouette', 'mean'), Silhouette_sem=('Silhouette', 'sem'),
                DBI_mean=('DBI', 'mean'), DBI_sem=('DBI', 'sem')
            ).reset_index()

        else: # Single run logic
            print("Running single evaluation...")
            X = self.X_reduced_df.drop(columns=[strata_col], errors='ignore').values
            all_results = {}
            for linkage in linkage_methods:
                def _process_linkage_k(k):
                    labels = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(X)
                    if len(set(labels)) > 1:
                        return {'k': k, 'TWSS': self._calculate_twss(X, labels), 'Silhouette': silhouette_score(X, labels), 'DBI': davies_bouldin_score(X, labels)}
                    return {'k': k, 'TWSS': float('inf'), 'Silhouette': -1, 'DBI': float('inf')}
                
                results_list = Parallel(n_jobs=self.n_jobs)(delayed(_process_linkage_k)(k) for k in self.k_range)
                all_results[linkage] = pd.DataFrame(results_list)

        # Plotting logic remains the same...
        fig, axs = plt.subplots(1, 3, figsize=(24, 7), sharex=True)
        metrics = ['TWSS', 'Silhouette', 'DBI']
        
        if n_iterations > 0:
            for ax, metric in zip(axs, metrics):
                mean_col, sem_col = f'{metric}_mean', f'{metric}_sem'
                for linkage_name in linkage_methods:
                    subset = grouped[grouped['linkage'] == linkage_name]
                    ax.plot(subset['k'], subset[mean_col], '-', label=linkage_name)
                    ax.fill_between(subset['k'], subset[mean_col] - subset[sem_col], subset[mean_col] + subset[sem_col], alpha=0.2)
                ax.set_title(f'Mean {metric} (± SEM)'); ax.set_xlabel('Number of Clusters (k)'); ax.legend()
        else:
            for ax, metric in zip(axs, metrics):
                for linkage_name, results_df in all_results.items():
                    ax.plot(results_df['k'], results_df[metric], '-', label=linkage_name)
                ax.set_title(metric); ax.set_xlabel('Number of Clusters (k)'); ax.legend()
        
        axs[0].set_ylabel('Score')
        plt.suptitle('Hierarchical Linkage Method Comparison'); plt.show()

    def plot_dendrogram(self, linkage=None, k_cut=None):
        linkage = linkage or self.linkage
        print(f"\n--- Generating Dendrogram with '{linkage}' linkage ---")
        Z = sch.linkage(self.X_reduced_df.values, method=linkage)
        plt.figure(figsize=(15, 7))
        if k_cut is not None and k_cut > 1:
            plt.title(f'Dendrogram with {k_cut} Clusters (Linkage: {linkage})')
            distances = sorted(Z[:, 2], reverse=True)
            threshold = distances[k_cut - 2]
            sch.dendrogram(Z, color_threshold=threshold)
            plt.axhline(y=threshold, c='grey', lw=1.5, linestyle='--')
        else:
            plt.title(f'Hierarchical Clustering Dendrogram (Linkage: {linkage})'); sch.dendrogram(Z)
        plt.xlabel('Data Points / Clusters'); plt.ylabel('Distance'); plt.grid(axis='y'); plt.show()
    
    def evaluate_gmm_criteria(self, covariance_types=['full', 'tied', 'diag'], n_iterations=0, strata_col=None):
        print(f"\n--- Evaluating GMM Covariance Types ---")
        
        if n_iterations > 0:
            print(f"Running cross-sample evaluation with {n_iterations} iterations...")
            if strata_col and strata_col not in self._master_original_df.columns:
                raise ValueError(f"Stratification column '{strata_col}' not found.")
                
            iteration_results = []
            for i in range(n_iterations):
                if strata_col:
                    iter_df = self._master_original_df.groupby(strata_col, group_keys=False).apply(
                        lambda x: x.sample(n=len(x), replace=True, random_state=self.random_state + i)
                    )
                    iter_X_redu = self._master_reduced_df.loc[iter_df.index]
                    # Ensure strata_col is not in the reduced features if it exists
                    if strata_col in iter_X_redu.columns:
                        iter_X_redu = iter_X_redu.drop(columns=[strata_col])
                else:
                    indices = np.random.choice(len(self._master_reduced_df), size=len(self.X_reduced_df), replace=True)
                    iter_X_redu = self._master_reduced_df.iloc[indices]
                
                X = iter_X_redu.values
                for cov_type in covariance_types:
                    def _process_gmm_k(k):
                        model = GaussianMixture(n_components=k, covariance_type=cov_type, random_state=self.random_state).fit(X)
                        return {'k': k, 'BIC': model.bic(X), 'ICL': self._calculate_icl(X, model)}
                    
                    results_list = Parallel(n_jobs=self.n_jobs)(delayed(_process_gmm_k)(k) for k in self.k_range)
                    df = pd.DataFrame(results_list).assign(cov_type=cov_type, iteration=i)
                    iteration_results.append(df)
            
            all_results_df = pd.concat(iteration_results)
            grouped = all_results_df.groupby(['cov_type', 'k']).agg(
                BIC_mean=('BIC', 'mean'), BIC_sem=('BIC', 'sem'),
                ICL_mean=('ICL', 'mean'), ICL_sem=('ICL', 'sem')
            ).reset_index()

        else: # Single run logic
            print("Running single evaluation...")
            X = self.X_reduced_df.drop(columns=[strata_col], errors='ignore').values
            all_results = {}
            for cov_type in covariance_types:
                def _process_gmm_k(k):
                    model = GaussianMixture(n_components=k, covariance_type=cov_type, random_state=self.random_state).fit(X)
                    return {'k': k, 'BIC': model.bic(X), 'ICL': self._calculate_icl(X, model)}
                
                results_list = Parallel(n_jobs=self.n_jobs)(delayed(_process_gmm_k)(k) for k in self.k_range)
                all_results[cov_type] = pd.DataFrame(results_list)
                
        # Plotting logic remains the same...
        fig, axs = plt.subplots(1, 2, figsize=(18, 7), sharex=True)
        metrics = ['BIC', 'ICL']

        for ax, metric in zip(axs, metrics):
            if n_iterations > 0:
                mean_col, sem_col = f'{metric}_mean', f'{metric}_sem'
                for cov_type in covariance_types:
                    subset = grouped[grouped['cov_type'] == cov_type]
                    ax.plot(subset['k'], subset[mean_col], '-', label=cov_type)
                    ax.fill_between(subset['k'], subset[mean_col] - subset[sem_col], subset[mean_col] + subset[sem_col], alpha=0.2)
                ax.set_title(f'Mean {metric} (± SEM)')
            else:
                for cov_type, results_df in all_results.items():
                    ax.plot(results_df['k'], results_df[metric], '-', label=cov_type)
                ax.set_title(metric)
            
            ax.set_xlabel('Number of Components (k)')
            ax.set_ylabel(f'{metric} (Lower is Better)')
            ax.legend()
            ax.grid(True)
        
        plt.suptitle('GMM Hyperparameter Evaluation'); plt.show()

    def plot_gmm_posterior_probabilities(self, palette='viridis'):
        if self.final_models.get('gmm') is None: print("Final GMM must be trained first."); return
        gmm_model = self.final_models['gmm']
        X_redu_no_clusters = self.X_reduced_df.drop(columns=[c for c in self.X_reduced_df if c.startswith('cluster')])
        posterior_probs = gmm_model.predict_proba(X_redu_no_clusters)
        assigned_clusters = self.X_reduced_df['cluster_gmm']
        confidence_scores = posterior_probs[np.arange(len(posterior_probs)), assigned_clusters]
        plot_df = pd.DataFrame({'Assigned Cluster': assigned_clusters, 'Posterior Probability': confidence_scores})
        
        plt.figure(figsize=(12, 7))
        sns.violinplot(data=plot_df, x='Assigned Cluster', y='Posterior Probability', hue='Assigned Cluster', palette=palette, legend=False)
        plt.title(f'GMM Posterior Probability Distribution (k={gmm_model.n_components})')
        plt.ylabel("Model's Confidence in Assignment"); plt.ylim(0, 1.05); plt.show()

    def evaluate_kprototypes_gamma(self, gamma_list, n_iterations=0, strata_col=None):
        """
        Evaluates K-Prototypes performance for different gamma values, with optional
        cross-sampling for robustness.
        """
        print(f"\n--- Evaluating K-Prototypes Gamma Values: {gamma_list} ---")

        def _process_gamma_run(k, gamma, data, cat_indices, gower_mat):
            model = KPrototypes(n_clusters=k, gamma=gamma, init='Cao', n_init=10, random_state=self.random_state, n_jobs=1)
            labels = model.fit_predict(data.values, categorical=cat_indices)
            if len(np.unique(labels)) > 1:
                sil = silhouette_score(gower_mat, labels, metric='precomputed')
                dunn_score = dunn(gower_mat, labels)
            else:
                sil, dunn_score = -1, -1
            return {'k': k, 'gamma': gamma, 'Cost': model.cost_, 'Silhouette': sil, 'Dunn': dunn_score}

        if n_iterations > 0:
            print(f"Running cross-sample evaluation with {n_iterations} iterations...")
            if strata_col and strata_col not in self._master_original_df.columns:
                raise ValueError(f"Stratification column '{strata_col}' not found.")
            
            all_results_list = []
            for i in range(n_iterations):
                print(f"  Iteration {i + 1}/{n_iterations}...")
                if strata_col:
                    iter_X_orig_sampled = self._master_original_df.groupby(strata_col, group_keys=False).apply(
                        lambda x: x.sample(n=len(x), replace=True, random_state=self.random_state + i)
                    )
                    # Drop strata_col BEFORE using the data
                    iter_X_orig = iter_X_orig_sampled.drop(columns=[strata_col])
                else:
                    indices = np.random.choice(len(self._master_original_df), size=len(self.X_original_df), replace=True)
                    iter_X_orig = self._master_original_df.iloc[indices]
                
                gower_mat_iter = gower.gower_matrix(iter_X_orig.to_numpy(dtype=object))
                cat_indices_iter = [iter_X_orig.columns.get_loc(c) for c in self.categorical_cols_kproto if c in iter_X_orig.columns]
                
                params_list = list(product(self.k_range, gamma_list))
                iter_results = Parallel(n_jobs=self.n_jobs)(
                    delayed(_process_gamma_run)(k, g, iter_X_orig, cat_indices_iter, gower_mat_iter) for k, g in params_list
                )
                all_results_list.extend(iter_results)
            
            results_df = pd.DataFrame(all_results_list)

        else: # Single run (original logic)
            print("Running single evaluation...")
            data = self.X_original_df.drop(columns=[strata_col], errors='ignore')
            cat_indices = [data.columns.get_loc(c) for c in self.categorical_cols_kproto if c in data.columns]
            gower_mat = self._cached_gower_matrix(data.to_numpy(dtype=object))
            
            params_list = list(product(self.k_range, gamma_list))
            results_list = Parallel(n_jobs=self.n_jobs)(
                delayed(_process_gamma_run)(k, g, data, cat_indices, gower_mat) for k, g in params_list
            )
            results_df = pd.DataFrame(results_list)

        # Plotting logic remains the same...
        fig, axs = plt.subplots(1, 3, figsize=(24, 7), sharex=True)
        metrics = ['Cost', 'Silhouette', 'Dunn']
        for ax, metric in zip(axs, metrics):
            sns.lineplot(data=results_df, x='k', y=metric, hue='gamma', marker='o', palette='viridis', ax=ax, errorbar='sd')
            plot_title = f'Mean {metric} (± SD)' if n_iterations > 0 else f'{metric} vs. k'
            ax.set_title(plot_title)
            ax.set_xlabel('Number of Clusters (k)')
            ax.grid(True)
            ax.legend(title='Gamma')
        
        plt.suptitle('K-Prototypes Gamma Parameter Tuning', y=1.02)
        plt.tight_layout()
        plt.show()

        return results_df

    # ==============================================================================
    # SECTION 5: VISUALIZATION
    # ==============================================================================
    def plot_evaluation_metrics(self):
        print("\n--- Plotting Evaluation Metrics ---")
        fig, axs = plt.subplots(1, 3, figsize=(24, 7))
        metrics = ['TWSS', 'Silhouette', 'DBI']
        
        for ax, metric in zip(axs, metrics):
            for algo in ['kmeans', 'hierarchical', 'gmm']:
                if not self.results[algo].empty:
                    res = self.results[algo]
                    # Check for single run vs cross-sample run columns
                    if f'{metric}_mean' in res.columns:
                        mean_col, sem_col = f'{metric}_mean', f'{metric}_sem'
                        x_axis = res.index
                        ax.plot(x_axis, res[mean_col], '-o', label=algo)
                        ax.fill_between(x_axis, res[mean_col] - res[sem_col], res[mean_col] + res[sem_col], alpha=0.2)
                    elif metric in res.columns:
                        x_axis = res.index
                        ax.plot(x_axis, res[metric], '-o', label=algo)
            ax.set_title(metric); ax.set_xlabel('Number of Clusters (k)'); ax.legend()
        plt.suptitle('Reduced-Space Model Evaluation Metrics', y=1.02); plt.tight_layout(); plt.show()

    def plot_kprototypes_evaluation_metrics(self):
        """
        Plots the evaluation metrics specifically for the K-Prototypes algorithm.
        Handles both single-run and cross-sampled results with error bars.
        """
        print("\n--- Plotting K-Prototypes Evaluation Metrics ---")
        if self.results['kprototypes'].empty:
            print("K-Prototypes results not found. Please run the evaluation first.")
            return

        fig, axs = plt.subplots(1, 3, figsize=(24, 7), sharex=True)
        metrics = ['Cost', 'Silhouette', 'Dunn']
        res = self.results['kprototypes']

        for ax, metric in zip(axs, metrics):
            # Check for single run vs cross-sample run columns
            if f'{metric}_mean' in res.columns:
                mean_col, sem_col = f'{metric}_mean', f'{metric}_sem'
                x_axis = res.index
                ax.plot(x_axis, res[mean_col], '-o', color='purple')
                ax.fill_between(x_axis, res[mean_col] - res[sem_col], res[mean_col] + res[sem_col], alpha=0.2, color='purple')
                ax.set_title(f'Mean {metric} (± SEM)')
            elif metric in res.columns:
                x_axis = res.index
                ax.plot(x_axis, res[metric], '-o', color='purple')
                ax.set_title(metric)
            
            ax.set_xlabel('Number of Clusters (k)')
            ax.grid(True, linestyle='--', alpha=0.6)

        axs[0].set_ylabel('Score')
        plt.suptitle('K-Prototypes Model Evaluation Metrics', y=1.02)
        plt.tight_layout()
        plt.show()

    def plot_cluster_matrix(self, algorithm, palette='gnuplot'):
        print(f"\n--- Visualizing Cluster Matrix for {algorithm.upper()} ---")
        cluster_col = f'cluster_{"kprototypes" if algorithm == "kprototypes" else algorithm}'
        if cluster_col not in self.X_original_df.columns: print(f"Error: Final model for '{algorithm}' has not been run."); return

        if algorithm in ['kmeans', 'hierarchical', 'gmm']:
            plot_df = self.X_reduced_df.copy()
            if len(plot_df.columns) -1 < 2: print("At least two features are required for a pair plot."); return
            g = sns.pairplot(plot_df, hue=cluster_col, palette=palette, corner=True); g.fig.suptitle(f'{algorithm.upper()} Clusters in Reduced Space', y=1.02); plt.show()
        elif algorithm == 'kprototypes':
            print("K-Prototypes visualization uses FAMD for 2D representation.")
            famd = FAMD(n_components=2, random_state=self.random_state)
            features_for_famd = self.X_original_df.drop(columns=[c for c in self.X_original_df.columns if c.startswith('cluster_')])
            components = famd.fit_transform(features_for_famd); components['Cluster'] = self.X_original_df[cluster_col].values
            plt.figure(figsize=(10, 8)); sns.scatterplot(x=0, y=1, hue='Cluster', data=components, palette=palette, s=80, alpha=0.8)
            plt.title('K-Prototypes Clusters in 2D FAMD Space'); plt.xlabel('Component 1'); plt.ylabel('Component 2'); plt.grid(True); plt.show()

    def plot_cluster_sizes(self, palette='viridis'):
        print("\n--- Plotting Cluster Size Distributions ---")
        cluster_cols = [c for c in self.X_original_df if c.startswith('cluster_')]
        if not cluster_cols: print("No final models have been trained."); return
            
        fig, axs = plt.subplots(1, len(cluster_cols), figsize=(6 * len(cluster_cols), 6), sharey=True)
        if len(cluster_cols) == 1: axs = [axs]
            
        for ax, col in zip(axs, cluster_cols):
            algo_name = col.replace('cluster_', '').upper()
            counts = self.X_original_df[col].value_counts().sort_index()
            sns.barplot(x=counts.index, y=counts.values, palette=palette, ax=ax, hue=counts.index, legend=False)
            ax.set_title(f'{algo_name} Cluster Sizes'); ax.set_xlabel('Cluster ID')
        axs[0].set_ylabel('Number of Data Points'); plt.tight_layout(); plt.show()

    # ==============================================================================
    # SECTION 6: ALGORITHM AGREEMENT & STABILITY
    # ==============================================================================
    def evaluate_algorithm_agreement(self):
        print("\n--- Evaluating Algorithm Agreement ---")
        cluster_cols = [c for c in self.X_original_df.columns if c.startswith('cluster_')]
        if len(cluster_cols) < 2: print("Requires at least two models to be trained."); return
        names = [c.replace('cluster_', '').replace('kproto', 'kprototypes') for c in cluster_cols]
        matrix = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
        for i, j in combinations(range(len(cluster_cols)), 2):
            ari = adjusted_rand_score(self.X_original_df[cluster_cols[i]], self.X_original_df[cluster_cols[j]])
            matrix.iloc[i, j] = matrix.iloc[j, i] = ari
        plt.figure(figsize=(8, 6)); sns.heatmap(matrix, annot=True, cmap='viridis', fmt='.3f')
        plt.title('Clustering Algorithm Agreement (Adjusted Rand Index)'); plt.show(); return matrix
        
    def evaluate_stability_by_seed(self, k_dict, random_seeds):
        if len(random_seeds) < 2: raise ValueError("Requires at least two random seeds.")
        print(f"\n--- Evaluating Stability by Seed ({len(random_seeds)} seeds) ---")
        
        ari_scores = {algo: [] for algo in k_dict}
        seed_pairs = list(combinations(random_seeds, 2))
        
        for algo, k in k_dict.items():
            print(f"  Processing {algo.capitalize()}...")
            if algo == 'hierarchical':
                ari_scores[algo] = [1.0] * len(seed_pairs)
                continue

            def _get_labels_for_seed(seed):
                if algo == 'kprototypes':
                    cat_indices = [self.X_original_df.columns.get_loc(c) for c in self.categorical_cols_kproto]
                    m = KPrototypes(n_clusters=k, init='Cao', random_state=seed, n_jobs=1).fit(self.X_original_df.values, categorical=cat_indices)
                    return m.labels_
                elif algo == 'kmeans':
                    m = KMeans(n_clusters=k, n_init='auto', random_state=seed).fit(self.X_reduced_df)
                    return m.labels_
                elif algo == 'gmm':
                    m = GaussianMixture(n_components=k, random_state=seed).fit(self.X_reduced_df)
                    return m.predict(self.X_reduced_df)
                return None
            
            # Get all labels in parallel first
            all_labels = Parallel(n_jobs=self.n_jobs)(delayed(_get_labels_for_seed)(seed) for seed in random_seeds)
            seed_to_labels = dict(zip(random_seeds, all_labels))

            # Calculate ARI scores
            for seed1, seed2 in seed_pairs:
                ari_scores[algo].append(adjusted_rand_score(seed_to_labels[seed1], seed_to_labels[seed2]))
        
        plot_data = {f"{algo.capitalize()} (k={k_dict[algo]})": scores for algo, scores in ari_scores.items()}
        plt.figure(figsize=(10, 6)); sns.kdeplot(data=plot_data, fill=True, common_norm=False, palette="magma")
        plt.title('Initialization Stability by Seed (Pairwise ARI)'); plt.xlabel('Adjusted Rand Score (ARI)'); plt.xlim(-0.1, 1.1); plt.show()

    def evaluate_perturbation_stability(self, k_dict, n_iterations=30, noise_level=0.1):
        """
        Evaluates clustering stability by adding Gaussian noise to the reduced-space dataset.
        This method is not applicable to K-Prototypes. Stability is quantified using the
        Adjusted Rand Index (ARI) against a baseline clustering on the original data.

        Args:
            k_dict (dict): Maps algorithm names to the number of clusters (k) to use.
            n_iterations (int): The number of noise perturbation iterations to run.
            noise_level (float): The standard deviation of the Gaussian noise to add.
        """
        algos_to_run = [algo for algo in k_dict if algo != 'kprototypes']
        if not algos_to_run:
            print("Perturbation analysis only applies to numeric-space models (KMeans, GMM, Hierarchical).")
            return

        print(f"\n--- Evaluating Noise Perturbation Stability (Noise Level: {noise_level}) ---")
        X = self.X_reduced_df.values
        ari_scores = {algo: [] for algo in algos_to_run}
        
        # --- 1. Calculate Baselines on original data ---
        baselines = {}
        print("Calculating baseline clusterings on clean data...")
        for algo in algos_to_run:
            k = k_dict[algo]
            if algo == 'kmeans': model = KMeans(n_clusters=k, n_init='auto', random_state=self.random_state)
            elif algo == 'gmm': model = GaussianMixture(n_components=k, random_state=self.random_state)
            elif algo == 'hierarchical': model = AgglomerativeClustering(n_clusters=k, linkage=self.linkage)
            baselines[algo] = model.fit_predict(X)

        # --- 2. Run Perturbation Iterations in Parallel ---
        print(f"Running {n_iterations} noise perturbation iterations...")
        def _process_perturbation(iteration_seed):
            rng = np.random.default_rng(iteration_seed)
            noise = rng.normal(loc=0.0, scale=noise_level, size=X.shape)
            X_perturbed = X + noise
            
            iter_aris = {}
            for algo in algos_to_run:
                k = k_dict[algo]
                if algo == 'kmeans': model_p = KMeans(n_clusters=k, n_init='auto', random_state=iteration_seed)
                elif algo == 'gmm': model_p = GaussianMixture(n_components=k, random_state=iteration_seed)
                elif algo == 'hierarchical': model_p = AgglomerativeClustering(n_clusters=k, linkage=self.linkage)
                
                perturbed_labels = model_p.fit_predict(X_perturbed)
                iter_aris[algo] = adjusted_rand_score(baselines[algo], perturbed_labels)
            return iter_aris

        # Generate seeds for reproducibility in parallel processes
        seeds = [self.random_state + i for i in range(n_iterations)]
        results_list = Parallel(n_jobs=self.n_jobs)(delayed(_process_perturbation)(seed) for seed in seeds)

        # Collate results
        for result in results_list:
            for algo, score in result.items():
                ari_scores[algo].append(score)

        # --- 3. Plotting ---
        plot_data = {f"{algo.capitalize()} (k={k_dict[algo]})": scores for algo, scores in ari_scores.items()}
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=plot_data, fill=True, common_norm=False, palette="plasma")
        plt.title(f'Noise Perturbation Stability (Noise Level: {noise_level})')
        plt.xlabel('Adjusted Rand Score (ARI) vs. Baseline')
        plt.xlim(-0.1, 1.1)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
        
        return ari_scores

    def evaluate_processing_time(self, k_dict, n_steps=10, n_repeats=5):
        """
        Evaluates processing time vs. sample size with repeated trials for statistical robustness.
        Plots the mean processing time with standard deviation as error bars.

        Args:
            k_dict (dict): Maps algorithm names to the number of clusters (k) to use.
            n_steps (int): The number of sample size increments to test.
            n_repeats (int): The number of times to repeat the timing for each sample size.
        """
        print(f"\n--- Evaluating Processing Time ({n_repeats} repeats per step) ---")
        total_samples = len(self.X_original_df)
        sample_sizes = np.linspace(max(100, int(total_samples * 0.1)), total_samples, n_steps, dtype=int)
        
        results = []
        for n in sample_sizes:
            print(f"  Timing for n={n} samples...")
            for i in range(n_repeats):
                # Use a different random state for each repeat to vary the sample
                rng = np.random.default_rng(self.random_state + i)
                indices = rng.choice(total_samples, n, replace=False)
                
                # --- Time Reduced-Space Models ---
                sample_X_redu = self.X_reduced_df.iloc[indices]
                if 'kmeans' in k_dict:
                    start = time.perf_counter()
                    KMeans(n_clusters=k_dict['kmeans'], n_init='auto').fit(sample_X_redu)
                    results.append({'algo': 'KMeans', 'samples': n, 'time': time.perf_counter() - start})
                if 'hierarchical' in k_dict:
                    start = time.perf_counter()
                    AgglomerativeClustering(n_clusters=k_dict['hierarchical'], linkage=self.linkage).fit(sample_X_redu)
                    results.append({'algo': 'Hierarchical', 'samples': n, 'time': time.perf_counter() - start})
                if 'gmm' in k_dict:
                    start = time.perf_counter()
                    GaussianMixture(n_components=k_dict['gmm']).fit(sample_X_redu)
                    results.append({'algo': 'GMM', 'samples': n, 'time': time.perf_counter() - start})
                
                # --- Time K-Prototypes ---
                if 'kprototypes' in k_dict:
                    sample_X_orig = self.X_original_df.iloc[indices]
                    cat_indices = [sample_X_orig.columns.get_loc(c) for c in self.categorical_cols_kproto if c in sample_X_orig.columns]
                    start = time.perf_counter()
                    KPrototypes(n_clusters=k_dict['kprototypes'], init='Cao', n_init=3).fit(sample_X_orig.values, categorical=cat_indices)
                    results.append({'algo': 'K-Prototypes', 'samples': n, 'time': time.perf_counter() - start})

        timing_df = pd.DataFrame(results)
        plt.figure(figsize=(12, 7))
        sns.lineplot(data=timing_df, x='samples', y='time', hue='algo', marker='o', errorbar='sd')
        plt.title('Processing Time vs. Sample Size (Mean & Std Dev)')
        plt.xlabel('Number of Samples')
        plt.ylabel('Processing Time (seconds)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Algorithm')
        plt.show()
        
        return timing_df

    # ==============================================================================
    # SECTION 7: VARIABLE IMPORTANCE & INTERPRETATION
    # ==============================================================================
    def plot_decision_tree_importance(self, algorithm, max_depth=3):
        cluster_col = f'cluster_{"kprototypes" if algorithm == "kprototypes" else algorithm}'
        if cluster_col not in self.X_original_df.columns: print(f"Error: Final model for '{algorithm}' has not been run."); return
        print(f"\n--- Generating Decision Tree for {algorithm.upper()} Clusters (max_depth={max_depth}) ---")
        X_features = self.X_original_df.drop(columns=[c for c in self.X_original_df.columns if c.startswith('cluster_')])
        y_labels = self.X_original_df[cluster_col]
        X_processed = pd.get_dummies(X_features, drop_first=True)
        tree_classifier = DecisionTreeClassifier(max_depth=max_depth, random_state=self.random_state).fit(X_processed, y_labels)
        plt.figure(figsize=(25, 12))
        plot_tree(tree_classifier, feature_names=X_processed.columns.tolist(), class_names=[f'Cluster {i}' for i in sorted(y_labels.unique())], filled=True, rounded=True, fontsize=10, precision=2)
        plt.title(f'Decision Tree Explaining {algorithm.upper()} Clusters', fontsize=16); plt.show()

    def plot_permutation_importance(self, algorithm, n_repeats=10, score_metric='accuracy'):
        cluster_col = f'cluster_{"kprototypes" if algorithm == "kprototypes" else algorithm}'
        if cluster_col not in self.X_original_df.columns: print(f"Error: Final model for '{algorithm}' has not been run."); return
        print(f"\n--- Calculating Permutation Importance for {algorithm.upper()} ---")
        X_features = self.X_original_df.drop(columns=[c for c in self.X_original_df.columns if c.startswith('cluster_')])
        y_labels = self.X_original_df[cluster_col]
        X_processed = pd.get_dummies(X_features, drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y_labels, test_size=0.3, random_state=self.random_state, stratify=y_labels)
        print("Training Random Forest surrogate model...")
        surrogate_model = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=self.n_jobs).fit(X_train, y_train)
        print(f"Calculating importance based on '{score_metric}' drop...")
        result = permutation_importance(surrogate_model, X_test, y_test, n_repeats=n_repeats, random_state=self.random_state, scoring=score_metric, n_jobs=self.n_jobs)
        importances = pd.DataFrame({'feature': X_processed.columns.tolist(), 'importance_mean': result.importances_mean}).sort_values('importance_mean', ascending=False)
        plt.figure(figsize=(12, 8)); sns.barplot(x='importance_mean', y='feature', data=importances, palette='viridis', hue='feature', dodge=False)
        plt.xlabel(f'Mean Importance ({score_metric.capitalize()} Drop)'); plt.ylabel('Feature'); plt.title(f'Permutation Feature Importance for {algorithm.upper()} Clusters')
        plt.tight_layout(); plt.show()

    def calculate_cluster_statistics(self, algorithm, p_value_threshold=0.05):
        """
        Performs ANOVA (for continuous) and Chi-square (for categorical) tests to
        find features that are significantly different across clusters.
        """
        cluster_col = f'cluster_{"kprototypes" if algorithm == "kprototypes" else algorithm}'
        if cluster_col not in self.X_original_df.columns:
            print(f"Error: Final model for '{algorithm}' has not been run.")
            return None, None

        print(f"\n--- Calculating Cluster Statistics for {algorithm.upper()} ---")
        df_stats = self.X_original_df.copy()
        
        # --- ANOVA for Continuous Variables ---
        num_vars = df_stats.select_dtypes(include=np.number).columns.drop([c for c in df_stats.columns if c.startswith('cluster_')], errors='ignore').tolist()
        anova_results = []
        for var in num_vars:
            groups = [df_stats[df_stats[cluster_col] == c][var].dropna() for c in df_stats[cluster_col].unique()]
            if len(groups) > 1 and all(len(g) > 1 for g in groups):
                f_stat, p_val = stats.f_oneway(*groups)
                anova_results.append({'Variable': var, 'F-statistic': f_stat, 'P-value': p_val,
                                    'Significant': p_val < p_value_threshold})
        
        # --- Chi-Square for Categorical Variables ---
        cat_vars = df_stats.select_dtypes(exclude=np.number).columns.tolist()
        chi2_results = []
        for var in cat_vars:
            contingency_table = pd.crosstab(df_stats[var], df_stats[cluster_col])
            if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                chi2, p_val, _, _ = stats.chi2_contingency(contingency_table)
                chi2_results.append({'Variable': var, 'Chi-Square': chi2, 'P-value': p_val,
                                    'Significant': p_val < p_value_threshold})

        return (pd.DataFrame(anova_results).set_index('Variable'), 
                pd.DataFrame(chi2_results).set_index('Variable'))

    def calculate_effect_sizes(self, algorithm):
        """
        Calculates effect sizes (Cohen's d and Cramér's V) to measure the
        magnitude of differences between clusters for each feature.
        """
        cluster_col = f'cluster_{"kprototypes" if algorithm == "kprototypes" else algorithm}'
        if cluster_col not in self.X_original_df.columns:
            print(f"Error: Final model for '{algorithm}' has not been run.")
            return None, None

        print(f"\n--- Calculating Effect Sizes for {algorithm.upper()} ---")
        df_full = self.X_original_df.copy()

        # --- Cohen's d for Continuous Variables ---
        num_vars = df_full.select_dtypes(include=np.number).columns.drop([c for c in df_full.columns if c.startswith('cluster_')], errors='ignore').tolist()
        cohens_d_results = []
        cluster_ids = sorted(df_full[cluster_col].unique())
        for var in num_vars:
            for c1, c2 in combinations(cluster_ids, 2):
                group1 = df_full[df_full[cluster_col] == c1][var].dropna()
                group2 = df_full[df_full[cluster_col] == c2][var].dropna()
                n1, n2 = len(group1), len(group2)
                if n1 > 1 and n2 > 1:
                    s1, s2 = group1.std(), group2.std()
                    # Check for zero standard deviation
                    if s1 > 0 and s2 > 0:
                        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
                        d = (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0.0
                        cohens_d_results.append({'Variable': var, 'Comparison': f'C{c1} vs C{c2}', "Cohen's d": d})
        
        # --- Cramér's V for Categorical Variables ---
        cat_vars = df_full.select_dtypes(exclude=np.number).columns.tolist()
        cramers_v_results = []
        for var in cat_vars:
            contingency_table = pd.crosstab(df_full[var], df_full[cluster_col])
            if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                chi2, _, _, _ = stats.chi2_contingency(contingency_table)
                n = contingency_table.sum().sum()
                phi2 = chi2 / n
                r, k = contingency_table.shape
                v = np.sqrt(phi2 / min(k - 1, r - 1))
                cramers_v_results.append({'Variable': var, "Cramér's V": v})

        return (pd.DataFrame(cohens_d_results).set_index(['Variable', 'Comparison']),
                pd.DataFrame(cramers_v_results).set_index('Variable'))

