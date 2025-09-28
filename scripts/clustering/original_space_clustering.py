import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gower
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score
from validclust import dunn 
from prince import FAMD 
from sklearn.tree import DecisionTreeClassifier, plot_tree
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from itertools import combinations
import scipy.stats as stats

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import normalized_mutual_info_score

class FeatureSelector:
    """
    A class for analyzing, visualizing, and selecting features from a 
    DataFrame with mixed numerical and categorical data.
    """
    def __init__(self, df, random_state=42):
        """
        Initializes the FeatureSelector.

        Args:
            df (pd.DataFrame): The input dataframe with mixed data types.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")
            
        self._original_df = df.copy()
        self.df = df.copy()
        self.random_state = random_state
        self.numerical_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(exclude=np.number).columns.tolist()
        
        print("--- FeatureSelector Initialized ---")
        print(f"Detected {len(self.numerical_cols)} numerical columns: {self.numerical_cols}")
        print(f"Detected {len(self.categorical_cols)} categorical columns: {self.categorical_cols}")
    
    def stratified_sample(self, strata_col, sample_frac):
        """
        Performs stratified sampling on the dataset.
        """
        if strata_col not in self._original_df.columns:
            print(f"Error: Stratification column '{strata_col}' not found.")
            return

        print(f"Performing stratified sampling on '{strata_col}' with a {sample_frac:.2%} fraction...")
        df_sampled = self._original_df.groupby(strata_col, group_keys=False).apply(
            lambda x: x.sample(frac=sample_frac, random_state=self.random_state),
            include_groups=False
        )
        original_rows = len(self.df)
        self.df = df_sampled
        new_rows = len(self.df)
        if strata_col in self.numerical_cols:
            self.numerical_cols.remove(strata_col)
        if strata_col in self.categorical_cols:
            self.categorical_cols.remove(strata_col)
        print(f"Sampling complete. Active dataset reduced from {original_rows} to {new_rows} rows.")
        return self.df

    def _calculate_normalized_entropy(self, series):
        """
        Calculates the normalized entropy of a categorical variable.
        Normalized entropy is a value between 0 (no information) and 1 (maximum information).
        """
        # Drop NaNs for this calculation
        series = series.dropna()
        if series.empty:
            return 0.0
            
        value_counts = series.value_counts()
        probs = value_counts / len(series)
        n_classes = len(probs)
        
        # Entropy is 0 if there's only one unique category or no data
        if n_classes <= 1:
            return 0.0
            
        # Standard entropy calculation: H(X) = -Σ p(x) * log2(p(x))
        entropy = -np.sum(probs * np.log2(probs))
        
        # Normalize by the maximum possible entropy for k classes, which is log2(k)
        normalized_entropy = entropy / np.log2(n_classes)
        return normalized_entropy

    def analyze_features(self):
        """
        Calculates and visualizes key metrics for feature selection without
        modifying the dataframe. Helps in deciding thresholds.
        """
        print("\n--- Starting Feature Analysis ---")
        
        # Determine subplot layout
        has_num = bool(self.numerical_cols)
        has_cat = bool(self.categorical_cols)
        
        if not has_num and not has_cat:
            print("No features to analyze.")
            return
            
        fig, axes = plt.subplots(
            nrows=2 if has_num and has_cat else 1, 
            ncols=2, 
            figsize=(18, 14 if has_num and has_cat else 7)
        )
        
        # --- Numerical Analysis ---
        if has_num:
            ax_row = axes[0] if has_num and has_cat else axes
            
            # Coefficient of Variation
            print("Calculating Coefficient of Variation for numerical features...")
            means = self.df[self.numerical_cols].mean().abs()
            stds = self.df[self.numerical_cols].std()
            cv = (stds / means.replace(0, np.nan)).fillna(0).sort_values(ascending=False)
            
            sns.barplot(x=cv.values, y=cv.index, ax=ax_row[0], hue = cv.index, palette='viridis', legend=False)
            ax_row[0].set_title('Coefficient of Variation (Numerical)')
            ax_row[0].set_xlabel('CV (Std Dev / Mean)')
            
            # Correlation Matrix
            print("Calculating Correlation Matrix for numerical features...")
            corr_matrix = self.df[self.numerical_cols].corr()
            sns.heatmap(corr_matrix, ax=ax_row[1], cmap='coolwarm', annot=False)
            ax_row[1].set_title('Correlation Matrix (Numerical)')

        # --- Categorical Analysis ---
        if has_cat:
            ax_row = axes[1] if has_num and has_cat else axes
            
            # Normalized Entropy
            print("Calculating Normalized Entropy for categorical features...")
            entropies = {col: self._calculate_normalized_entropy(self.df[col]) for col in self.categorical_cols}
            ent_series = pd.Series(entropies).sort_values(ascending=False)
            
            sns.barplot(x=ent_series.values, y=ent_series.index, ax=ax_row[0], hue=ent_series.index, palette='plasma', legend=False)
            ax_row[0].set_title('Normalized Entropy (Categorical)')
            ax_row[0].set_xlabel('Normalized Entropy')
            
            # Normalized Mutual Information
            print("Calculating Normalized Mutual Information for categorical features...")
            n = len(self.categorical_cols)
            nmi_matrix = pd.DataFrame(np.eye(n), index=self.categorical_cols, columns=self.categorical_cols)
            for i in range(n):
                for j in range(i + 1, n):
                    nmi_score = normalized_mutual_info_score(self.df[self.categorical_cols[i]], self.df[self.categorical_cols[j]])
                    nmi_matrix.iloc[i, j] = nmi_matrix.iloc[j, i] = nmi_score
            
            sns.heatmap(nmi_matrix, ax=ax_row[1], cmap='magma', annot=False)
            ax_row[1].set_title('Normalized Mutual Information (Categorical)')

        plt.suptitle('Feature Analysis Dashboard', fontsize=20)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.show()

    def select_features(
        self,
        cv_threshold=None,
        correlation_threshold=None,
        entropy_threshold=None,
        mutual_info_threshold=None
    ):
        """
        Performs feature selection based on provided thresholds and returns
        a new DataFrame with the selected columns removed.

        Args:
            cv_threshold (float, optional): 
                Drops numerical columns with a Coefficient of Variation below this value.
            correlation_threshold (float, optional): 
                For numerical pairs with correlation above this, drops the one with the
                higher average correlation to all other numerical variables.
            entropy_threshold (float, optional): 
                Drops categorical columns with normalized entropy below this value.
            mutual_info_threshold (float, optional): 
                For categorical pairs with NMI above this, drops the one with the
                higher average NMI to all other categorical variables.
        
        Returns:
            pd.DataFrame: A new DataFrame with the low-value features removed.
        """
        print("\n--- Starting Feature Selection Process ---")
        
        # Work on a copy of the original dataframe
        df_selected = self.df.copy()
        cols_to_drop = set()

        # Part 1: Numerical Feature Selection
        if self.numerical_cols:
            print(f"\nAnalyzing {len(self.numerical_cols)} numerical features...")
            if cv_threshold is not None:
                print(f"Checking for Coefficient of Variation below {cv_threshold}...")
                means = df_selected[self.numerical_cols].mean().abs()
                stds = df_selected[self.numerical_cols].std()
                cv = stds / means.replace(0, np.nan)
                low_cv_cols = cv[cv < cv_threshold].index.tolist()
                
                if low_cv_cols:
                    print(f"  -> Found {len(low_cv_cols)} low-CV columns to drop: {low_cv_cols}")
                    cols_to_drop.update(low_cv_cols)
                else:
                    print("  -> No low-CV columns found.")

            if correlation_threshold is not None:
                print(f"Checking for Pearson correlations above {correlation_threshold}...")
                remaining_num_cols = [c for c in self.numerical_cols if c not in cols_to_drop]
                corr_matrix = df_selected[remaining_num_cols].corr().abs()
                
                high_corr_cols_to_drop = set()
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i):
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        if corr_matrix.iloc[i, j] > correlation_threshold:
                            mean_corr_col1 = corr_matrix[col1].mean()
                            mean_corr_col2 = corr_matrix[col2].mean()
                            
                            if mean_corr_col1 > mean_corr_col2:
                                high_corr_cols_to_drop.add(col1)
                            else:
                                high_corr_cols_to_drop.add(col2)
                
                if high_corr_cols_to_drop:
                    print(f"  -> Found {len(high_corr_cols_to_drop)} highly-correlated columns to drop: {list(high_corr_cols_to_drop)}")
                    cols_to_drop.update(high_corr_cols_to_drop)
                else:
                    print("  -> No highly correlated column pairs found.")

        # Part 2: Categorical Feature Selection
        if self.categorical_cols:
            if entropy_threshold is not None:
                print(f"\nChecking for normalized entropy below {entropy_threshold}...")
                low_entropy_cols = [col for col in self.categorical_cols if self._calculate_normalized_entropy(df_selected[col]) < entropy_threshold]
                
                if low_entropy_cols:
                    print(f"  -> Found {len(low_entropy_cols)} low-entropy columns to drop: {low_entropy_cols}")
                    cols_to_drop.update(low_entropy_cols)
                else:
                    print("  -> No low-entropy columns found.")

            if mutual_info_threshold is not None:
                print(f"Checking for normalized mutual information above {mutual_info_threshold}...")
                remaining_cat_cols = [c for c in self.categorical_cols if c not in cols_to_drop]
                n = len(remaining_cat_cols)
                
                nmi_matrix = pd.DataFrame(np.eye(n), index=remaining_cat_cols, columns=remaining_cat_cols)
                for i in range(n):
                    for j in range(i + 1, n):
                        nmi_score = normalized_mutual_info_score(df_selected[remaining_cat_cols[i]], df_selected[remaining_cat_cols[j]])
                        nmi_matrix.iloc[i, j] = nmi_matrix.iloc[j, i] = nmi_score

                high_mi_cols_to_drop = set()
                
                for i in range(n):
                    for j in range(i):
                        col1, col2 = nmi_matrix.columns[i], nmi_matrix.columns[j]
                        if nmi_matrix.iloc[i, j] > mutual_info_threshold:
                            mean_nmi_col1 = nmi_matrix[col1].mean()
                            mean_nmi_col2 = nmi_matrix[col2].mean()
                            
                            if mean_nmi_col1 > mean_nmi_col2:
                                high_mi_cols_to_drop.add(col1)
                            else:
                                high_mi_cols_to_drop.add(col2)

                if high_mi_cols_to_drop:
                    print(f"  -> Found {len(high_mi_cols_to_drop)} high-NMI columns to drop: {list(high_mi_cols_to_drop)}")
                    cols_to_drop.update(high_mi_cols_to_drop)
                else:
                    print("  -> No high mutual information column pairs found.")

        # Part 3: Drop selected columns and return the new DataFrame
        if cols_to_drop:
            print(f"\nDropping a total of {len(cols_to_drop)} unique columns: {list(cols_to_drop)}")
            df_selected.drop(columns=list(cols_to_drop), inplace=True)
            print("Feature selection complete.")
        else:
            print("\nNo columns were dropped based on the provided thresholds.")
        
        print("\n--- Feature Selection Finished ---")
        return df_selected

class KPrototypesEvaluator:
    """
    A class for performing, evaluating, and visualizing K-Prototypes clustering
    on mixed numerical and categorical data.
    """
    def __init__(self, df, k_range, original_df_full=None,gamma_list=None, random_state=42):
        """
        Initializes the KPrototypesEvaluator.

        Args:
            df (pd.DataFrame): The input dataframe with mixed data types.
            k_range (range): A range of k values to test (e.g., range(2, 11)).
            original_df_full (pd.DataFrame, optional): The complete original dataframe with all features,
                                                    before any feature selection. If None, it's assumed
                                                    that `df` contains all features. Defaults to None.
            gamma_list (list, optional): A list of gamma values to test (e.g., [0.1, 0.5, 1, 2]). 
                                    If None, uses default gamma from KPrototypes.
                                    
            random_state (int): Seed for reproducibility.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")
            
        self._original_df = df.copy()
        self.df = df.copy()
        self.k_range = k_range
        self.gamma_list = gamma_list
        self.random_state = random_state

        if original_df_full is not None:
            self.full_df = original_df_full.copy()
            print("Full original dataframe loaded for statistical analysis.")
        else:
            self.full_df = self._original_df.copy()
            print("Assuming input dataframe contains all features for statistical analysis.")

        self.numerical_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(exclude=np.number).columns.tolist()
        
        print(f"Detected {len(self.numerical_cols)} numerical columns: {self.numerical_cols}")
        print(f"Detected {len(self.categorical_cols)} categorical columns: {self.categorical_cols}")

        self.results = pd.DataFrame()
        self.final_model = None
        self.scaler = None
        self.gower_matrix = None
        self.df_unscaled = None

    # ==============================================================================
    # DATA PREPARATION
    # ==============================================================================
    def stratified_sample(self, strata_col, sample_frac):
        """
        Performs stratified sampling on the dataset.
        """
        if strata_col not in self._original_df.columns:
            print(f"Error: Stratification column '{strata_col}' not found.")
            return

        print(f"Performing stratified sampling on '{strata_col}' with a {sample_frac:.2%} fraction...")
        df_sampled = self._original_df.groupby(strata_col, group_keys=False).apply(
            lambda x: x.sample(frac=sample_frac, random_state=self.random_state),
            include_groups=False
        )
        original_rows = len(self.df)
        self.df = df_sampled
        new_rows = len(self.df)
        if strata_col in self.numerical_cols:
            self.numerical_cols.remove(strata_col)
        if strata_col in self.categorical_cols:
            self.categorical_cols.remove(strata_col)
        print(f"Sampling complete. Active dataset reduced from {original_rows} to {new_rows} rows.")
        return self.df
    
    def scale_numerical_features(self):
        """
        Scales only the numerical feature columns using StandardScaler.
        """
        if not self.numerical_cols:
            print("No numerical columns to scale.")
            return self.df

        print(f"Scaling {len(self.numerical_cols)} numerical features...")

        self.df_unscaled = self.df.copy() 

        self.scaler = StandardScaler()
        self.df[self.numerical_cols] = self.scaler.fit_transform(self.df[self.numerical_cols])
        
        print("Scaling complete. Active dataset has been updated.")
        return self.df
    
    def _calculate_normalized_entropy(self, series):
        """
        Calculates the normalized entropy of a categorical variable.
        Normalized entropy is a value between 0 (no information) and 1 (maximum information).
        """
        # Drop NaNs for this calculation
        series = series.dropna()
        if series.empty:
            return 0.0
            
        value_counts = series.value_counts()
        probs = value_counts / len(series)
        n_classes = len(probs)
        
        # Entropy is 0 if there's only one unique category or no data
        if n_classes <= 1:
            return 0.0
            
        # Standard entropy calculation: H(X) = -Σ p(x) * log2(p(x))
        entropy = -np.sum(probs * np.log2(probs))
        
        # Normalize by the maximum possible entropy for k classes, which is log2(k)
        normalized_entropy = entropy / np.log2(n_classes)
        return normalized_entropy

    # ==============================================================================
    # CLUSTERING & EVALUATION
    # ==============================================================================
    def run_and_evaluate(self):
        """
        Runs the K-Prototypes algorithm for each k in k_range and gamma in gamma_list,
        evaluates all combinations, and stores the results.
        """
        # Determine gamma values to test
        if self.gamma_list is None:
            print(f"\nRunning K-Prototypes for k in {list(self.k_range)} with default gamma...")
            gamma_values = [None]  # Use default gamma
        else:
            print(f"\nRunning K-Prototypes for k in {list(self.k_range)} and gamma in {self.gamma_list}...")
            gamma_values = self.gamma_list
        
        categorical_indices = [self.df.columns.get_loc(col) for col in self.categorical_cols]
        
        print("Calculating Gower distance matrix for evaluation metrics...")
        self.gower_matrix = gower.gower_matrix(self.df)
        print("Gower matrix calculation complete.")
        
        results_list = []
        
        for gamma in gamma_values:
            for k in self.k_range:
                gamma_str = f"{gamma:.2f}" if gamma is not None else "default"
                print(f"  - Testing k={k}, gamma={gamma_str}...")
                
                # Create model with or without explicit gamma
                if gamma is not None:
                    model = KPrototypes(
                        n_clusters=k,
                        gamma=gamma,
                        init='Cao',
                        n_init=10,
                        random_state=self.random_state
                    )
                else:
                    model = KPrototypes(
                        n_clusters=k, 
                        init='Cao',
                        n_init=10,
                        random_state=self.random_state
                    )
                
                labels = model.fit_predict(self.df.values, categorical=categorical_indices)
                cost = model.cost_
                
                sil_score = silhouette_score(self.gower_matrix, labels, metric='precomputed')
                dunn_score = dunn(self.gower_matrix, labels)
                
                results_list.append({
                    'k': k,
                    'gamma': gamma if gamma is not None else model.gamma,
                    'Cost': cost,
                    'Silhouette Score': sil_score,
                    'Dunn Index': dunn_score
                })
        
        self.results = pd.DataFrame(results_list)
        print("\nEvaluation complete.")
        return self.results

    def run_final_model(self, optimal_k, optimal_gamma=None):
        """
        Runs the final K-Prototypes model with the chosen k and gamma.
        """
        if optimal_gamma is not None:
            print(f"\nRunning final K-Prototypes model with k={optimal_k}, gamma={optimal_gamma}...")
        else:
            print(f"\nRunning final K-Prototypes model with k={optimal_k} and default gamma...")
        
        categorical_indices = [self.df.columns.get_loc(col) for col in self.categorical_cols]
        
        if optimal_gamma is not None:
            model = KPrototypes(
                n_clusters=optimal_k,
                gamma=optimal_gamma,
                init='Cao', 
                n_init=10, 
                random_state=self.random_state
            )
        else:
            model = KPrototypes(
                n_clusters=optimal_k, 
                init='Cao', 
                n_init=10, 
                random_state=self.random_state
            )

        model.fit(self.df.values, categorical=categorical_indices)
        self.final_model = model
        self.df['cluster_kproto'] = model.labels_
        print(f'Final Gamma used: {model.gamma}')
        
        print("Final model is trained and cluster labels are added to the DataFrame.")
        return self.df
    
    # ==============================================================================
    # VISUALIZATION
    # ==============================================================================
    def plot_evaluation_metrics(self):
        """
        Plots the evaluation metrics (Cost, Silhouette, Dunn Index) against k,
        with separate lines for different gamma values if gamma tuning was performed.
        """
        if self.results.empty:
            print("Please run `run_and_evaluate()` first.")
            return

        metrics = ['Cost', 'Silhouette Score', 'Dunn Index']
        
        # Check if gamma tuning was performed
        unique_gammas = self.results['gamma'].unique()
        has_multiple_gammas = len(unique_gammas) > 1
        
        if has_multiple_gammas:
            print(f"Plotting metrics for {len(unique_gammas)} gamma values: {sorted(unique_gammas)}")
            fig, axs = plt.subplots(1, 3, figsize=(24, 7))
            palette = sns.color_palette("husl", len(unique_gammas))
            
            for ax, metric in zip(axs, metrics):
                sns.lineplot(
                    data=self.results,
                    x='k',
                    y=metric,
                    hue='gamma',
                    palette=palette,
                    ax=ax
                )
                ax.set_xlabel('Number of Clusters (k)')
                ax.set_ylabel(metric)
                ax.set_title(f'{metric} vs. k (Multiple Gamma Values)')
                ax.grid(True)
                ax.legend(title='Gamma (γ)')
            
            fig.suptitle('K-Prototypes Evaluation Metrics with Gamma Tuning', fontsize=18)
        else:
            print("Plotting metrics for single gamma value")
            fig, axs = plt.subplots(1, 3, figsize=(20, 6))
            
            for ax, metric in zip(axs, metrics):
                ax.plot(self.results['k'], self.results[metric], '-o', label='K-Prototypes')
                ax.set_xlabel('Number of Clusters (k)')
                ax.set_ylabel(metric)
                ax.set_title(f'{metric} vs. k')
                ax.grid(True)
                ax.legend()
            
            fig.suptitle('K-Prototypes Evaluation Metrics', fontsize=16)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_clusters_2d(self, palette='gnuplot'):
        """
        Reduces the mixed data to 2D using FAMD and plots the clusters.
        """
        if self.final_model is None or 'cluster_kproto' not in self.df.columns:
            print("Please run `run_final_model()` before plotting clusters.")
            return

        print("\n--- Generating 2D Cluster Plot using FAMD on all features ---")

        # Use the unscaled dataframe if it exists, otherwise use the current df
        if self.df_unscaled is not None:
            print("Using the pre-scaled dataframe for FAMD.")
            df_for_famd = self.df_unscaled.copy()
        else:
            print("No pre-scaled dataframe found. Using the current dataframe for FAMD.")
            df_for_famd = self.df.drop('cluster_kproto', axis=1, errors='ignore')

        famd = FAMD(n_components=2, random_state=self.random_state)
        components = famd.fit_transform(df_for_famd)
        components.columns = ['Component 1','Component 2']

        # Then add cluster labels with proper alignment
        cluster_labels = self.df['cluster_kproto']
        components['Cluster'] = cluster_labels

        print("FAMD components generated. Plotting now...")

        # Plotting logic remains the same
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x='Component 1', 
            y='Component 2', 
            hue='Cluster', 
            data=components,
            palette=palette,
            s=80,
            alpha=0.8
        )
        
        k = self.final_model.n_clusters
        plt.title(f'K-Prototypes Clusters (k={k}) in 2D FAMD Space', fontsize=16)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True)
        plt.legend(title='Cluster')
        plt.show()

    def plot_cluster_sizes(self, palette='viridis'):
        """
        Plots a bar chart showing the number of members in each cluster.
        Checks if the final model has been run before proceeding.
        """
        if self.final_model is None or 'cluster_kproto' not in self.df.columns:
            print("Please run `run_final_model()` before plotting cluster sizes.")
            return

        print("\n--- Generating Cluster Size Plot ---")

        # Get the cluster counts and sort by cluster ID for a clean plot
        cluster_counts = self.df['cluster_kproto'].value_counts().sort_index()

        plt.figure(figsize=(12, 7))
        ax = sns.barplot(
            x=cluster_counts.index, 
            y=cluster_counts.values, 
            palette=palette,
            hue=cluster_counts.index, # Use hue to ensure distinct colors per bar
            legend=False
        )

        ax.set_title('Cluster Size Distribution', fontsize=16)
        ax.set_xlabel('Cluster ID', fontsize=12)
        ax.set_ylabel('Number of Data Points (Size)', fontsize=12)
        
        # Add data labels on top of the bars for clarity
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        fontsize=11,
                        xytext=(0, 9),
                        textcoords='offset points')

        plt.ylim(0, cluster_counts.max() * 1.15) # Give some space for labels
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    # ==============================================================================
    # EXTERNAL VALIDITY & MODEL COMPARISON
    # ==============================================================================
    def cross_predict_and_evaluate_kproto(self, target_evaluator, name_source='Source', name_target='Target'):
        """
        Evaluates how well this instance's (source) final model can predict the cluster 
        assignments of another KPrototypesEvaluator instance (target).

        This is a measure of external validity: if the source model discovers a structure
        that is meaningful, it should be able to replicate the target's clustering
        with some degree of success, as measured by the Adjusted Rand Index (ARI).

        Args:
            target_evaluator (KPrototypesEvaluator): Another trained instance of the evaluator.
            name_source (str): A descriptive name for this (the source) model.
            name_target (str): A descriptive name for the target model.

        Returns:
            float: The Adjusted Rand Score (ARI), or None if evaluation fails.
        """
        print(f"\n--- Cross-Model Evaluation: '{name_source}' Model -> '{name_target}' Data ---")

        # 1. Check if both models are trained
        if self.final_model is None:
            print(f"[ERROR] The source model ('{name_source}') has not been trained. Please run `run_final_model()` first.")
            return None
        if target_evaluator.final_model is None:
            print(f"[ERROR] The target evaluator ('{name_target}') does not have a trained model.")
            return None

        # 2. Get the necessary data and feature names
        source_model = self.final_model
        # The features used to train the source model
        source_features = self.df.drop('cluster_kproto', axis=1, errors='ignore').columns.tolist()
        
        target_df = target_evaluator.df.copy()
        
        # 3. Align features: ensure the target data has the columns the source model expects
        if not set(source_features).issubset(target_df.columns):
            print(f"[ERROR] Target data is missing features required by the source model.")
            missing_cols = set(source_features) - set(target_df.columns)
            print(f"       Missing features: {list(missing_cols)}")
            return None

        # Prepare the target data for prediction, using only the features from the source model
        target_data_for_prediction = target_df[source_features]
        
        # Get the original cluster labels from the target
        original_target_labels = target_df['cluster_kproto']

        # 4. Predict on the target data using the source model
        # We need to find the categorical indices for the aligned target data
        source_cat_cols = self.categorical_cols
        categorical_indices = [
            target_data_for_prediction.columns.get_loc(col) 
            for col in source_cat_cols 
            if col in target_data_for_prediction.columns
        ]

        print(f"Predicting labels on '{name_target}' data using '{name_source}' model...")
        predicted_labels = source_model.predict(target_data_for_prediction.values, categorical=categorical_indices)

        # 5. Calculate and report the ARI score
        ari = adjusted_rand_score(original_target_labels, predicted_labels)
        print(f"✅ Adjusted Rand Score (ARI): {ari:.4f}")
        
        return ari

    def plot_cross_prediction_comparison(self, target_evaluator, name_source='Source', name_target='Target'):
        """
        Visually compares the target's original clusters with the clusters predicted
        by this instance's (source) model using FAMD for dimensionality reduction.

        Args:
            target_evaluator (KPrototypesEvaluator): Another trained instance of the evaluator.
            name_source (str): A descriptive name for this (the source) model.
            name_target (str): A descriptive name for the target model.
        """
        print(f"\n--- Visualizing Cross-Prediction: '{name_source}' Model vs. '{name_target}' Clusters ---")

        # Basic checks, similar to the evaluation method
        if self.final_model is None or target_evaluator.final_model is None:
            print("[ERROR] Both source and target models must be trained first.")
            return
            
        source_features = self.df.drop('cluster_kproto', axis=1, errors='ignore').columns.tolist()
        target_df = target_evaluator.df.copy()

        if not set(source_features).issubset(target_df.columns):
            print(f"[ERROR] Target data is missing features required by the source model.")
            return

        # Perform prediction (this logic is repeated from the method above)
        target_data_for_prediction = target_df[source_features]
        categorical_indices = [
            target_data_for_prediction.columns.get_loc(col) 
            for col in self.categorical_cols if col in target_data_for_prediction.columns
        ]
        predicted_labels = self.final_model.predict(target_data_for_prediction.values, categorical=categorical_indices)
        
        # Use the unscaled version of the target data for a more interpretable FAMD plot
        if target_evaluator.df_unscaled is not None:
            print("Using the target's pre-scaled data for FAMD visualization.")
            df_for_famd = target_evaluator.df_unscaled[source_features]
        else:
            df_for_famd = target_data_for_prediction

        # Reduce dimensionality of the target data to 2D using FAMD
        print("Reducing target data to 2D using FAMD...")
        famd = FAMD(n_components=2, random_state=self.random_state)
        components = famd.fit_transform(df_for_famd)
        
        components.columns = ['Component 1','Component 2']
        components['Original Cluster'] = target_df['cluster_kproto'].values
        components['Predicted Cluster'] = predicted_labels
        
        # Create side-by-side plots
        fig, axs = plt.subplots(1, 2, figsize=(18, 8), sharex=True, sharey=True)
        fig.suptitle(f"Cross-Prediction Visualization: K-Prototypes", fontsize=18)

        # Left Plot: Original Clusters from the target model
        sns.scatterplot(ax=axs[0], data=components, x='Component 1', y='Component 2', hue='Original Cluster', palette='viridis', s=60, alpha=0.8)
        axs[0].set_title(f"Original Clusters ({name_target})", fontsize=14)
        axs[0].legend(title='Original Cluster')
        axs[0].grid(True, linestyle='--', alpha=0.6)

        # Right Plot: Predicted Clusters from the source model
        sns.scatterplot(ax=axs[1], data=components, x='Component 1', y='Component 2', hue='Predicted Cluster', palette='viridis', s=60, alpha=0.8)
        axs[1].set_title(f"Clusters Predicted by {name_source} Model", fontsize=14)
        axs[1].legend(title='Predicted Cluster')
        axs[1].grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
        plt.show()

    # ==============================================================================
    # STABILITY AND SCALABILITY
    # ==============================================================================
    def evaluate_cross_sample_stability(self, strata_col, sample_frac, optimal_k, optimal_gamma=None, n_iterations=50):
        """
        Evaluates model stability by training on new stratified samples from the
        original full dataset and predicting on the current (active) sample.

        This method assesses if the clustering structure is robust to the initial
        stratified sampling process itself. It uses the Adjusted Rand Index (ARI)
        to measure stability.

        Args:
            strata_col (str): The column in the original, full dataframe to stratify on.
            sample_frac (float): The fraction of data to draw for each new sample.
            optimal_k (int): The number of clusters (k) for the K-Prototypes model.
            optimal_gamma (float, optional): The gamma value for K-Prototypes. Defaults to None.
            n_iterations (int): The number of new stratified samples to generate and test.

        Returns:
            list: A list containing the ARI scores for each iteration.
        """
        # --- 1. Input Validation and Setup ---
        if not hasattr(self, '_original_df'):
            print("[ERROR] Original dataframe `_original_df` not found. Stratified sampling must have been run.")
            return

        print(f"\n--- Evaluating Cross-Sample Stability ({n_iterations} iterations) ---")
        ari_scores = []
        
        # --- 2. Establish Baseline Clustering on the Current Active Sample ---
        print("Calculating baseline clustering on the current active sample...")
        
        if self.final_model and self.final_model.n_clusters == optimal_k:
            print("Using existing final model as baseline.")
            baseline_labels = self.final_model.labels_
        else:
            print("Final model not found or k differs. Fitting a new baseline model.")
            categorical_indices_baseline = [self.df.columns.get_loc(col) for col in self.categorical_cols]
            baseline_model = KPrototypes(
                n_clusters=optimal_k, 
                gamma=optimal_gamma, 
                init='Cao', 
                n_init=10, 
                random_state=self.random_state
            )
            baseline_labels = baseline_model.fit_predict(self.df.values, categorical=categorical_indices_baseline)

        baseline_data_for_prediction = self.df.drop('cluster_kproto', axis=1, errors='ignore')

        # --- 3. Loop, Create New Samples, Train, Predict, and Compare ---
        print(f"Beginning {n_iterations} cross-sample validation iterations...")
        for i in range(n_iterations):
            if (i + 1) % 10 == 0:
                print(f"  Iteration {i + 1}/{n_iterations}...")

            # a) Draw a NEW stratified sample from the full original dataset
            new_sample_df = self._original_df.groupby(strata_col, group_keys=False).apply(
                lambda x: x.sample(frac=sample_frac, random_state=self.random_state + i),
                include_groups=False
            )
            
            # b) Preprocess the new sample just like the original active data
            if self.scaler:
                new_sample_df[self.numerical_cols] = self.scaler.transform(new_sample_df[self.numerical_cols])
            
            # Ensure the new sample has the same columns as the baseline data
            new_sample_df = new_sample_df[baseline_data_for_prediction.columns]
            categorical_indices_new = [new_sample_df.columns.get_loc(col) for col in self.categorical_cols]

            # c) Train a new model on this new sample
            model_new = KPrototypes(
                n_clusters=optimal_k, 
                gamma=optimal_gamma,
                init='Cao', 
                n_init=10, 
                random_state=self.random_state + i
            )
            model_new.fit(new_sample_df.values, categorical=categorical_indices_new)
            
            # d) Predict on the original baseline data
            predicted_labels = model_new.predict(baseline_data_for_prediction.values, categorical=categorical_indices_new)
            
            # e) Calculate ARI and store it
            ari = adjusted_rand_score(baseline_labels, predicted_labels)
            ari_scores.append(ari)

        # --- 4. Report Results and Plot ---
        print("\nCross-sample stability evaluation complete.")
        print(f"K-Prototypes (k={optimal_k}) Stability: Mean ARI = {np.mean(ari_scores):.4f}, Std Dev = {np.std(ari_scores):.4f}")
        
        plt.figure(figsize=(10, 6))
        sns.kdeplot(ari_scores, fill=True, color="purple")
        plt.title('Cross-Sample Model Stability (K-Prototypes)')
        plt.xlabel('Adjusted Rand Score (ARI) vs. Baseline')
        plt.xlim(-0.1, 1.1)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
        
        return ari_scores
    
    def evaluate_stability_by_seed(self, optimal_k, optimal_gamma, random_seeds):
        """
        (NEW)
        Evaluates sensitivity to the initialization seed for K-Prototypes.

        Compares clustering results from different random_state initializations
        against a baseline result (from the first seed). A mean ARI close to 1.0
        indicates high stability and low sensitivity to the initial seed.

        Args:
            optimal_k (int): The number of clusters (k) to evaluate.
            optimal_gamma (float): The gamma value to use.
            random_seeds (list): A list of integers to use as random seeds.
                            Must contain at least two seeds.
        Returns:
            float: The mean ARI score across all seed comparisons.
        """
        if not isinstance(random_seeds, list) or len(random_seeds) < 2:
            raise ValueError("Please provide a list with at least two random seeds.")

        print(f"\n--- Evaluating Stability by Seed for k={optimal_k} ---")
        
        baseline_seed = random_seeds[0]
        other_seeds = random_seeds[1:]
        
        data_for_clustering = self.df.drop('cluster_kproto', axis=1, errors='ignore')
        categorical_indices = [data_for_clustering.columns.get_loc(col) for col in self.categorical_cols]

        # Baseline K-Prototypes run
        baseline_model = KPrototypes(
            n_clusters=optimal_k, 
            gamma=optimal_gamma, 
            init='Cao', 
            n_init=10, 
            random_state=baseline_seed
        )
        baseline_labels = baseline_model.fit_predict(data_for_clustering.values, categorical=categorical_indices)
        
        # Compare with other seeds
        ari_scores_kproto = []
        for seed in other_seeds:
            model = KPrototypes(
                n_clusters=optimal_k, 
                gamma=optimal_gamma, 
                init='Cao', 
                n_init=10, 
                random_state=seed
            )
            labels = model.fit_predict(data_for_clustering.values, categorical=categorical_indices)
            ari_scores_kproto.append(adjusted_rand_score(baseline_labels, labels))

        mean_ari_kproto = np.mean(ari_scores_kproto)
        print(f"K-Prototypes (k={optimal_k}) Mean ARI vs. baseline seed: {mean_ari_kproto:.4f} (Indicates sensitivity to initialization)")
        
        return mean_ari_kproto

    def evaluate_processing_time(self, optimal_k, optimal_gamma, n_steps=10):
        """
        Evaluates and plots the processing time of K-Prototypes with respect
        to the number of samples.

        Args:
            optimal_k (int): The number of clusters (k) to use for timing.
            optimal_gamma (float): The gamma coefficient to use. 
            n_steps (int, optional): The number of sample size increments to test.
        """
        print(f"\n--- Evaluating Processing Time for k={optimal_k} ---")

        total_samples = self.df.shape[0]
        sample_sizes = np.linspace(max(10, int(total_samples / n_steps)), total_samples, n_steps, dtype=int)
        times = []
        categorical_indices = [self.df.columns.get_loc(col) for col in self.categorical_cols]

        for n in sample_sizes:
            print(f"Timing for n={n} samples...")
            # Use a random subset of the data for timing
            sample_df = self.df.sample(n=n, random_state=self.random_state)

            start = time.perf_counter()
            if optimal_gamma is not None: 
                model = KPrototypes(
                    n_clusters=optimal_k, 
                    gamma=optimal_gamma,
                    init='Cao', 
                    n_init=3, 
                    random_state=self.random_state
                    ) 
            else:
                model = KPrototypes(
                    n_clusters=optimal_k, 
                    init='Cao', 
                    n_init=3, 
                    random_state=self.random_state
                    ) 
            model.fit(sample_df.values, categorical=categorical_indices)
            times.append(time.perf_counter() - start)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(sample_sizes, times, 'o-', label=f'K-Prototypes (k={optimal_k})')
        plt.xlabel('Number of Samples (n)')
        plt.ylabel('Processing Time (seconds)')
        plt.title('Processing Time vs. Sample Size')
        plt.grid(True)
        plt.legend()
        plt.show()

        return pd.DataFrame({'Number of Samples': sample_sizes, 'Processing Time (s)': times})

    # ==============================================================================
    # VARIABLE IMPORTANCE
    # ==============================================================================

    def plot_decision_tree_importance(self, max_depth=3):
        """
        Trains a Decision Tree Classifier to predict cluster labels based on the
        original features and visualizes the resulting tree. This helps identify
        which features are most important for defining the clusters.

        Args:
            max_depth (int): The maximum depth of the decision tree. A smaller
                            number (e.g., 3) is more interpretable.
        """
        if self.final_model is None:
            print("Please run `run_final_model()` before analyzing variable importance.")
            return

        print(f"\n--- Generating Decision Tree Visualization (max_depth={max_depth}) 🌳 ---")

        # Use the unscaled data for interpretability
        if self.df_unscaled is not None:
            X_original_features = self.df_unscaled.copy()
            # Ensure it only contains columns used in the final model
            final_cols = self.df.drop('cluster_kproto', axis=1).columns
            X_original_features = X_original_features[final_cols]
        else:
            X_original_features = self.df.drop('cluster_kproto', axis=1).copy()

        # Get the cluster labels
        y_labels = self.df['cluster_kproto']

        # Preprocess categorical features using one-hot encoding for the tree
        categorical_cols_tree = X_original_features.select_dtypes(include=['object', 'category']).columns
        if not categorical_cols_tree.empty:
            print(f"Found categorical features: {list(categorical_cols_tree)}. Applying One-Hot Encoding for the tree...")
            X_processed = pd.get_dummies(X_original_features, columns=categorical_cols_tree, drop_first=True)
        else:
            X_processed = X_original_features

        feature_names = X_processed.columns.tolist()
        class_names = [f'Cluster {i}' for i in sorted(y_labels.unique())]

        # Train Decision Tree Classifier
        tree_classifier = DecisionTreeClassifier(max_depth=max_depth, random_state=self.random_state)
        tree_classifier.fit(X_processed, y_labels)

        # Plot the Tree
        plt.figure(figsize=(25, 12))
        plot_tree(
            tree_classifier,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            fontsize=10,
            precision=2
        )
        plt.title(f'Decision Tree for Explaining K-Prototypes Clusters', fontsize=16)
        plt.show()

    def plot_permutation_importance(self, n_repeats=10, score_metric='accuracy'):
        """
        Calculates and plots feature importance using permutation with a Random Forest surrogate model.

        A Random Forest Classifier is trained to predict cluster labels from the
        original features. The importance of each feature is the measured drop in
        model performance when that feature's values are randomly shuffled.

        Args:
            n_repeats (int, optional): Number of times to permute a feature to get stable results. Defaults to 10.
            score_metric (str, optional): The metric to evaluate the surrogate model. Defaults to 'accuracy'.
        """
        if self.final_model is None:
            print("Error: Final model has not been run. Please run `run_final_model()` first.")
            return

        print(f"\n--- Calculating Permutation Importance for K-Prototypes ---")
        
        # 1. Prepare data: Original features (X) and cluster labels (y)
        if self.df_unscaled is not None:
            X_original = self.df_unscaled.copy()
            final_cols = self.df.drop('cluster_kproto', axis=1).columns
            X_original = X_original[final_cols]
        else:
            X_original = self.df.drop('cluster_kproto', axis=1).copy()
            
        y_labels = self.df['cluster_kproto']

        # One-hot encode categorical features to create a purely numeric feature set for the RF model
        X_processed = pd.get_dummies(X_original, drop_first=True)
        feature_names = X_processed.columns.tolist()

        # 2. Split data for training the surrogate model
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_labels, test_size=0.3, random_state=self.random_state, stratify=y_labels
        )

        # 3. Train the Random Forest surrogate classifier
        print("Training Random Forest surrogate model...")
        surrogate_model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        surrogate_model.fit(X_train, y_train)
        
        # 4. Calculate permutation importance
        print(f"Calculating importance based on '{score_metric}' drop...")
        result = permutation_importance(
            surrogate_model, X_test, y_test, n_repeats=n_repeats,
            random_state=self.random_state, scoring=score_metric, n_jobs=-1
        )
        
        importances = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        }).sort_values('importance_mean', ascending=False)

        # 5. Plot the results
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance_mean', y='feature', data=importances, palette='viridis', hue='feature', dodge=False, legend=False)
        plt.xlabel(f'Mean Importance ({score_metric.capitalize()} Drop)')
        plt.ylabel('Feature')
        plt.title(f'Permutation Feature Importance for K-Prototypes Clusters')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        
        return importances

    def calculate_cluster_statistics(self, p_value_threshold=0.05):
        """
        Calculates statistics to check for significant differences between clusters.

        - For continuous variables: Performs a one-way ANOVA across all clusters.
        - For categorical variables: Performs a Chi-square test of independence.

        Args:
            p_value_threshold (float, optional): The alpha level for significance testing. Defaults to 0.05.

        Returns:
            tuple: A tuple containing two DataFrames: (anova_results, chi2_results)
        """
        if self.final_model is None:
            print("Error: Final model has not been run. Please run `run_final_model()` first.")
            return None, None

        print(f"\n--- Calculating Cluster Statistics for K-Prototypes ---")
        
        # 1. Use the full original dataframe and add the cluster labels
        df_for_stats = self.full_df.copy()
        # Add the cluster labels from the active dataframe. The indices must align.
        df_for_stats['cluster_kproto'] = self.df['cluster_kproto']
        # Drop rows that might not have a cluster label (e.g., if sampling happened)
        df_for_stats.dropna(subset=['cluster_kproto'], inplace=True)
        cluster_col = 'cluster_kproto'

        # 2. Identify variable types from this full dataframe
        continuous_vars = df_for_stats.select_dtypes(include=np.number).columns.tolist()
        if cluster_col in continuous_vars:
            continuous_vars.remove(cluster_col) # Don't test the cluster label itself
        categorical_vars = df_for_stats.select_dtypes(include=['object', 'category']).columns.tolist()

        # --- 3. ANOVA for Continuous Variables ---
        anova_results_list = []
        if continuous_vars:
            cluster_ids = sorted(df_for_stats[cluster_col].unique())
            if len(cluster_ids) > 1:
                for var in continuous_vars:
                    groups = [df_for_stats[df_for_stats[cluster_col] == c][var].dropna() for c in cluster_ids]
                    if all(len(g) > 1 for g in groups):
                        f_stat, p_val = stats.f_oneway(*groups)
                        anova_results_list.append({
                            'Variable': var, 'F-statistic': f_stat, 'P-value': p_val,
                            'Significant': p_val < p_value_threshold
                        })
        
        anova_results = pd.DataFrame(anova_results_list).set_index('Variable') if anova_results_list else pd.DataFrame()
        
        # --- 4. Chi-Square Tests for Categorical Variables ---
        chi2_results_list = []
        if categorical_vars:
            for var in categorical_vars:
                contingency_table = pd.crosstab(df_for_stats[var], df_for_stats[cluster_col])
                if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                    chi2, p_val, _, _ = stats.chi2_contingency(contingency_table)
                    chi2_results_list.append({
                        'Variable': var, 'Chi-Square': chi2, 'P-value': p_val,
                        'Significant': p_val < p_value_threshold
                    })

        chi2_results = pd.DataFrame(chi2_results_list).set_index('Variable') if chi2_results_list else pd.DataFrame()
            
        print("Statistical analysis complete.")
        return anova_results, chi2_results

    def calculate_effect_sizes(self):
        """
        Calculates effect sizes to measure the magnitude of differences between clusters.

        - For continuous variables: Calculates Cohen's d for each pair of clusters.
        - For categorical variables: Calculates Cramér's V across all clusters.

        Returns:
            tuple: A tuple containing two DataFrames: (cohens_d_results, cramers_v_results)
        """
        if self.final_model is None:
            print("Error: Final model has not been run. Please run `run_final_model()` first.")
            return None, None

        print(f"\n--- Calculating Effect Sizes for K-Prototypes ---")
        
        # Use the full original dataframe and add the cluster labels
        df_for_stats = self.full_df.copy()
        # Add the cluster labels from the active dataframe. The indices must align.
        df_for_stats['cluster_kproto'] = self.df['cluster_kproto']
        # Drop rows that might not have a cluster label
        df_for_stats.dropna(subset=['cluster_kproto'], inplace=True)
        cluster_col = 'cluster_kproto'

        continuous_vars = df_for_stats.select_dtypes(include=np.number).columns.tolist()
        if cluster_col in continuous_vars:
            continuous_vars.remove(cluster_col) # Don't test the cluster label itself
        categorical_vars = df_for_stats.select_dtypes(include=['object', 'category']).columns.tolist()

        # --- Cohen's d for Continuous Variables ---
        cohens_d_results_list = []
        if continuous_vars:
            cluster_ids = sorted(df_for_stats[cluster_col].unique())
            for var in continuous_vars:
                for c1, c2 in combinations(cluster_ids, 2):
                    group1 = df_for_stats[df_for_stats[cluster_col] == c1][var].dropna()
                    group2 = df_for_stats[df_for_stats[cluster_col] == c2][var].dropna()
                    
                    n1, n2 = len(group1), len(group2)
                    if n1 > 1 and n2 > 1:
                        mean1, mean2 = group1.mean(), group2.mean()
                        std1, std2 = group1.std(), group2.std()
                        
                        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
                        d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
                            
                        cohens_d_results_list.append({
                            'Variable': var, 'Comparison': f'Cluster {c1} vs {c2}', "Cohen's d": d
                        })
        
        cohens_d_results = pd.DataFrame(cohens_d_results_list).set_index(['Variable', 'Comparison']) if cohens_d_results_list else pd.DataFrame()

        # --- Cramér's V for Categorical Variables ---
        cramers_v_results_list = []
        if categorical_vars:
            for var in categorical_vars:
                contingency_table = pd.crosstab(df_for_stats[var], df_for_stats[cluster_col])
                if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                    chi2, _, _, _ = stats.chi2_contingency(contingency_table)
                    n = contingency_table.sum().sum()
                    phi2 = chi2 / n
                    r, k = contingency_table.shape
                    v = np.sqrt(phi2 / min(k - 1, r - 1))
                    cramers_v_results_list.append({'Variable': var, "Cramér's V": v})

        cramers_v_results = pd.DataFrame(cramers_v_results_list).set_index('Variable') if cramers_v_results_list else pd.DataFrame()

        print("Effect size calculation complete.")
        return cohens_d_results, cramers_v_results