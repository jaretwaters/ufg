import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
import seaborn as sns
import os
from itertools import combinations
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

class ClusterEvaluator:
    """
    A class for performing, evaluating, and comparing K-Means, Hierarchical, and GMM clustering.
    """
    def __init__(self, X, X_pre_rd, k_range, random_state=42, linkage='ward', covariance_type='full'):
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

        self.k_range = k_range
        self.random_state = random_state
        self.linkage = linkage
        self.covariance_type = covariance_type
        
        self.results = {'kmeans': pd.DataFrame(), 'hierarchical': pd.DataFrame(), 'gmm': pd.DataFrame()}
        self.final_models = {'kmeans': None, 'hierarchical': None, 'gmm': None}

    # ==============================================================================
    # DATA PREPARATION
    # ==============================================================================
    def stratified_sample(self, strata_col, sample_frac, random_state=42):
        """
        Performs stratified sampling on the active (RD) dataset and creates a
        corresponding sample from the original (pre-RD) dataset.

        It assumes the original, pre-RD data is stored in `self._original_pre_rd_df`.
        """
        # --- Validate Inputs ---
        if strata_col not in self._original_X_df.columns:
            print(f"Error: Stratification column '{strata_col}' not found.")
            return

        # Check if the pre-RD dataframe exists
        if not hasattr(self, '_original_pre_rd_df'):
            print("Error: Original pre-RD dataframe not found. Please store it in `self._original_pre_rd_df`.")
            return

        print(f"Performing stratified sampling on '{strata_col}' with a {sample_frac:.2%} fraction...")
        
        # --- Step 1: Sample the active (RD) DataFrame ---
        df_sampled_rd = self._original_X_df.groupby(strata_col, group_keys=False).apply(
            lambda x: x.sample(frac=sample_frac, random_state=random_state), 
            include_groups=False
        )

        # --- Step 2: Use the index to sample the original (pre-RD) DataFrame ---
        sampled_indices = df_sampled_rd.index
        df_sampled_original = self._original_pre_rd_df.loc[sampled_indices]

        # --- Step 3: Update class attributes ---
        original_rows = len(self._original_X_df)
        
        # Update the active dataset to the new RD sample
        self.X_df = df_sampled_rd
        self.X = self.X_df.values
        
        # Store the corresponding sample of original data as a new attribute
        self.X_df_original_sampled = df_sampled_original
        
        new_rows = len(self.X_df)
        
        print(f"Sampling complete. Active dataset reduced from {original_rows} to {new_rows} rows.")
        print(f"A corresponding sample from the original data is now available in `self.X_df_original_sampled`.")
        
        return self.X_df
    
    def scale_features(self, columns_to_scale=None):
        """
        Scales specified feature columns using StandardScaler.

        If no columns are specified, all columns in the active DataFrame are scaled.
        The scaler is stored in self.scaler for potential inverse transformations.

        Args:
            columns_to_scale (list, optional): A list of column names to scale. 
            Defaults to None, which scales all columns.
        """
        self.scaler = StandardScaler()
        
        if columns_to_scale is None:
            # Scale all columns by default
            columns_to_scale = self.X_df.columns.tolist()
            print(f"Scaling all {len(columns_to_scale)} features...")
        else:
            # Check if specified columns exist
            missing_cols = [col for col in columns_to_scale if col not in self.X_df.columns]
            if missing_cols:
                print(f"Error: The following columns to scale were not found: {missing_cols}")
                return
            print(f"Scaling {len(columns_to_scale)} specified features: {columns_to_scale}...")

        # Create a copy to avoid SettingWithCopyWarning
        df_copy = self.X_df.copy()

        # Fit and transform the specified columns
        df_copy[columns_to_scale] = self.scaler.fit_transform(df_copy[columns_to_scale])
        
        # Update the class attributes with the scaled data
        self.X_df = df_copy
        self.X = self.X_df.values
        
        print("Scaling complete. Active dataset has been updated.")
        return self.X_df

    # ==============================================================================
    # CLUSTERING ALGORITHMS
    # ==============================================================================
    def run_and_evaluate(self, algorithms=['kmeans', 'hierarchical', 'gmm']):
        """
        Runs and evaluates the specified clustering algorithms over the k_range.
        """
        if 'kmeans' in algorithms:
            self._evaluate_kmeans()
        if 'hierarchical' in algorithms:
            self._evaluate_hierarchical()
        if 'gmm' in algorithms:
            self._evaluate_gmm()
        
        print("\nEvaluation complete for all specified algorithms.")

    def _evaluate_kmeans(self):
        print(f"\nRunning K-Means for k in {list(self.k_range)}...")
        results_k = {}
        for k in self.k_range:
            model = KMeans(n_clusters=k, n_init='auto', random_state=self.random_state)
            labels = model.fit_predict(self.X)
            results_k[k] = {
                'TWSS': model.inertia_,
                'Silhouette Score': silhouette_score(self.X, labels),
                'DBI Score': davies_bouldin_score(self.X, labels)
            }
        self.results['kmeans'] = pd.DataFrame.from_dict(results_k, orient='index')

    def _calculate_twss(self, labels):
        twss = 0
        for label in np.unique(labels):
            cluster_points = self.X[labels == label]
            if len(cluster_points) > 0:
                centroid = cluster_points.mean(axis=0)
                twss += np.sum((cluster_points - centroid) ** 2)
        return twss

    def _evaluate_hierarchical(self):
        print(f"\nRunning Hierarchical Clustering (linkage='{self.linkage}') for k in {list(self.k_range)}...")
        results_h = {}
        for k in self.k_range:
            model = AgglomerativeClustering(n_clusters=k, linkage=self.linkage)
            labels = model.fit_predict(self.X)
            results_h[k] = {
                'TWSS': self._calculate_twss(labels),
                'Silhouette Score': silhouette_score(self.X, labels),
                'DBI Score': davies_bouldin_score(self.X, labels)
            }
        self.results['hierarchical'] = pd.DataFrame.from_dict(results_h, orient='index')
        
    def _evaluate_gmm(self):
        """Runs Gaussian Mixture Model for each k."""
        print(f"\nRunning GMM (covariance_type='{self.covariance_type}') for k in {list(self.k_range)}...")
        results_g = {}
        for k in self.k_range:
            model = GaussianMixture(n_components=k, covariance_type=self.covariance_type, random_state=self.random_state)
            labels = model.fit_predict(self.X)
            results_g[k] = {
                'TWSS': self._calculate_twss(labels),
                'Silhouette Score': silhouette_score(self.X, labels),
                'DBI Score': davies_bouldin_score(self.X, labels),
                'BIC': model.bic(self.X),
                'AIC': model.aic(self.X)
            }
        self.results['gmm'] = pd.DataFrame.from_dict(results_g, orient='index')

    def get_results_df(self, algorithm):
        return self.results[algorithm]

    def run_final_models(self, optimal_k_dict):
        feature_df = self.X_df[[col for col in self.X_df.columns if not col.startswith('cluster')]]
        
        if 'kmeans' in optimal_k_dict:
            k = optimal_k_dict['kmeans']
            print(f"\nRunning final K-Means model with k={k}...")
            model = KMeans(n_clusters=k, n_init='auto', random_state=self.random_state)
            model.fit(feature_df)
            self.final_models['kmeans'] = model
            self.X_df['cluster_kmeans'] = model.labels_

        if 'hierarchical' in optimal_k_dict:
            k = optimal_k_dict['hierarchical']
            print(f"\nRunning final Hierarchical model (linkage='{self.linkage}') with k={k}...")
            model = AgglomerativeClustering(n_clusters=k, linkage=self.linkage)
            model.fit(feature_df)
            self.final_models['hierarchical'] = model
            self.X_df['cluster_hierarchical'] = model.labels_
            
        if 'gmm' in optimal_k_dict:
            k = optimal_k_dict['gmm']
            print(f"\nRunning final GMM (covariance_type='{self.covariance_type}') with k={k}...")
            model = GaussianMixture(n_components=k, covariance_type=self.covariance_type, random_state=self.random_state)
            model.fit(feature_df)
            self.final_models['gmm'] = model
            self.X_df['cluster_gmm'] = model.predict(feature_df)
            
        print("\nFinal models are trained and cluster labels are added to the DataFrame.")
        return self.X_df

    # ==============================================================================
    # VISUALIZATION & METRICS
    # ==============================================================================
    def plot_evaluation_metrics(self):
        if all(df.empty for df in self.results.values()):
            print("Please run `run_and_evaluate()` first.")
            return

        metrics = ['TWSS', 'Silhouette Score', 'DBI Score']
        fig, axs = plt.subplots(1, 3, figsize=(20, 6))
        
        for ax, metric in zip(axs, metrics):
            if not self.results['kmeans'].empty:
                ax.plot(self.results['kmeans'].index, self.results['kmeans'][metric], '-', label='K-Means')
            if not self.results['hierarchical'].empty:
                ax.plot(self.results['hierarchical'].index, self.results['hierarchical'][metric], '-', label=f'Hierarchical ({self.linkage})')
            if not self.results['gmm'].empty:
                ax.plot(self.results['gmm'].index, self.results['gmm'][metric], '-', label=f'GMM ({self.covariance_type})')
            
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel(metric)
            ax.set_title(metric)
            ax.grid(True)
            ax.legend()
        
        fig.suptitle('Cluster Evaluation Metrics Comparison', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_clusters(self, palette='gnuplot'):
        if all(model is None for model in self.final_models.values()):
            print("Please run `run_final_models()` before plotting clusters.")
            return
            
        feature_columns = [col for col in self.X_df.columns if not col.startswith('cluster')]
        if len(feature_columns) < 2:
            print("At least two features are required for 2D plots.")
            return

        for feature1, feature2 in combinations(feature_columns, 2):
            print(f"--- Plotting clusters for: {feature1} vs. {feature2} ---")
            fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)

            # K-Means
            if 'cluster_kmeans' in self.X_df.columns:
                sns.scatterplot(ax=axs[0], x=feature1, y=feature2, hue='cluster_kmeans', data=self.X_df, palette=palette)
                axs[0].set_title(f'K-Means (k={self.final_models["kmeans"].n_clusters})')
            else:
                axs[0].set_title('K-Means (not run)')

            # Hierarchical
            if 'cluster_hierarchical' in self.X_df.columns:
                sns.scatterplot(ax=axs[1], x=feature1, y=feature2, hue='cluster_hierarchical', data=self.X_df, palette=palette)
                axs[1].set_title(f'Hierarchical ({self.linkage}) (k={self.final_models["hierarchical"].n_clusters})')
            else:
                axs[1].set_title('Hierarchical (not run)')
                
            # GMM
            if 'cluster_gmm' in self.X_df.columns:
                sns.scatterplot(ax=axs[2], x=feature1, y=feature2, hue='cluster_gmm', data=self.X_df, palette=palette)
                axs[2].set_title(f'GMM ({self.covariance_type}) (k={self.final_models["gmm"].n_components})')
            else:
                axs[2].set_title('GMM (not run)')
                
            plt.tight_layout()
            plt.show()

    def plot_cluster_matrix(self, algorithm, palette='gnuplot'):
        """
        Creates a single pair plot (scatter matrix) for a given clustering algorithm.

        This visualization is similar to a correlation matrix heatmap, showing scatter plots
        for each pair of features off the diagonal and feature distributions on the diagonal.
        All plots are colored by the final cluster assignments of the specified algorithm.

        Args:
            algorithm (str): The algorithm to plot ('kmeans', 'hierarchical', or 'gmm').
            palette (str, optional): The color palette to use for the plots.
        """
        cluster_col = f'cluster_{algorithm}'
        model = self.final_models.get(algorithm)

        if model is None or cluster_col not in self.X_df.columns:
            print(f"Please run `run_final_models()` for '{algorithm}' before plotting.")
            return
            
        feature_columns = [col for col in self.X_df.columns if not col.startswith('cluster')]
        if len(feature_columns) < 2:
            print("At least two features are required for a pair plot.")
            return
        
        print(f"\n--- Generating Cluster Matrix Plot for {algorithm.upper()} ---")

        plot_df = self.X_df[feature_columns + [cluster_col]]
        # Using corner=True creates a plot similar to a correlation matrix heatmap
        g = sns.pairplot(plot_df, hue=cluster_col, palette=palette, corner=True)
        
        if algorithm == 'kmeans':
            title = f'K-Means Clustering (k={model.n_clusters})'
        elif algorithm == 'hierarchical':
            title = f'Hierarchical Clustering ({self.linkage}, k={model.n_clusters})'
        elif algorithm == 'gmm':
            title = f'GMM Clustering ({self.covariance_type}, k={model.n_components})'
        else:
            title = f'{algorithm.upper()} Clustering'

        g.fig.suptitle(title, y=1.02, fontsize=16)
        plt.show()

    def plot_cluster_sizes(self):
        if all(model is None for model in self.final_models.values()):
            print("Please run `run_final_models()` before plotting cluster sizes.")
            return

        fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
        fig.suptitle('Comparison of Cluster Sizes', fontsize=16)

        # K-Means
        if 'cluster_kmeans' in self.X_df.columns:
            sizes = self.X_df['cluster_kmeans'].value_counts().sort_index()
            sns.barplot(ax=axs[0], x=sizes.index, y=sizes.values, palette='viridis', hue=sizes.index, legend=False)
            axs[0].set_title(f'K-Means (k={self.final_models["kmeans"].n_clusters})')
            axs[0].set_ylabel('Number of Points (Size)')
        else:
            axs[0].set_title('K-Means (not run)')

        # Hierarchical
        if 'cluster_hierarchical' in self.X_df.columns:
            sizes = self.X_df['cluster_hierarchical'].value_counts().sort_index()
            sns.barplot(ax=axs[1], x=sizes.index, y=sizes.values, palette='plasma', hue=sizes.index, legend=False)
            axs[1].set_title(f'Hierarchical ({self.linkage}) (k={self.final_models["hierarchical"].n_clusters})')
        else:
            axs[1].set_title('Hierarchical (not run)')
            
        # GMM
        if 'cluster_gmm' in self.X_df.columns:
            sizes = self.X_df['cluster_gmm'].value_counts().sort_index()
            sns.barplot(ax=axs[2], x=sizes.index, y=sizes.values, palette='magma', hue=sizes.index, legend=False)
            axs[2].set_title(f'GMM ({self.covariance_type}) (k={self.final_models["gmm"].n_components})')
        else:
            axs[2].set_title('GMM (not run)')

        for ax in axs:
            ax.set_xlabel('Cluster ID')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    
    # ==============================================================================
    # HIERARCHICAL-SPECIFIC METHODS
    # ==============================================================================
    def evaluate_linkage_methods(self, linkage_methods=['ward', 'complete', 'average', 'single']):
        print(f"\n--- Evaluating Linkage Methods for k in {list(self.k_range)} ---")
        all_results = {}
        for linkage in linkage_methods:
            print(f"Testing linkage: '{linkage}'...")
            current_results = {}
            for k in self.k_range:
                model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
                labels = model.fit_predict(self.X)
                if len(set(labels)) > 1:
                    silhouette, dbi = silhouette_score(self.X, labels), davies_bouldin_score(self.X, labels)
                else:
                    silhouette, dbi = -1, -1
                current_results[k] = {'Silhouette Score': silhouette, 'DBI Score': dbi}
            all_results[linkage] = pd.DataFrame.from_dict(current_results, orient='index')
        
        fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
        fig.suptitle('Hierarchical Linkage Method Comparison', fontsize=16)
        
        for ax, metric in zip(axs, ['Silhouette Score', 'DBI Score']):
            for linkage_name, results_df in all_results.items():
                ax.plot(results_df.index, results_df[metric], '-', label=linkage_name)
            ax.set_title(metric)
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Score')
            ax.grid(True)
            ax.legend()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        return all_results

    def plot_dendrogram(self, linkage=None, k_cut=None):
        linkage = linkage or self.linkage
        print(f"\n--- Generating Dendrogram with '{linkage}' linkage ---")
        Z = sch.linkage(self.X, method=linkage)
        plt.figure(figsize=(15, 7))
        if k_cut is not None and k_cut > 1:
            plt.title(f'Dendrogram with {k_cut} Clusters (Linkage: {linkage})')
            distances = sorted(Z[:, 2], reverse=True)
            threshold = distances[k_cut - 2]
            sch.dendrogram(Z, color_threshold=threshold)
            plt.axhline(y=threshold, c='grey', lw=1.5, linestyle='--')
        else:
            plt.title(f'Hierarchical Clustering Dendrogram (Linkage: {linkage})')
            sch.dendrogram(Z)
        plt.xlabel('Data Points / Clusters')
        plt.ylabel('Distance (Dissimilarity)')
        plt.grid(axis='y')
        plt.show()
    
    # ==============================================================================
    # GMM-SPECIFIC METHODS
    # ==============================================================================
    def evaluate_gmm_criteria(self, covariance_types=['full', 'tied', 'diag', 'spherical']):
        """
        Evaluates GMM performance using BIC and ICL
        for different covariance types across a range of k values.
        
        Args:
            covariance_types (list): GMM covariance types to test.
        """
        print(f"\n--- Evaluating GMM Information Criteria for k in {list(self.k_range)} ---")
        all_results = {}

        for cov_type in covariance_types:
            print(f"Testing covariance type: '{cov_type}'...")
            current_results = {}
            for k in self.k_range:
                model = GaussianMixture(n_components=k, covariance_type=cov_type, random_state=self.random_state)
                labels = model.fit_predict(self.X)
                
                # Calculate both metrics
                bic = model.bic(self.X)
                probs = model.predict_proba(self.X)
                entropy = -np.sum(probs * np.log(probs + 1e-9))
                icl = bic + entropy
                
                current_results[k] = {
                    'BIC': bic, 
                    'ICL': icl
                }
            all_results[cov_type] = pd.DataFrame.from_dict(current_results, orient='index')

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
            for cov_type, results_df in all_results.items():
                ax.plot(results_df.index, results_df[metric], '-', label=cov_type)
            ax.set_title(metric)
            ax.set_ylabel(metric_properties[metric])
            ax.set_xlabel('Number of Components (k)')
            ax.grid(True)
            ax.legend()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        
        return all_results
    
    def plot_feature_distributions(self, palette='viridis'):
        """
        Creates violin plots to compare the distribution of each feature across clusters
        for each of the final trained models.

        This helps to understand how each feature contributes to the separation of clusters
        in each algorithm. For best results, it's recommended to scale your features
        using the `.scale_features()` method before running the final models.

        Args:
            palette (str, optional): The color palette to use for the plots.
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
            fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
            fig.suptitle(f'Distribution of "{feature}" Across Clusters', fontsize=16)

            # Plot for K-Means
            if 'cluster_kmeans' in self.X_df.columns:
                sns.violinplot(ax=axs[0], data=self.X_df, x='cluster_kmeans', y=feature, hue='cluster_kmeans', palette=palette, legend=False)
                axs[0].set_title(f'K-Means (k={self.final_models["kmeans"].n_clusters})')
            else:
                axs[0].set_title('K-Means (not run)')

            # Plot for Hierarchical
            if 'cluster_hierarchical' in self.X_df.columns:
                sns.violinplot(ax=axs[1], data=self.X_df, x='cluster_hierarchical', y=feature, hue='cluster_hierarchical', palette=palette, legend=False)
                axs[1].set_title(f'Hierarchical ({self.linkage}) (k={self.final_models["hierarchical"].n_clusters})')
            else:
                axs[1].set_title('Hierarchical (not run)')

            # Plot for GMM
            if 'cluster_gmm' in self.X_df.columns:
                sns.violinplot(ax=axs[2], data=self.X_df, x='cluster_gmm', y=feature, hue='cluster_gmm', palette=palette, legend=False)
                axs[2].set_title(f'GMM ({self.covariance_type}) (k={self.final_models["gmm"].n_components})')
            else:
                axs[2].set_title('GMM (not run)')

            # Apply common labels
            for ax in axs:
                ax.set_xlabel('Cluster ID')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
    
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

        plt.title(f'GMM Posterior Probability Distribution (k={gmm_model.n_components})', fontsize=16)
        plt.xlabel('Assigned Cluster ID')
        plt.ylabel("Model's Confidence in Assignment")
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
        Uses this evaluator's models to predict on another's dataset.
        Skips Hierarchical clustering as it lacks a direct predict method.
        """
        print(f"\n--- Cross-Model Prediction: '{name_source}' -> '{name_target}' ---")
        ari_scores = {}
        
        for model_type in ['kmeans', 'gmm']:
            source_model = self.final_models.get(model_type)
            target_model = target_evaluator.final_models.get(model_type)
            
            if source_model is None or target_model is None:
                print(f"[INFO] Skipping {model_type.upper()}: One or both models not trained.")
                continue

            print(f"\n--- Evaluating {model_type.upper()} ---")
            
            source_features = source_model.feature_names_in_
            target_df = target_evaluator.X_df
            
            if not set(source_features).issubset(target_df.columns):
                print(f"[ERROR] Target is missing features required by source {model_type.upper()} model.")
                continue
            
            # Add an info message if target has extra columns
            if set(source_features) != set(target_df.columns):
                print(f"[INFO] Target has different features. Aligning to {len(source_features)} features from source model for prediction.")

            target_data_for_prediction = target_df[source_features]
            original_target_labels = target_df[f'cluster_{model_type}']
            
            predicted_labels = source_model.predict(target_data_for_prediction)
            ari = adjusted_rand_score(original_target_labels, predicted_labels)
            ari_scores[model_type] = ari
            print(f"Adjusted Rand Score (ARI) for {model_type.upper()}: {ari:.4f}")

        return pd.Series(ari_scores, name='ARI Scores')

    def plot_cross_prediction_results(self, target_evaluator, algorithm, name_source='Source', name_target='Target'):
        """
        Visually compares the original target clusters with the clusters predicted by the source model.
        """
        if algorithm not in ['kmeans', 'gmm']:
            print(f"[ERROR] Visualization is only supported for 'kmeans' or 'gmm'.")
            return

        source_model = self.final_models.get(algorithm)
        target_model = target_evaluator.final_models.get(algorithm)

        if source_model is None or target_model is None:
            print(f"[ERROR] Skipping {algorithm.upper()}: Final models must be trained for both source and target.")
            return

        target_df = target_evaluator.X_df.copy()
        original_cluster_col = f'cluster_{algorithm}'
        
        if original_cluster_col not in target_df.columns:
            print(f"[ERROR] Target evaluator is missing original cluster labels for {algorithm.upper()}.")
            return
            
        source_features = source_model.feature_names_in_
        
        if not set(source_features).issubset(target_df.columns):
            print(f"[ERROR] Target is missing features required by the source model.")
            return

        target_data_for_prediction = target_df[source_features]
        predicted_labels = source_model.predict(target_data_for_prediction)
        target_df['predicted_cluster'] = predicted_labels
        
        feature_columns_to_plot = source_features
        
        print(f"\n--- Generating Cross-Prediction Plots for {algorithm.upper()} ---")
        # Generate combinations from the corrected feature list
        for feature1, feature2 in combinations(feature_columns_to_plot, 2):
            fig, axs = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
            fig.suptitle(f"Cross-Prediction Visualization: {algorithm.upper()} for {feature1} vs {feature2}", fontsize=16)

            # Left Plot: Original Clusters
            sns.scatterplot(ax=axs[0], data=target_df, x=feature1, y=feature2, hue=original_cluster_col, palette='viridis')
            axs[0].set_title(f"Original Clusters ({name_target})")
            axs[0].legend(title='Original Cluster')

            # Right Plot: Predicted Clusters
            sns.scatterplot(ax=axs[1], data=target_df, x=feature1, y=feature2, hue='predicted_cluster', palette='viridis')
            axs[1].set_title(f"Clusters Predicted by {name_source}")
            axs[1].legend(title='Predicted Cluster')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.93])
            plt.show()

    def plot_cross_prediction_matrix(self, target_evaluator, algorithm, name_source='Source', name_target='Target', palette='viridis'):
        """
        Creates a pair plot for the original and predicted cluster labels in a cross-prediction scenario.

        This generates two separate matrix plots:
        1. A pair plot of the target dataset colored by its original cluster labels.
        2. A pair plot of the target dataset colored by the cluster labels predicted by the source model.

        Args:
            target_evaluator (ClusterEvaluator): The evaluator object containing the target dataset.
            algorithm (str): The algorithm to visualize ('kmeans' or 'gmm').
            name_source (str, optional): A name for the source model/dataset.
            name_target (str, optional): A name for the target model/dataset.
            palette (str, optional): The color palette to use for the plots.
        """
        if algorithm not in ['kmeans', 'gmm']:
            print("[ERROR] Visualization is only supported for 'kmeans' or 'gmm'.")
            return

        source_model = self.final_models.get(algorithm)
        if source_model is None or target_evaluator.final_models.get(algorithm) is None:
            print(f"[ERROR] Final models must be trained for both source and target for {algorithm.upper()}.")
            return

        target_df = target_evaluator.X_df.copy()
        original_cluster_col = f'cluster_{algorithm}'
        if original_cluster_col not in target_df.columns:
            print(f"[ERROR] Target evaluator is missing original cluster labels for {algorithm.upper()}.")
            return
            
        source_features = source_model.feature_names_in_
        if not set(source_features).issubset(target_df.columns):
            print("[ERROR] Target is missing features required by the source model.")
            return

        predicted_labels = source_model.predict(target_df[source_features])
        predicted_cluster_col = 'predicted_cluster'
        target_df[predicted_cluster_col] = predicted_labels
        
        print(f"\n--- Generating Cross-Prediction Matrix Plots for {algorithm.upper()} ---")

        print(f"Plotting original clusters for '{name_target}'...")
        g_original = sns.pairplot(target_df, vars=source_features, hue=original_cluster_col, palette=palette, corner=True)
        g_original.fig.suptitle(f"Original Clusters ({name_target})", y=1.02, fontsize=16)
        plt.show()
        
        print(f"Plotting clusters predicted by '{name_source}'...")
        g_predicted = sns.pairplot(target_df, vars=source_features, hue=predicted_cluster_col, palette=palette, corner=True)
        g_predicted.fig.suptitle(f"Clusters Predicted by {name_source}", y=1.02, fontsize=16)
        plt.show()

    # ==============================================================================
    # STABILITY AND SCALABILITY
    # ==============================================================================
    def evaluate_cross_sample_stability(self, strata_col, sample_frac, k_dict, n_iterations=50):
        """
        Evaluates model stability by training on new stratified samples and predicting
        on the original sample.

        This method assesses if the clustering structure is robust to the initial
        stratified sampling process itself. It uses the Adjusted Rand Index (ARI)
        to measure stability, which is robust to the "label switching" problem.

        Args:
            strata_col (str): The column in the original, pre-RD dataframe to stratify on.
            sample_frac (float): The fraction of data to draw for each new sample.
            k_dict (dict): Dictionary mapping 'kmeans' and/or 'gmm' to their k-values.
            n_iterations (int): The number of new stratified samples to generate and test.

        Returns:
            dict: A dictionary containing the lists of ARI scores for each algorithm.
        """
        # --- 1. Input Validation and Setup ---
        if 'hierarchical' in k_dict:
            print("[INFO] Hierarchical clustering is excluded from this analysis as it lacks a predict method.")
        
        algos_to_run = [algo for algo in k_dict if algo in ['kmeans', 'gmm']]
        if not algos_to_run:
            print("[ERROR] This method requires 'kmeans' or 'gmm' in k_dict.")
            return

        if not hasattr(self, '_original_X_df') or not hasattr(self, '_original_pre_rd_df'):
            print("[ERROR] Original dataframes `_original_X_df` and `_original_pre_rd_df` not found.")
            return

        print(f"\n--- Evaluating Cross-Sample Stability ({n_iterations} iterations) ---")
        ari_scores = {algo: [] for algo in algos_to_run}
        baselines = {}
        
        # --- 2. Establish Baseline Clustering on the Current Sample ---
        print("Calculating baseline clusterings on the current stratified sample...")
        baseline_data = self.X
        baseline_indices = self.X_df.index
        
        if 'kmeans' in algos_to_run:
            k = k_dict['kmeans']
            model = KMeans(n_clusters=k, n_init='auto', random_state=self.random_state)
            baselines['kmeans'] = model.fit_predict(baseline_data)
        
        if 'gmm' in algos_to_run:
            k = k_dict['gmm']
            model = GaussianMixture(n_components=k, covariance_type=self.covariance_type, random_state=self.random_state)
            baselines['gmm'] = model.fit_predict(baseline_data)

        # --- 3. Loop, Create New Samples, Train, Predict, and Compare ---
        print(f"Beginning {n_iterations} cross-sample validation iterations...")
        for i in range(n_iterations):
            if (i + 1) % 10 == 0:
                print(f"  Iteration {i + 1}/{n_iterations}...")

            # a) Draw a NEW stratified sample from the full original dataset
            new_sample_df_rd = self._original_X_df.groupby(strata_col, group_keys=False).apply(
                lambda x: x.sample(frac=sample_frac, random_state=self.random_state + i),
                include_groups=False
            )
            X_new_sample = new_sample_df_rd.values

            # b) Train new models on this new sample
            if 'kmeans' in algos_to_run:
                k = k_dict['kmeans']
                model_new = KMeans(n_clusters=k, n_init='auto', random_state=self.random_state)
                model_new.fit(X_new_sample)
                # c) Predict on the original baseline data
                predicted_labels = model_new.predict(baseline_data)
                # d) Calculate ARI and store it
                ari = adjusted_rand_score(baselines['kmeans'], predicted_labels)
                ari_scores['kmeans'].append(ari)

            if 'gmm' in algos_to_run:
                k = k_dict['gmm']
                model_new = GaussianMixture(n_components=k, covariance_type=self.covariance_type, random_state=self.random_state)
                model_new.fit(X_new_sample)
                # c) Predict on the original baseline data
                predicted_labels = model_new.predict(baseline_data)
                # d) Calculate ARI and store it
                ari = adjusted_rand_score(baselines['gmm'], predicted_labels)
                ari_scores['gmm'].append(ari)

        # --- 4. Report Results and Plot ---
        print("\nCross-sample stability evaluation complete.")
        for name, scores in ari_scores.items():
            k = k_dict[name]
            print(f"{name.capitalize()} (k={k}) Stability: Mean ARI = {np.mean(scores):.4f}, Std Dev = {np.std(scores):.4f}")
        
        plot_data = {f"{algo.capitalize()} (k={k_dict[algo]})": scores for algo, scores in ari_scores.items()}
        
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=plot_data, fill=True, common_norm=False, palette="viridis")
        plt.title('Cross-Sample Model Stability')
        plt.xlabel('Adjusted Rand Score (ARI) vs. Baseline')
        plt.xlim(-0.1, 1.1)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
        
        return ari_scores
    
    def evaluate_stability_by_seed(self, k_dict, random_seeds):
        """
        Evaluates sensitivity to initialization seeds for KMeans and GMM.

        Compares clustering results from different random_state initializations
        against a baseline result (from the first seed). A mean ARI close to 1.0
        indicates high stability and low sensitivity to the initial seed.
        Also evaluates Hierarchical clustering to show it is deterministic.

        Args:
            k_dict (dict): A dictionary where keys are algorithm names
                        ('kmeans', 'hierarchical', 'gmm') and values are
                        the number of clusters (k) to evaluate.
            random_seeds (list): A list of integers to use as random seeds.
                            Must contain at least two seeds.
        Returns:
            dict: A dictionary with the mean ARI scores for each evaluated algorithm.
        """
        if not isinstance(k_dict, dict) or not k_dict:
            print("[ERROR] Please provide a non-empty dictionary for k_dict.")
            return
        if not isinstance(random_seeds, list) or len(random_seeds) < 2:
            raise ValueError("Please provide a list with at least two random seeds.")

        title_k_str = ', '.join([f"{name.capitalize()}: {k}" for name, k in k_dict.items()])
        print(f"\n--- Evaluating Stability by Seed for k values ({title_k_str}) ---")
        
        results = {}
        baseline_seed = random_seeds[0]
        other_seeds = random_seeds[1:]

        # K-Means
        if 'kmeans' in k_dict:
            k = k_dict['kmeans']
            baseline_km = KMeans(n_clusters=k, n_init='auto', random_state=baseline_seed).fit_predict(self.X)
            ari_scores_km = [
                adjusted_rand_score(baseline_km, KMeans(n_clusters=k, n_init='auto', random_state=seed).fit_predict(self.X)) 
                for seed in other_seeds
            ]
            mean_ari_km = np.mean(ari_scores_km)
            results['kmeans'] = mean_ari_km
            print(f"K-Means (k={k}) Mean ARI vs. baseline seed: {mean_ari_km:.4f} (Indicates sensitivity to initialization)")
        
        # GMM
        if 'gmm' in k_dict:
            k = k_dict['gmm']
            baseline_gmm = GaussianMixture(n_components=k, covariance_type=self.covariance_type, random_state=baseline_seed).fit_predict(self.X)
            ari_scores_gmm = [
                adjusted_rand_score(baseline_gmm, GaussianMixture(n_components=k, covariance_type=self.covariance_type, random_state=seed).fit_predict(self.X)) 
                for seed in other_seeds
            ]
            mean_ari_gmm = np.mean(ari_scores_gmm)
            results['gmm'] = mean_ari_gmm
            print(f"GMM ({self.covariance_type}, k={k}) Mean ARI vs. baseline seed: {mean_ari_gmm:.4f} (Indicates sensitivity to initialization)")
        
        # Hierarchical
        if 'hierarchical' in k_dict:
            k = k_dict['hierarchical']
            # Hierarchical clustering is deterministic; its result doesn't depend on a random seed.
            # Comparing its output to itself will always yield a perfect ARI of 1.0.
            labels = AgglomerativeClustering(n_clusters=k, linkage=self.linkage).fit_predict(self.X)
            ari_h = adjusted_rand_score(labels, labels) # Will be 1.0
            results['hierarchical'] = ari_h
            print(f"Hierarchical ({self.linkage}, k={k}) ARI vs. baseline: {ari_h:.4f} (Deterministic, always 1.0)")
            
        return results

    def evaluate_processing_time(self, k_dict, n_steps=10):
        """
        Evaluates and plots the processing time (scalability) for algorithms
        with respect to the number of samples.

        Args:
            k_dict (dict): A dictionary where keys are algorithm names
                        ('kmeans', 'hierarchical', 'gmm') and values are
                        the number of clusters (k) to use for timing.
            n_steps (int, optional): The number of sample size increments to test.
                                    Defaults to 10.
        Returns:
            pd.DataFrame: A DataFrame containing the processing times for each
                        evaluated algorithm at different sample sizes.
        """
        if not isinstance(k_dict, dict) or not k_dict:
            print("[ERROR] Please provide a non-empty dictionary for k_dict.")
            return

        title_k_str = ', '.join([f"{name.capitalize()}: {k}" for name, k in k_dict.items()])
        print(f"\n--- Evaluating Processing Time for k values ({title_k_str}) ---")
        
        total_samples = self.X.shape[0]
        sample_sizes = np.linspace(max(10, int(total_samples / n_steps)), total_samples, n_steps, dtype=int)
        
        # Initialize dictionary to store times only for requested algorithms
        times = {name: [] for name in k_dict}
        
        for n in sample_sizes:
            print(f"Timing for n={n} samples...")
            sample_X = self.X[np.random.choice(total_samples, n, replace=False)]
            
            if 'kmeans' in k_dict:
                k = k_dict['kmeans']
                start = time.perf_counter()
                KMeans(n_clusters=k, n_init='auto', random_state=self.random_state).fit(sample_X)
                times['kmeans'].append(time.perf_counter() - start)
            
            if 'hierarchical' in k_dict:
                k = k_dict['hierarchical']
                start = time.perf_counter()
                AgglomerativeClustering(n_clusters=k, linkage=self.linkage).fit(sample_X)
                times['hierarchical'].append(time.perf_counter() - start)

            if 'gmm' in k_dict:
                k = k_dict['gmm']
                start = time.perf_counter()
                GaussianMixture(n_components=k, covariance_type=self.covariance_type, random_state=self.random_state).fit(sample_X)
                times['gmm'].append(time.perf_counter() - start)
                
        # --- Plotting ---
        plt.figure(figsize=(10, 6))
        
        if 'kmeans' in times:
            plt.plot(sample_sizes, times['kmeans'], '-', label='K-Means')
        if 'hierarchical' in times:
            plt.plot(sample_sizes, times['hierarchical'], '--', label=f'Hierarchical ({self.linkage})')
        if 'gmm' in times:
            plt.plot(sample_sizes, times['gmm'], '-.', label=f'GMM ({self.covariance_type})')
            
        plt.xlabel('Number of Samples (n)')
        plt.ylabel('Processing Time (seconds)')
        plt.title(f'Processing Time vs. Sample Size ({title_k_str})')
        plt.grid(True)
        plt.legend()
        plt.show()

        # --- DataFrame Creation ---
        timing_data = {'Number of Samples': sample_sizes}
        column_map = {
            'kmeans': 'KMeans Time (s)',
            'hierarchical': 'Hierarchical Time (s)',
            'gmm': 'GMM Time (s)'
        }
        for name, time_list in times.items():
            timing_data[column_map[name]] = time_list
            
        timing_df = pd.DataFrame(timing_data)
        return timing_df
    
    def evaluate_perturbation_stability(self, k_dict, n_iterations=50, noise_level=0.1):
        """
        Evaluates clustering stability by adding Gaussian noise to the dataset.

        This method compares a baseline clustering on the original data with clusterings
        performed on noisy versions of the data. Stability is quantified using the
        Adjusted Rand Index (ARI).

        A high and stable ARI across iterations indicates a robust clustering solution
        that is resilient to small, random variations or measurement errors in the data.

        Args:
            k_dict (dict): A dictionary mapping algorithm names ('kmeans', 'hierarchical', 'gmm')
                        to the number of clusters (k) to use.
            n_iterations (int): The number of noise perturbation iterations to run.
            noise_level (float): The standard deviation (scale) of the Gaussian noise to add.
            seed (int, optional): A seed for the random number generator to ensure
                                reproducible noise generation. Defaults to None.
        """
        # --- 1. Input Validation ---
        if not k_dict:
            print("Error: The provided k_dict is empty. Please specify algorithms and k-values.")
            return

        # --- 2. Setup and Baseline Calculation ---
        print(f"\n--- Evaluating Noise Perturbation Stability (Noise Level: {noise_level}) ---")

        rng = np.random.default_rng(self.random_state)
        
        ari_scores = {algo: [] for algo in k_dict.keys()}
        baselines = {}

        print("Calculating baseline clusterings on original data...")
        if 'kmeans' in k_dict:
            k = k_dict['kmeans']
            baselines['kmeans'] = KMeans(n_clusters=k, n_init='auto', random_state=self.random_state).fit_predict(self.X)
        
        if 'hierarchical' in k_dict:
            k = k_dict['hierarchical']
            baselines['hierarchical'] = AgglomerativeClustering(n_clusters=k, linkage=self.linkage).fit_predict(self.X)
        
        if 'gmm' in k_dict:
            k = k_dict['gmm']
            baselines['gmm'] = GaussianMixture(n_components=k, covariance_type=self.covariance_type, random_state=self.random_state).fit_predict(self.X)

        # --- 3. Run Perturbation Iterations ---
        print("Baselines calculated. Starting noise iterations...")
        for i in range(n_iterations):
            if (i + 1) % 10 == 0:
                print(f"  Iteration {i + 1}/{n_iterations}...")
            
            # <<< CHANGE: Use the seeded generator to create noise >>>
            noise = rng.normal(loc=0.0, scale=noise_level, size=self.X.shape)
            X_perturbed = self.X + noise

            # Cluster the perturbed data and compare to the baseline
            if 'kmeans' in k_dict:
                k = k_dict['kmeans']
                perturbed_labels = KMeans(n_clusters=k, n_init='auto', random_state=self.random_state + i).fit_predict(X_perturbed)
                ari_scores['kmeans'].append(adjusted_rand_score(baselines['kmeans'], perturbed_labels))
            
            if 'hierarchical' in k_dict:
                k = k_dict['hierarchical']
                perturbed_labels = AgglomerativeClustering(n_clusters=k, linkage=self.linkage).fit_predict(X_perturbed)
                ari_scores['hierarchical'].append(adjusted_rand_score(baselines['hierarchical'], perturbed_labels))
            
            if 'gmm' in k_dict:
                k = k_dict['gmm']
                perturbed_labels = GaussianMixture(n_components=k, covariance_type=self.covariance_type, random_state=self.random_state + i).fit_predict(X_perturbed)
                ari_scores['gmm'].append(adjusted_rand_score(baselines['gmm'], perturbed_labels))

        # --- 4. Report Results and Plot ---
        print("\nNoise evaluation complete.")
        for name, scores in ari_scores.items():
            k = k_dict[name]
            print(f"{name.capitalize()} (k={k}) Stability: Mean ARI = {np.mean(scores):.4f}, Std Dev = {np.std(scores):.4f}")
        
        plot_data = {f"{algo.capitalize()} (k={k_dict[algo]})": scores for algo, scores in ari_scores.items()}
        
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=plot_data, fill=True, common_norm=False, palette="viridis")
        plt.title(f'Noise Perturbation Stability (Noise Level: {noise_level})')
        plt.xlabel('Adjusted Rand Score (ARI) vs. Baseline')
        plt.xlim(-0.1, 1.1)
        plt.grid(True, linestyle='--', alpha=0.6)
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
        for algorithm in ['kmeans', 'hierarchical', 'gmm']:
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
        for algorithm in ['kmeans', 'hierarchical', 'gmm']:
            cluster_col = f'cluster_{algorithm}'
            if cluster_col not in self.X_df.columns:
                print(f"Error: Final model for '{algorithm}' has not been run. Please run `run_final_models()` first.")
                return

            print(f"\n--- Calculating Permutation Importance for {algorithm.upper()} ---")
            
            # 1. Prepare data: Original features (X) and cluster labels (y)
            X_original_features = self._original_pre_rd_df.loc[self.X_df.index]
            y_labels = self.X_df[cluster_col]

            # One-hot encode categorical features to create a purely numeric feature set
            X_processed = pd.get_dummies(X_original_features, drop_first=True)
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
            sns.barplot(x='importance_mean', y='feature', data=importances, palette='viridis', hue='feature', dodge=False)
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
            algorithm (str): The algorithm to evaluate ('kmeans', 'hierarchical', or 'gmm').
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
            algorithm (str): The algorithm to evaluate ('kmeans', 'hierarchical', or 'gmm').

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