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
        series = series.dropna()
        if series.empty:
            return 0.0
            
        value_counts = series.value_counts()
        probs = value_counts / len(series)
        n_classes = len(probs)
        
        if n_classes <= 1:
            return 0.0
            
        entropy = -np.sum(probs * np.log2(probs))
        normalized_entropy = entropy / np.log2(n_classes)
        return normalized_entropy

    def analyze_features(self):
        """
        Calculates and visualizes key metrics for feature selection without
        modifying the dataframe. Helps in deciding thresholds.
        """
        print("\n--- Starting Feature Analysis ---")
        
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
            
            print("Calculating Coefficient of Variation for numerical features...")
            means = self.df[self.numerical_cols].mean().abs()
            stds = self.df[self.numerical_cols].std()
            cv = (stds / means.replace(0, np.nan)).fillna(0).sort_values(ascending=False)
            
            sns.barplot(x=cv.values, y=cv.index, ax=ax_row[0], hue=cv.index, palette='viridis', legend=False)
            ax_row[0].set_title('Coefficient of Variation (Numerical)')
            ax_row[0].set_xlabel('CV (Std Dev / Mean)')
            
            print("Calculating Correlation Matrix for numerical features...")
            corr_matrix = self.df[self.numerical_cols].corr()
            sns.heatmap(corr_matrix, ax=ax_row[1], cmap='coolwarm', annot=False)
            ax_row[1].set_title('Correlation Matrix (Numerical)')

        # --- Categorical Analysis ---
        if has_cat:
            ax_row = axes[1] if has_num and has_cat else axes
            
            print("Calculating Normalized Entropy for categorical features...")
            entropies = {col: self._calculate_normalized_entropy(self.df[col]) for col in self.categorical_cols}
            ent_series = pd.Series(entropies).sort_values(ascending=False)
            
            sns.barplot(x=ent_series.values, y=ent_series.index, ax=ax_row[0], hue=ent_series.index, palette='plasma', legend=False)
            ax_row[0].set_title('Normalized Entropy (Categorical)')
            ax_row[0].set_xlabel('Normalized Entropy')
            
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
        
    def select_features(self, cv_min=None, entropy_min=None, corr_max=None, nmi_max=None):
        """
        Selects features based on specified thresholds for variance and redundancy metrics.

        Args:
            cv_min (float, optional): The minimum coefficient of variation for numerical features. 
                                    Features below this are marked for removal. Defaults to None.
            entropy_min (float, optional): The minimum normalized entropy for categorical features.
                                        Features below this are marked for removal. Defaults to None.
            corr_max (float, optional): The maximum absolute correlation between numerical features.
                                        For a pair above this, one feature is marked for removal. Defaults to None.
            nmi_max (float, optional): The maximum normalized mutual information between categorical features.
                                    For a pair above this, one feature is marked for removal. Defaults to None.

        Returns:
            tuple: A tuple containing two lists: (features_to_keep, features_to_remove).
        """
        print("\n--- Starting Feature Selection Process ---")
        features_to_remove = set()

        # 1. Low Variance/Information Checks
        if cv_min is not None and self.numerical_cols:
            print(f"Checking for numerical features with CV < {cv_min}...")
            means = self.df[self.numerical_cols].mean().abs()
            stds = self.df[self.numerical_cols].std()
            cv = (stds / means.replace(0, np.nan)).fillna(0)
            low_cv_features = cv[cv < cv_min].index.tolist()
            if low_cv_features:
                print(f"  -> Found {len(low_cv_features)} features with low CV: {low_cv_features}")
                features_to_remove.update(low_cv_features)

        if entropy_min is not None and self.categorical_cols:
            print(f"Checking for categorical features with Normalized Entropy < {entropy_min}...")
            entropies = {col: self._calculate_normalized_entropy(self.df[col]) for col in self.categorical_cols}
            ent_series = pd.Series(entropies)
            low_entropy_features = ent_series[ent_series < entropy_min].index.tolist()
            if low_entropy_features:
                print(f"  -> Found {len(low_entropy_features)} features with low entropy: {low_entropy_features}")
                features_to_remove.update(low_entropy_features)
                
        # 2. Redundancy Checks
        if corr_max is not None and len(self.numerical_cols) > 1:
            print(f"Checking for highly correlated numerical features (Correlation > {corr_max})...")
            corr_matrix = self.df[self.numerical_cols].corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            for col in upper_tri.columns:
                correlated_partners = upper_tri.index[upper_tri[col] > corr_max].tolist()
                for partner in correlated_partners:
                    if col in features_to_remove or partner in features_to_remove:
                        continue
                    
                    mean_corr_col = corr_matrix[col].mean()
                    mean_corr_partner = corr_matrix[partner].mean()
                    
                    if mean_corr_col >= mean_corr_partner:
                        features_to_remove.add(col)
                        print(f"  -> Marking '{col}' for removal (highly correlated with '{partner}')")
                    else:
                        features_to_remove.add(partner)
                        print(f"  -> Marking '{partner}' for removal (highly correlated with '{col}')")

        if nmi_max is not None and len(self.categorical_cols) > 1:
            print(f"Checking for redundant categorical features (NMI > {nmi_max})...")
            n = len(self.categorical_cols)
            nmi_matrix = pd.DataFrame(np.eye(n), index=self.categorical_cols, columns=self.categorical_cols)
            for i in range(n):
                for j in range(i + 1, n):
                    col1, col2 = self.categorical_cols[i], self.categorical_cols[j]
                    nmi_score = normalized_mutual_info_score(self.df[col1], self.df[col2])
                    nmi_matrix.loc[col1, col2] = nmi_matrix.loc[col2, col1] = nmi_score
            
            upper_tri_nmi = nmi_matrix.where(np.triu(np.ones(nmi_matrix.shape), k=1).astype(bool))

            for col in upper_tri_nmi.columns:
                nmi_partners = upper_tri_nmi.index[upper_tri_nmi[col] > nmi_max].tolist()
                for partner in nmi_partners:
                    if col in features_to_remove or partner in features_to_remove:
                        continue

                    mean_nmi_col = nmi_matrix[col].mean()
                    mean_nmi_partner = nmi_matrix[partner].mean()

                    if mean_nmi_col >= mean_nmi_partner:
                        features_to_remove.add(col)
                        print(f"  -> Marking '{col}' for removal (high NMI with '{partner}')")
                    else:
                        features_to_remove.add(partner)
                        print(f"  -> Marking '{partner}' for removal (high NMI with '{col}')")

        # 3. Finalize and Return
        all_features = set(self.df.columns)
        features_to_keep = sorted(list(all_features - features_to_remove))
        features_to_remove_sorted = sorted(list(features_to_remove))
        
        print("\n--- Feature Selection Complete ---")
        print(f"Total features to remove: {len(features_to_remove_sorted)}")
        print(f"Features to remove: {features_to_remove_sorted}")
        print(f"Total features to keep: {len(features_to_keep)}")
        print(f"Features to keep: {features_to_keep}")
        
        return features_to_keep, features_to_remove_sorted