"""Data preprocessing"""

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Any, Optional
import lightgbm as lgb
import math 
import textwrap


def load_and_filter_data(file_path: str, filter_conditions: Dict[str, List[Any]], chunk_size: int = 30_000) -> pd.DataFrame:
    """
    Loads and filters a CSV file in chunks based on multiple column conditions.
    
    This is memory-efficient for large files. It keeps only the rows that
    satisfy ALL of the conditions provided (logical AND).

    Args:
        file_path: Path to the CSV file.
        filter_conditions: A dictionary where keys are the column names to filter
                           and values are lists of the desired values to keep for
                           each respective column.
                           Example: {'Region': ['North', 'West'], 'Status': ['Active']}
        chunk_size: The number of rows per chunk to process at a time.

    Returns:
        A pandas DataFrame containing only the data that meets all filter conditions.
        Returns an empty DataFrame if no data matches or if an error occurs.
    """
    filtered_chunks = []
    
    if not isinstance(filter_conditions, dict):
        print("Error: filter_conditions must be a dictionary.")
        return pd.DataFrame()
        
    try:
        # The 'with' statement ensures the reader is properly closed.
        with pd.read_csv(file_path, sep=';', encoding='ISO-8859-1', chunksize=chunk_size, on_bad_lines='warn') as reader:
            for i, chunk in enumerate(reader):
                # Start with a mask that includes all rows in the chunk.
                combined_mask = pd.Series(True, index=chunk.index)
                
                # Sequentially apply each filter condition to the mask.
                for column, desired_values in filter_conditions.items():
                    if column not in chunk.columns:
                        print(f"Warning: Column '{column}' not found in the CSV. Skipping this filter condition.")
                        continue
                    
                    # Use '&' to ensure rows satisfy ALL conditions.
                    combined_mask &= chunk[column].isin(desired_values)
                
                # Apply the final combined mask to the chunk.
                filtered_chunk = chunk[combined_mask]
                
                if not filtered_chunk.empty:
                    filtered_chunks.append(filtered_chunk)
                    
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while loading or filtering the data: {e}")
        return pd.DataFrame()
    
    if not filtered_chunks:
        print("No data found that matches all the specified filter conditions.")
        return pd.DataFrame()

    return pd.concat(filtered_chunks, ignore_index=True)

def filter_student_presence(df, presence_vars=None, redaction_status_var="TP_STATUS_REDACAO", mode='all'):
    """
    Filters students based on their presence in exams and their essay status.

    This function can be configured to filter for students who were present in ALL specified exams
    or in AT LEAST ONE of them.

    Parameters
    ----------
    df : pandas.DataFrame
        The original DataFrame with the ENEM data.
    presence_vars : list, default=None
        A list of column names that track exam presence. If None, it defaults to the four main tests.
    redaction_status_var : str, default="TP_STATUS_REDACAO"
        The name of the column that tracks the essay's status.
    mode : str, default='all'
        The filtering mode for presence.
        - 'all': Keeps students who were present in ALL of the exams in `presence_vars`.
        - 'any': Keeps students who were present in AT LEAST ONE of the exams in `presence_vars`.

    Returns
    -------
    pandas.DataFrame
        A cleaned DataFrame with absent students filtered according to the specified mode.
    """
    print("Dataset original:", df.shape)

    # Default to the four main subject tests if no list is provided
    if presence_vars is None:
        presence_vars = ["TP_PRESENCA_CN", "TP_PRESENCA_CH", "TP_PRESENCA_LC", "TP_PRESENCA_MT"]

    # Apply the presence filter based on the selected mode
    if mode == 'all':
        presence_filter = df[presence_vars].eq(1).all(axis=1)
    elif mode == 'any':
        presence_filter = df[presence_vars].eq(1).any(axis=1)
    else:
        raise ValueError("Mode must be either 'all' or 'any'")

    df_clean = df[presence_filter]

    # Further filter for students with a valid essay status
    if redaction_status_var in df_clean.columns:
        df_clean = df_clean[df_clean[redaction_status_var] == 1]

    # Reset the index after dropping rows
    df_clean = df_clean.reset_index(drop=True)

    print(f"Após a filtragem (modo='{mode}'):", df_clean.shape)
    
    return df_clean

def select_and_map_features(df, columns_to_select, mappings, zero_to_nan_cols=None):
    """
    Selects a subset of columns, applies various mappings for data transformation,
    and converts specified zeros to NaN.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    columns_to_select : list
        A list of column names to keep from the original DataFrame.
    mappings : dict
        A master dictionary where keys are column names (str) or a tuple of
        column names, and values are the mapping dictionaries to be applied.
    zero_to_nan_cols : list, optional
        A list of columns where zeros should be replaced with np.nan.

    Returns
    -------
    pandas.DataFrame
        The processed and transformed DataFrame.
    """
    # 1. Select the columns of interest and create a copy to avoid warnings
    df_processed = df[columns_to_select].copy()
    print(f"Selected {len(columns_to_select)} columns. Shape: {df_processed.shape}")

    # 2. Apply all mappings
    for cols, a_map in mappings.items():
        # Ensure the column(s) are always in an iterable list
        cols_to_map = [cols] if isinstance(cols, str) else cols
        
        for col in cols_to_map:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].replace(a_map)

    # 3. Replace 0 with NaN in specified numeric columns
    if zero_to_nan_cols:
        df_processed[zero_to_nan_cols] = df_processed[zero_to_nan_cols].replace(0, np.nan)

    print("Data mapping and processing complete.")
    return df_processed

def compute_missing(df, normalize=True):
    """
    Calculates the pct/count of missing values per column.

    Parameters
    ----------
    df : `pandas.DataFrame`
    normalize : boolean, default=True

    Returns
    ----------
    missing_df : `pandas.DataFrame`
        DataFrame with the pct/counts of missing values per column.
    """
    missing_df = (df.isnull().sum()).to_frame('missing').reset_index().rename(columns={'index': 'var_name'})
    if normalize:
        missing_df['missing'] = missing_df['missing'] * 100 / df.shape[0]
    missing_df = missing_df.sort_values('missing', ascending=False)
    return missing_df

def savefig(output_path=None, savefig_kws=None):
    if output_path is not None:
        if savefig_kws is not None:
            plt.savefig(output_path, **savefig_kws)
        else:
            plt.savefig(output_path, format='jpg', bbox_inches='tight', dpi=300)

def missing_values_heatmap(df, output_path=None, savefig_kws=None):
    """
    Plots a heatmap to visualize missing values (light color).

    Parameters
    ----------
    df : `pandas.DataFrame`
       DataFrame containing the data.
    output_path : str, default=None
       Path to save figure as image.
    savefig_kws : dict, default=None
       Save figure options.
    """
    fig, ax = plt.subplots(figsize=(18, 10))
    sns.heatmap(df.isnull().astype(int), cbar=False)
    fig.tight_layout()
    savefig(output_path=output_path, savefig_kws=savefig_kws)

def mice_impute_lightgbm(
    df: pd.DataFrame,
    numerical_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Imputes missing values using MICE with a highly efficient LightGBM estimator.

    This function automatically handles mixed data types by label-encoding
    categorical features before imputation and decoding them afterward.

    Args:
        df: The input pandas DataFrame with missing values.
        numerical_cols: A list of column names to be treated as numerical.
        categorical_cols: A list of column names to be treated as categorical.

    Returns:
        A new pandas DataFrame with missing values imputed.
    """
    imputed_df = df.copy()

    # If column lists are not provided, identify them automatically
    if numerical_cols is None or categorical_cols is None:
        print("Column types not provided, auto-identifying...")
        numerical_cols = imputed_df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = imputed_df.select_dtypes(exclude=np.number).columns.tolist()
    
    # Basic validation for provided columns
    all_cols = numerical_cols + categorical_cols
    if not set(all_cols).issubset(set(df.columns)):
        raise ValueError("One or more provided column names are not in the DataFrame.")
    if len(set(numerical_cols).intersection(set(categorical_cols))) > 0:
        raise ValueError("A column cannot be both numerical and categorical.")

    # --- Step 1: Encode Categorical Columns ---
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        non_missing_data = imputed_df[col][imputed_df[col].notna()]
        le.fit(non_missing_data)
        label_encoders[col] = le
        
        transformed_col = imputed_df[col].copy()
        mask = imputed_df[col].notna()
        transformed_col[mask] = le.transform(imputed_df.loc[mask, col])
        imputed_df[col] = transformed_col.astype(float)

    # --- Step 2: Perform MICE imputation with LightGBM ---
    mice_imputer = IterativeImputer(
        estimator=lgb.LGBMRegressor(
            n_estimators=50,  # A reasonable number of trees
            n_jobs=-1,        # Use all available CPU cores for max speed
            random_state=0
        ),
        max_iter=5,           # Imputation often converges in fewer iterations
        random_state=0,
        imputation_order='roman'
    )

    imputed_matrix = mice_imputer.fit_transform(imputed_df[all_cols])
    imputed_subset_df = pd.DataFrame(imputed_matrix, columns=all_cols)

    for col in all_cols:
        imputed_df[col] = imputed_subset_df[col]

    # --- Step 3: Decode the imputed categorical columns ---
    for col in categorical_cols:
        imputed_df[col] = np.round(imputed_df[col]).astype(int)
        
        le = label_encoders[col]
        min_code, max_code = 0, len(le.classes_) - 1
        imputed_df[col] = np.clip(imputed_df[col], min_code, max_code)
        
        imputed_df[col] = le.inverse_transform(imputed_df[col])

    # Restore original data types for numerical columns
    for col in numerical_cols:
        imputed_df[col] = imputed_df[col].astype(df[col].dtype)

    return imputed_df

def plot_variables(df, ignore_cols=None):
    """
    Creates visualizations of variables in a DataFrame.

    This function automatically handles mixed data types, creating a single combined
    boxplot for numeric variables (if they are on a similar scale) and a grid
    of bar charts for categorical variables.

    Args:
        df (pd.DataFrame): The input pandas DataFrame to be plotted.
        ignore_cols (list or str, optional): A column name or list of column
            names to exclude from plotting. Defaults to None.

    Returns:
        None. Displays the plots.
    """
    # --- 1. Handle Columns to Ignore ---
    if ignore_cols is None:
        cols_to_ignore = []
    elif isinstance(ignore_cols, str):
        cols_to_ignore = [ignore_cols]
    else:
        cols_to_ignore = ignore_cols

    # --- 2. Separate Column Types for Plotting ---
    all_numeric = df.select_dtypes(include=np.number).columns.tolist()
    all_categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Filter out the ignored columns
    numeric_cols = [col for col in all_numeric if col not in cols_to_ignore]
    categorical_cols = [col for col in all_categorical if col not in cols_to_ignore]

    print(f"Found {len(numeric_cols)} numeric variables to plot: {numeric_cols}")
    print(f"Found {len(categorical_cols)} categorical variables to plot: {categorical_cols}\n")

    # --- 3. Create a Single Combined Plot for Numeric Variables ---
    if numeric_cols:
        print("Generating a single combined boxplot for numeric variables...")
        plt.figure(figsize=(12, 7))

        # "Melt" the DataFrame to a long format for plotting
        df_melted = df[numeric_cols].melt(var_name='Variable', value_name='Value')

        # Create the combined boxplot
        sns.boxplot(data=df_melted, x='Variable', y='Value')

        plt.title('Distribution of Numeric Variables', fontsize=16)
        plt.xlabel('Variable')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # --- 4. Create a Single Figure with Subplots for Categorical Variables ---
    if categorical_cols:
        print("Generating consolidated bar chart figure for categorical variables...")
        n_vars = len(categorical_cols)
        n_cols = 3
        n_rows = math.ceil(n_vars / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 7, n_rows * 5))
        axes = axes.flatten()

        for i, col in enumerate(categorical_cols):
            ax = axes[i]
            order = df[col].value_counts().index
            sns.countplot(data=df, y=col, hue=col, ax=ax, order=order, palette='viridis', legend=False)
            ax.set_title(f'Frequency of {col}', fontsize=12)
            ax.set_ylabel('')
            ax.set_xlabel('Count')

            labels = [item.get_text() for item in ax.get_yticklabels()]
            wrapped_labels = [textwrap.fill(label, width=25) for label in labels]
            ax.set_yticks(ax.get_yticks(), labels=wrapped_labels)

        for i in range(n_vars, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle('Bar Charts of Categorical Variables', fontsize=16, y=1.0)
        plt.tight_layout(pad=1.5)
        plt.show()