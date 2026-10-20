"""
Module: cardinality.py
Description: Functions for analyzing categorical cardinality, identifying high-cardinality
             features, and calculating sparsity distributions on training data.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_training_cardinality(X_train: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
    """
    Calculates cardinality summary for all categorical columns in X_train:
    - column_name
    - unique_categories
    - missing_values
    - most_frequent_category
    - frequency_most_frequent
    - categories_lt_10
    - categories_lt_50
    - categories_lt_100
    - cardinality_ratio
    """
    n_rows = len(X_train)
    records = []
    
    for col in categorical_cols:
        vc = X_train[col].value_counts()
        n_unique = int(X_train[col].nunique())
        n_missing = int(X_train[col].isnull().sum())
        top_cat = vc.index[0]
        top_freq = int(vc.iloc[0])
        lt_10 = int((vc < 10).sum())
        lt_50 = int((vc < 50).sum())
        lt_100 = int((vc < 100).sum())
        ratio = round(n_unique / n_rows, 6)
        
        records.append({
            "column_name": col,
            "unique_categories": n_unique,
            "missing_values": n_missing,
            "most_frequent_category": top_cat,
            "frequency_most_frequent": top_freq,
            "categories_lt_10": lt_10,
            "categories_lt_50": lt_50,
            "categories_lt_100": lt_100,
            "cardinality_ratio": ratio,
        })
        
    return pd.DataFrame(records)


def get_high_cardinality_feature(cardinality_summary_df: pd.DataFrame) -> Tuple[str, int]:
    """Identifies the feature with highest unique categories."""
    sorted_df = cardinality_summary_df.sort_values(by="unique_categories", ascending=False)
    top_row = sorted_df.iloc[0]
    return str(top_row["column_name"]), int(top_row["unique_categories"])


def calculate_feature_sparsity_metrics(series: pd.Series) -> Dict[str, Any]:
    """
    Calculates sparsity metrics for a categorical feature:
    - Frequency of every category
    - Count of categories appearing only once
    - Count appearing < 10 times
    - Count appearing < 50 times
    - Count appearing < 100 times
    - Percentage of observations belonging to rare categories (< 100)
    - Percentage of categories that are rare (< 100)
    """
    vc = series.value_counts()
    n_total_categories = len(vc)
    n_total_observations = len(series)
    
    once_count = int((vc == 1).sum())
    lt_10_count = int((vc < 10).sum())
    lt_50_count = int((vc < 50).sum())
    lt_100_count = int((vc < 100).sum())
    
    obs_lt_100 = int(vc[vc < 100].sum())
    pct_obs_rare_100 = (obs_lt_100 / n_total_observations) * 100
    pct_cats_rare_100 = (lt_100_count / n_total_categories) * 100
    
    return {
        "value_counts": vc,
        "n_categories": n_total_categories,
        "n_observations": n_total_observations,
        "once_count": once_count,
        "lt_10_count": lt_10_count,
        "lt_50_count": lt_50_count,
        "lt_100_count": lt_100_count,
        "obs_lt_100": obs_lt_100,
        "pct_obs_rare_100": pct_obs_rare_100,
        "pct_cats_rare_100": pct_cats_rare_100,
    }
