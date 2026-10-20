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
