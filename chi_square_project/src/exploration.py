"""
Module: exploration.py
Description: Reusable functions for exploratory data analysis, categorical cardinality checks,
             and target distribution visualisations.
"""

from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_combine_data(features_path: Path, target_path: Path) -> pd.DataFrame:
    """Loads features and target CSV files and concatenates them into a single DataFrame."""
    X = pd.read_csv(features_path)
    y = pd.read_csv(target_path)
    df = pd.concat([X, y], axis=1)
    return df


def calculate_categorical_cardinality(df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
    """
    Computes cardinality and frequency statistics for specified categorical columns:
    - Column name
    - Number of unique categories
    - Number of missing values
    - Most frequent category
    - Frequency of most frequent category
    - Number of categories appearing fewer than 10 times
    - Number of categories appearing fewer than 50 times
    - Number of categories appearing fewer than 100 times
    """
    records = []
    for col in categorical_cols:
        vc = df[col].value_counts()
        n_unique = df[col].nunique()
        n_missing = df[col].isnull().sum()
        top_cat = vc.index[0]
        top_freq = vc.iloc[0]
        lt_10 = int((vc < 10).sum())
        lt_50 = int((vc < 50).sum())
        lt_100 = int((vc < 100).sum())
        
        records.append({
            "column_name": col,
            "unique_categories": n_unique,
            "missing_values": n_missing,
            "most_frequent_category": top_cat,
            "frequency_most_frequent": top_freq,
            "categories_lt_10": lt_10,
            "categories_lt_50": lt_50,
            "categories_lt_100": lt_100,
        })
        
    return pd.DataFrame(records)
