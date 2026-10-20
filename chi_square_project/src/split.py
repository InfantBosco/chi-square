"""
Module: split.py
Description: Stratified train-test splitting utilities ensuring clean data isolation
             without data leakage.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def perform_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs stratified train-test split on features and target.
    
    Parameters:
        X: Feature matrix DataFrame
        y: Binary target Series
        test_size: Proportion of test set (default 0.20)
        random_state: Random seed for reproducibility (default 42)
        
    Returns:
        (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


def generate_split_summary(y_full: pd.Series, y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    """
    Generates summary table containing:
    - Dataset name
    - Number of rows
    - Number of target class 0
    - Percentage of target class 0
    - Number of target class 1
    - Percentage of target class 1
    """
    records = []
    datasets = [
        ("Full Dataset", y_full),
        ("Training Set", y_train),
        ("Testing Set", y_test),
    ]
    for name, series in datasets:
        counts = series.value_counts().sort_index()
        pcts = (counts / len(series)) * 100
        records.append({
            "Dataset name": name,
            "Number of rows": int(len(series)),
            "Number of target class 0": int(counts[0]),
            "Percentage of target class 0": round(float(pcts[0]), 4),
            "Number of target class 1": int(counts[1]),
            "Percentage of target class 1": round(float(pcts[1]), 4),
        })
    return pd.DataFrame(records)


def verify_split_proportions(y_train: pd.Series, y_test: pd.Series, tolerance: float = 0.001) -> bool:
    """
    Verifies that the target proportions between train and test sets match within tolerance.
    """
    train_pct1 = (y_train == 1).mean()
    test_pct1 = (y_test == 1).mean()
    diff = abs(train_pct1 - test_pct1)
    assert diff < tolerance, f"Target proportions mismatch: Train={train_pct1:.4f}, Test={test_pct1:.4f}, Diff={diff:.4f}"
    return True
