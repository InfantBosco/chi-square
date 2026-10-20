"""
Module: split.py
Description: Stratified train-test splitting utilities ensuring clean data isolation
             without data leakage.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, List
import pandas as pd
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
