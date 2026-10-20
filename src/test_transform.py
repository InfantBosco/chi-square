"""
Module: test_transform.py
Description: Applies training-learned category mapping to test set strictly without data leakage.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import json
import pandas as pd


def apply_transformation_to_test(
    X_test: pd.DataFrame,
    feature_name: str,
    mapping: Dict[str, str]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Applies the training-derived mapping to X_test.
    Any unseen category is mapped to 'Rare'.
    """
    df_out = X_test.copy()
    grouped_col = f"{feature_name}_grouped"
    
    test_orig_cats = set(df_out[feature_name].astype(str).unique())
    mapping_keys = set(mapping.keys())
    unseen_cats = test_orig_cats - mapping_keys
    
    # Apply mapping with fallback to 'Rare'
    df_out[grouped_col] = df_out[feature_name].astype(str).map(
        lambda x: mapping.get(x, "Rare")
    )
    
    n_orig_cats = len(test_orig_cats)
    n_trans_cats = int(df_out[grouped_col].nunique())
    n_unseen = len(unseen_cats)
    n_rare_obs = int((df_out[grouped_col] == "Rare").sum())
    total_obs = len(df_out)
    pct_rare_obs = round(float((n_rare_obs / total_obs) * 100), 4)
    
    report = {
        "Number of test categories before transformation": n_orig_cats,
        "Number of test categories after transformation": n_trans_cats,
        "Number of unseen test categories": n_unseen,
        "Number of test observations assigned to Rare": n_rare_obs,
        "Percentage of test observations assigned to Rare": pct_rare_obs,
    }
    
    return df_out, report
