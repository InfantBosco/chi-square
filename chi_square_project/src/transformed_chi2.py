"""
Module: transformed_chi2.py
Description: Computes Chi-Square tests of independence on transformed/grouped features,
             generates comparison metrics, and plots before-and-after evaluations.
"""

from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency


def compute_transformed_contingency(
    X_train_grouped: pd.DataFrame,
    y_train: pd.Series,
    grouped_col_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Computes contingency table, expected counts, and chi-square statistics for grouped feature.
    """
    ct = pd.crosstab(
        X_train_grouped[grouped_col_name],
        y_train,
        rownames=[grouped_col_name],
        colnames=["income_above_50k"]
    )
    
    observed = ct.values.astype(float)
    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    total_obs = observed.sum()
    
    expected = (row_totals * col_totals) / total_obs
    expected_df = pd.DataFrame(expected, index=ct.index, columns=ct.columns)
    
    manual_chi2 = float(np.sum(((observed - expected) ** 2) / expected))
    scipy_res = chi2_contingency(ct, correction=False)
    scipy_chi2 = float(scipy_res.statistic)
    dof = int(scipy_res.dof)
    p_val = float(scipy_res.pvalue)
    
    cells_lt_5 = int((expected < 5).sum())
    total_cells = int(expected.size)
    pct_cells_lt_5 = float((cells_lt_5 / total_cells) * 100)
    
    cat_totals = row_totals.flatten()
    cats_lt_10 = int((cat_totals < 10).sum())
    cats_lt_50 = int((cat_totals < 50).sum())
    
    metrics = {
        "Number of transformed categories": len(ct),
        "Chi-square statistic": round(manual_chi2, 4),
        "Degrees of freedom": dof,
        "p-value": p_val,
        "Number of cells with expected count below 5": cells_lt_5,
        "Percentage of cells with expected count below 5": round(pct_cells_lt_5, 4),
        "Number of categories with fewer than 10 observations": cats_lt_10,
        "Number of categories with fewer than 50 observations": cats_lt_50,
        "scipy_chi2": scipy_chi2,
    }
    return ct, expected_df, metrics


def build_before_after_comparison(
    baseline_results: Dict[str, Any],
    transformed_results: Dict[str, Any]
) -> pd.DataFrame:
    """
    Constructs comparison table containing:
    - Feature version
    - Number of categories
    - Chi-square statistic
    - Degrees of freedom
    - p-value
    - Number of low expected cells
    - Percentage of low expected cells
    """
    records = [
        {
            "Feature version": "Baseline (OCCP)",
            "Number of categories": int(baseline_results["Number of categories"]),
            "Chi-square statistic": round(float(baseline_results["Chi-square statistic"]), 4),
            "Degrees of freedom": int(baseline_results["Degrees of freedom"]),
            "p-value": float(baseline_results["p-value"]),
            "Number of low expected cells": int(baseline_results["Number of cells with expected count below 5"]),
            "Percentage of low expected cells": round(float(baseline_results["Percentage of cells with expected count below 5"]), 4),
        },
        {
            "Feature version": "Transformed (OCCP_grouped)",
            "Number of categories": int(transformed_results["Number of transformed categories"]),
            "Chi-square statistic": round(float(transformed_results["Chi-square statistic"]), 4),
            "Degrees of freedom": int(transformed_results["Degrees of freedom"]),
            "p-value": float(transformed_results["p-value"]),
            "Number of low expected cells": int(transformed_results["Number of cells with expected count below 5"]),
            "Percentage of low expected cells": round(float(transformed_results["Percentage of cells with expected count below 5"]), 4),
        }
    ]
    return pd.DataFrame(records)
