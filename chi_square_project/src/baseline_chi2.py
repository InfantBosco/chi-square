"""
Module: baseline_chi2.py
Description: Baseline Chi-Square test of independence on raw high-cardinality features.
             Computes manual and scipy chi2 statistics, contingency tables, expected counts,
             and top-category heatmaps.
"""

from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency


def compute_baseline_contingency(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Computes contingency table, expected counts, and chi-square statistics.
    """
    contingency_table = pd.crosstab(
        X_train[feature_name],
        y_train,
        rownames=[feature_name],
        colnames=["income_above_50k"]
    )
    
    observed = contingency_table.values
    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    total_obs = observed.sum()
    
    # Manual expected counts
    expected = (row_totals * col_totals) / total_obs
    expected_df = pd.DataFrame(
        expected,
        index=contingency_table.index,
        columns=contingency_table.columns
    )
    
    # Manual chi-square statistic
    manual_chi2 = float(np.sum(((observed - expected) ** 2) / expected))
    
    # Scipy chi-square test
    scipy_res = chi2_contingency(contingency_table, correction=False)
    scipy_chi2 = float(scipy_res.statistic)
    dof = int(scipy_res.dof)
    p_val = float(scipy_res.pvalue)
    
    # Expected counts below 5
    cells_lt_5 = int((expected < 5).sum())
    total_cells = int(expected.size)
    pct_cells_lt_5 = float((cells_lt_5 / total_cells) * 100)
    
    # Sparsity thresholds on category totals
    cat_totals = row_totals.flatten()
    cats_lt_10 = int((cat_totals < 10).sum())
    cats_lt_50 = int((cat_totals < 50).sum())
    cats_lt_100 = int((cat_totals < 100).sum())
    
    metrics = {
        "Number of categories": len(contingency_table),
        "Chi-square statistic": manual_chi2,
        "Degrees of freedom": dof,
        "p-value": p_val,
        "Number of cells with expected count below 5": cells_lt_5,
        "Percentage of cells with expected count below 5": round(pct_cells_lt_5, 4),
        "Number of categories with fewer than 10 observations": cats_lt_10,
        "Number of categories with fewer than 50 observations": cats_lt_50,
        "Number of categories with fewer than 100 observations": cats_lt_100,
        "scipy_chi2": scipy_chi2,
    }
    
    return contingency_table, expected_df, metrics


def plot_top_categories_heatmap(
    contingency_table: pd.DataFrame,
    output_path: Path,
    top_n: int = 30
):
    """
    Creates a heatmap of top N categories by total observation frequency,
    showing counts across target classes 0 and 1.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Rank by row total count
    row_sums = contingency_table.sum(axis=1)
    top_cats = row_sums.sort_values(ascending=False).head(top_n).index
    top_table = contingency_table.loc[top_cats].copy()
    
    # Label columns and index nicely
    top_table.columns = ["<= $50k (Class 0)", "> $50k (Class 1)"]
    top_table.index = [f"OCCP {int(idx)}" if isinstance(idx, (int, float)) and not np.isnan(idx) else str(idx) for idx in top_table.index]
    
    plt.figure(figsize=(8, 11), dpi=300)
    sns.heatmap(
        top_table,
        annot=True,
        fmt=",d",
        cmap="YlGnBu",
        cbar_kws={"label": "Observed Observation Count"},
        linewidths=0.5,
        linecolor="#dddddd"
    )
    plt.title(f"Top {top_n} OCCP Categories Contingency Heatmap", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Occupation Category Code", fontsize=11)
    plt.xlabel("Target Income Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
