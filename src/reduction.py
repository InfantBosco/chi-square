"""
Module: reduction.py
Description: Category grouping and cardinality reduction for high-cardinality features.
             Learns mapping strictly from training data and pools rare categories.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def build_category_mapping(series: pd.Series, threshold: int = 100) -> Dict[str, str]:
    """
    Builds a category mapping dictionary based on minimum frequency threshold.
    Categories with frequency >= threshold retain their category value.
    Categories with frequency < threshold are mapped to 'Rare'.
    """
    vc = series.value_counts()
    mapping = {}
    for cat, count in vc.items():
        mapping[str(cat)] = str(cat) if count >= threshold else "Rare"
    return mapping


def apply_category_reduction(
    df: pd.DataFrame,
    feature_name: str,
    mapping: Dict[str, str]
) -> pd.DataFrame:
    """
    Applies the learned category mapping to create <feature_name>_grouped
    without modifying the original feature column.
    """
    df_out = df.copy()
    grouped_col_name = f"{feature_name}_grouped"
    
    # Map unseen or rare categories gracefully to 'Rare'
    df_out[grouped_col_name] = df_out[feature_name].astype(str).map(
        lambda x: mapping.get(x, "Rare")
    )
    return df_out


def generate_reduction_summary(
    orig_series: pd.Series,
    grouped_series: pd.Series,
    mapping: Dict[str, str]
) -> pd.DataFrame:
    """
    Generates comparison summary table containing:
    - Original number of categories
    - Transformed number of categories
    - Number of categories grouped as Rare
    - Number of observations assigned to Rare
    - Percentage of observations assigned to Rare
    - Cardinality reduction percentage
    """
    n_orig_cats = int(orig_series.nunique())
    n_trans_cats = int(grouped_series.nunique())
    n_rare_cats = sum(1 for v in mapping.values() if v == "Rare")
    n_rare_obs = int((grouped_series == "Rare").sum())
    total_obs = len(grouped_series)
    pct_rare_obs = round(float((n_rare_obs / total_obs) * 100), 4)
    card_red_pct = round(float(((n_orig_cats - n_trans_cats) / n_orig_cats) * 100), 4)
    
    record = {
        "Original number of categories": n_orig_cats,
        "Transformed number of categories": n_trans_cats,
        "Number of categories grouped as Rare": n_rare_cats,
        "Number of observations assigned to Rare": n_rare_obs,
        "Percentage of observations assigned to Rare": pct_rare_obs,
        "Cardinality reduction percentage": card_red_pct,
    }
    return pd.DataFrame([record])


def plot_categories_before_grouping(orig_series: pd.Series, output_path: Path):
    """Plots category counts before grouping (highlighting the long tail)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vc = orig_series.value_counts()
    ranks = np.arange(len(vc))
    
    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.plot(ranks, vc.values, color="#d62728", lw=2, label="Original Category Counts")
    plt.fill_between(ranks, vc.values, color="#d62728", alpha=0.15)
    plt.axhline(100, color="#1f77b4", linestyle="--", lw=1.5, label="Threshold = 100 observations")
    plt.yscale("log")
    plt.title(f"Category Frequencies Before Grouping ({orig_series.name}: {len(vc)} Categories)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Category Rank (Ordered by Descending Frequency)", fontsize=11)
    plt.ylabel("Observation Count (Log Scale)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5, which="both")
    plt.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_categories_after_grouping(grouped_series: pd.Series, output_path: Path):
    """Plots category counts after grouping (showing eliminated tail)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vc = grouped_series.value_counts()
    ranks = np.arange(len(vc))
    
    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.plot(ranks, vc.values, color="#2ca02c", lw=2, label="Transformed Category Counts")
    plt.fill_between(ranks, vc.values, color="#2ca02c", alpha=0.15)
    plt.axhline(100, color="#1f77b4", linestyle="--", lw=1.5, label="Threshold = 100 observations")
    plt.yscale("log")
    plt.title(f"Category Frequencies After Grouping ({grouped_series.name}: {len(vc)} Categories)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Category Rank (Ordered by Descending Frequency)", fontsize=11)
    plt.ylabel("Observation Count (Log Scale)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5, which="both")
    plt.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

