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


def plot_top_categories(series: pd.Series, output_path: Path, top_n: int = 30):
    """Plots horizontal bar chart of top N most frequent categories."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vc = series.value_counts().head(top_n)
    
    plt.figure(figsize=(10, 8), dpi=300)
    y_pos = np.arange(len(vc))
    plt.barh(y_pos, vc.values, color="#1f77b4", edgecolor="black", alpha=0.85)
    plt.yticks(y_pos, [f"Cat {int(cat)}" if isinstance(cat, (int, float)) and not np.isnan(cat) else str(cat) for cat in vc.index], fontsize=9)
    plt.gca().invert_yaxis()
    plt.xlabel("Observation Count (Training Set)", fontsize=11)
    plt.ylabel(f"Category (Top {top_n})", fontsize=11)
    plt.title(f"Top {top_n} Most Frequent Categories for {series.name}", fontsize=13, fontweight="bold", pad=12)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_frequency_distribution(series: pd.Series, output_path: Path):
    """Plots category-frequency distribution on log scale."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vc = series.value_counts()
    ranks = np.arange(len(vc))
    
    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.plot(ranks, vc.values, color="#2ca02c", lw=2.2, label=f"{series.name} Frequency Rank")
    plt.fill_between(ranks, vc.values, color="#2ca02c", alpha=0.2)
    
    plt.axhline(100, color="#d62728", linestyle="--", lw=1.2, label="Threshold = 100 observations")
    plt.axhline(50, color="#ff7f0e", linestyle="-.", lw=1.2, label="Threshold = 50 observations")
    plt.axhline(10, color="#9467bd", linestyle=":", lw=1.2, label="Threshold = 10 observations")
    
    plt.yscale("log")
    plt.xlabel("Category Rank (Ordered by Descending Frequency)", fontsize=11)
    plt.ylabel("Observation Count (Log Scale)", fontsize=11)
    plt.title(f"Category Frequency Distribution for {series.name} (Zipfian Long Tail)", fontsize=13, fontweight="bold", pad=12)
    plt.grid(True, linestyle="--", alpha=0.5, which="both")
    plt.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_cumulative_frequency(series: pd.Series, output_path: Path):
    """Plots cumulative percentage of observations vs cumulative percentage of categories."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vc = series.value_counts()
    cum_obs_pct = (np.cumsum(vc.values) / len(series)) * 100
    cum_cat_pct = ((np.arange(len(vc)) + 1) / len(vc)) * 100
    
    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.plot(cum_cat_pct, cum_obs_pct, color="#9467bd", lw=2.5, label="Cumulative Coverage")
    plt.plot([0, 100], [0, 100], color="gray", linestyle="--", lw=1.2, label="Uniform Baseline")
    
    # Annotate Pareto points
    idx_50 = np.searchsorted(cum_obs_pct, 50)
    idx_80 = np.searchsorted(cum_obs_pct, 80)
    
    plt.scatter([cum_cat_pct[idx_50], cum_cat_pct[idx_80]], [cum_obs_pct[idx_50], cum_obs_pct[idx_80]], color="#d62728", zorder=5)
    plt.annotate(f"50% obs covered by top {cum_cat_pct[idx_50]:.1f}% cats", 
                 (cum_cat_pct[idx_50], cum_obs_pct[idx_50]),
                 xytext=(cum_cat_pct[idx_50] + 5, cum_obs_pct[idx_50] - 8),
                 arrowprops=dict(arrowstyle="->", color="#d62728"))
    plt.annotate(f"80% obs covered by top {cum_cat_pct[idx_80]:.1f}% cats", 
                 (cum_cat_pct[idx_80], cum_obs_pct[idx_80]),
                 xytext=(cum_cat_pct[idx_80] + 5, cum_obs_pct[idx_80] - 8),
                 arrowprops=dict(arrowstyle="->", color="#d62728"))
                 
    plt.xlabel("Cumulative Percentage of Categories (%)", fontsize=11)
    plt.ylabel("Cumulative Percentage of Observations (%)", fontsize=11)
    plt.title(f"Cumulative Frequency (Lorenz/Pareto) Curve for {series.name}", fontsize=13, fontweight="bold", pad=12)
    plt.xlim(0, 100)
    plt.ylim(0, 105)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

