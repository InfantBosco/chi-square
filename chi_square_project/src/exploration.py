"""
Module: exploration.py
Description: Reusable functions for exploratory data analysis, categorical cardinality checks,
             and target distribution visualisations.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple
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
        n_missing = int(df[col].isnull().sum())
        top_cat = vc.index[0]
        top_freq = int(vc.iloc[0])
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


def analyze_and_plot_occp(df: pd.DataFrame, output_path: Path) -> Tuple[int, pd.Series, Dict[str, int]]:
    """
    Analyzes OCCP (Occupation Code):
    - Counts unique categories
    - Identifies top 20 most frequent categories
    - Computes categories with <10, <50, <100 observations
    - Plots frequency distribution and saves to output_path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    occp_counts = df["OCCP"].value_counts()
    n_unique = df["OCCP"].nunique()
    top_20 = occp_counts.head(20)
    
    threshold_counts = {
        "fewer_than_10": int((occp_counts < 10).sum()),
        "fewer_than_50": int((occp_counts < 50).sum()),
        "fewer_than_100": int((occp_counts < 100).sum()),
    }
    
    # Plot frequency distribution
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(range(len(occp_counts)), occp_counts.values, color="#1f77b4", lw=2)
    plt.fill_between(range(len(occp_counts)), occp_counts.values, color="#1f77b4", alpha=0.2)
    plt.title("OCCP (Occupation Code) Category Frequency Distribution", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Category Rank (Ordered by Frequency)", fontsize=12)
    plt.ylabel("Category Frequency (Count)", fontsize=12)
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.5, which="both")
    
    # Annotate key characteristics
    plt.axvline(x=len(occp_counts) - threshold_counts["fewer_than_100"], color="#d62728", linestyle=":", label=f"<100 count: {threshold_counts['fewer_than_100']} categories")
    plt.axvline(x=len(occp_counts) - threshold_counts["fewer_than_50"], color="#ff7f0e", linestyle=":", label=f"<50 count: {threshold_counts['fewer_than_50']} categories")
    plt.axvline(x=len(occp_counts) - threshold_counts["fewer_than_10"], color="#2ca02c", linestyle=":", label=f"<10 count: {threshold_counts['fewer_than_10']} categories")
    
    plt.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return n_unique, top_20, threshold_counts


def analyze_and_plot_pincp(df: pd.DataFrame, output_path: Path) -> Dict[str, float]:
    """
    Analyzes PINCP target column:
    - Computes min, max, mean, median, Q1 (25%), Q3 (75%)
    - Plots distribution histogram with median and mean markers
    - Saves plot to output_path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pincp = df["PINCP"]
    
    stats = {
        "min": float(pincp.min()),
        "q1": float(pincp.quantile(0.25)),
        "median": float(pincp.median()),
        "mean": float(pincp.mean()),
        "q3": float(pincp.quantile(0.75)),
        "max": float(pincp.max()),
        "std": float(pincp.std()),
    }
    
    plt.figure(figsize=(10, 6), dpi=300)
    # Clip view to 99th percentile for clear histogram visibility while plotting
    q99 = pincp.quantile(0.99)
    sns.histplot(pincp[pincp <= q99], bins=60, color="#2ca02c", kde=True, edgecolor="black", alpha=0.6)
    
    plt.axvline(stats["median"], color="#1f77b4", linestyle="--", lw=2, label=f"Median: ${stats['median']:,.0f}")
    plt.axvline(stats["mean"], color="#d62728", linestyle="-", lw=2, label=f"Mean: ${stats['mean']:,.0f}")
    plt.axvline(stats["q1"], color="#ff7f0e", linestyle=":", lw=1.5, label=f"Q1 (25%): ${stats['q1']:,.0f}")
    plt.axvline(stats["q3"], color="#9467bd", linestyle=":", lw=1.5, label=f"Q3 (75%): ${stats['q3']:,.0f}")
    
    plt.title("Distribution of Total Person's Income (PINCP) - CA 2018 (up to 99th percentile)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Annual Income (USD)", fontsize=12)
    plt.ylabel("Frequency (Person Count)", fontsize=12)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${int(x):,}"))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return stats

