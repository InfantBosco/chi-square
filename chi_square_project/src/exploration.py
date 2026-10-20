"""
Module: exploration.py
Description: Reusable functions for exploratory data analysis, categorical cardinality checks,
             and target distribution visualisations.
"""

from pathlib import Path
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
