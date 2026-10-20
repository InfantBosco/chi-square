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
