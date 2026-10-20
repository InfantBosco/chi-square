"""
Module: verify_phase6.py
Description: Validates Phase 6 cardinality reduction deliverables and summary values.
"""

from pathlib import Path
import json
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

MAPPING_JSON = PROJECT_ROOT / "results" / "06_category_mapping.json"
SUMMARY_CSV = PROJECT_ROOT / "results" / "06_cardinality_reduction_summary.csv"
X_GROUPED_CSV = PROJECT_ROOT / "data" / "processed" / "X_train_grouped.csv"
FIG_BEFORE = PROJECT_ROOT / "figures" / "06_categories_before_grouping.png"
FIG_AFTER = PROJECT_ROOT / "figures" / "06_categories_after_grouping.png"
NOTEBOOK = PROJECT_ROOT / "notebooks" / "06_cardinality_reduction.ipynb"


def verify_phase6():
    print("Verifying Phase 6 deliverables...")
    assert MAPPING_JSON.exists(), f"Missing {MAPPING_JSON}"
    assert SUMMARY_CSV.exists(), f"Missing {SUMMARY_CSV}"
    assert X_GROUPED_CSV.exists(), f"Missing {X_GROUPED_CSV}"
    assert FIG_BEFORE.exists(), f"Missing {FIG_BEFORE}"
    assert FIG_AFTER.exists(), f"Missing {FIG_AFTER}"
    assert NOTEBOOK.exists(), f"Missing {NOTEBOOK}"
    
    with open(MAPPING_JSON, "r") as f:
        mapping = json.load(f)
    assert len(mapping) == 529, f"Expected 529 categories in mapping, got {len(mapping)}"
    
    summary_df = pd.read_csv(SUMMARY_CSV)
    row = summary_df.iloc[0]
    assert row["Original number of categories"] == 529
    assert row["Transformed number of categories"] == 259
    assert row["Number of categories grouped as Rare"] == 271
    assert row["Number of observations assigned to Rare"] == 11326
    assert abs(row["Percentage of observations assigned to Rare"] - 7.3448) < 0.01
    assert abs(row["Cardinality reduction percentage"] - 51.0397) < 0.01
    
    X_gr = pd.read_csv(X_GROUPED_CSV)
    assert "OCCP_grouped" in X_gr.columns
    assert len(X_gr) == 154204
    
    print("All Phase 6 verification checks passed successfully!")


if __name__ == "__main__":
    verify_phase6()
