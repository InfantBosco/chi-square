"""
Module: verify_phase1.py
Description: Validates Phase 1 deliverables, including summary table statistics,
             figure existence, and notebook output compliance.
"""

from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

SUMMARY_CSV = PROJECT_ROOT / "results" / "01_categorical_cardinality_summary.csv"
OCCP_FIG = PROJECT_ROOT / "figures" / "01_occp_frequency_distribution.png"
INCOME_FIG = PROJECT_ROOT / "figures" / "01_income_distribution.png"
NOTEBOOK_FILE = PROJECT_ROOT / "notebooks" / "01_data_exploration.ipynb"


def verify_phase1():
    print("Checking Phase 1 deliverables...")
    assert SUMMARY_CSV.exists(), f"Missing {SUMMARY_CSV}"
    assert OCCP_FIG.exists(), f"Missing {OCCP_FIG}"
    assert INCOME_FIG.exists(), f"Missing {INCOME_FIG}"
    assert NOTEBOOK_FILE.exists(), f"Missing {NOTEBOOK_FILE}"
    
    summary_df = pd.read_csv(SUMMARY_CSV)
    print("Summary columns:", list(summary_df.columns))
    assert "column_name" in summary_df.columns
    assert "unique_categories" in summary_df.columns
    assert len(summary_df) == 8, f"Expected 8 categorical columns, got {len(summary_df)}"
    
    occp_row = summary_df[summary_df["column_name"] == "OCCP"].iloc[0]
    assert occp_row["unique_categories"] == 529
    assert occp_row["categories_lt_10"] == 14
    assert occp_row["categories_lt_50"] == 139
    assert occp_row["categories_lt_100"] == 235
    
    print("All Phase 1 verifications passed successfully!")


if __name__ == "__main__":
    verify_phase1()
