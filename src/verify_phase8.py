"""
Module: verify_phase8.py
Description: Validates Phase 8 test transformation deliverables, column names, and audit statistics.
"""

from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

X_TEST_GROUPED = PROJECT_ROOT / "data" / "processed" / "X_test_grouped.csv"
REPORT_CSV = PROJECT_ROOT / "results" / "08_test_transformation_report.csv"
NOTEBOOK = PROJECT_ROOT / "notebooks" / "08_apply_transformation_to_test.ipynb"


def verify_phase8():
    print("Verifying Phase 8 deliverables...")
    assert X_TEST_GROUPED.exists(), f"Missing {X_TEST_GROUPED}"
    assert REPORT_CSV.exists(), f"Missing {REPORT_CSV}"
    assert NOTEBOOK.exists(), f"Missing {NOTEBOOK}"
    
    df_test = pd.read_csv(X_TEST_GROUPED)
    assert len(df_test) == 38551, f"Expected 38,551 rows, got {len(df_test)}"
    assert "OCCP_grouped" in df_test.columns, "OCCP_grouped not found in test columns"
    assert "OCCP" in df_test.columns, "Original OCCP column was not preserved"
    
    report_df = pd.read_csv(REPORT_CSV)
    row = report_df.iloc[0]
    assert row["Number of test categories before transformation"] == 518
    assert row["Number of test categories after transformation"] == 259
    assert row["Number of unseen test categories"] == 0
    assert row["Number of test observations assigned to Rare"] == 2840
    assert abs(row["Percentage of test observations assigned to Rare"] - 7.3669) < 0.01
    
    print("All Phase 8 verification assertions passed successfully!")


if __name__ == "__main__":
    verify_phase8()
