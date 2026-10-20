"""
Module: verify_phase7.py
Description: Validates Phase 7 transformed chi-square deliverables, comparison metrics,
             and absence of low-expected-count cells.
"""

from pathlib import Path
import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

RES_CSV = PROJECT_ROOT / "results" / "07_transformed_chi_square_results.csv"
CONT_CSV = PROJECT_ROOT / "results" / "07_transformed_contingency_table.csv"
EXP_CSV = PROJECT_ROOT / "results" / "07_transformed_expected_counts.csv"
COMP_CSV = PROJECT_ROOT / "results" / "07_chi_square_before_after_comparison.csv"
FIG_COMP = PROJECT_ROOT / "figures" / "07_chi_square_before_after_comparison.png"
NOTEBOOK = PROJECT_ROOT / "notebooks" / "07_chi_square_transformed.ipynb"


def verify_phase7():
    print("Verifying Phase 7 deliverables...")
    assert RES_CSV.exists(), f"Missing {RES_CSV}"
    assert CONT_CSV.exists(), f"Missing {CONT_CSV}"
    assert EXP_CSV.exists(), f"Missing {EXP_CSV}"
    assert COMP_CSV.exists(), f"Missing {COMP_CSV}"
    assert FIG_COMP.exists(), f"Missing {FIG_COMP}"
    assert NOTEBOOK.exists(), f"Missing {NOTEBOOK}"
    
    res_df = pd.read_csv(RES_CSV)
    row = res_df.iloc[0]
    assert row["Number of transformed categories"] == 259
    assert row["Degrees of freedom"] == 258
    assert np.isclose(row["Chi-square statistic"], 48444.0265, rtol=1e-3)
    assert row["p-value"] == 0.0
    assert row["Number of cells with expected count below 5"] == 0
    assert row["Percentage of cells with expected count below 5"] == 0.0
    assert row["Number of categories with fewer than 10 observations"] == 0
    assert row["Number of categories with fewer than 50 observations"] == 0
    
    comp_df = pd.read_csv(COMP_CSV)
    assert len(comp_df) == 2, "Expected 2 rows in comparison table"
    assert comp_df.iloc[0]["Number of low expected cells"] == 60
    assert comp_df.iloc[1]["Number of low expected cells"] == 0
    
    print("All Phase 7 verification assertions passed successfully!")


if __name__ == "__main__":
    verify_phase7()
