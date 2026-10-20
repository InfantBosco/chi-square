"""
Module: verify_phase5.py
Description: Validates Phase 5 baseline chi-square deliverables and metrics.
"""

from pathlib import Path
import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

RES_CSV = PROJECT_ROOT / "results" / "05_baseline_chi_square_results.csv"
CONT_CSV = PROJECT_ROOT / "results" / "05_baseline_contingency_table.csv"
EXP_CSV = PROJECT_ROOT / "results" / "05_baseline_expected_counts.csv"
HEATMAP = PROJECT_ROOT / "figures" / "05_baseline_top_categories_heatmap.png"
NOTEBOOK = PROJECT_ROOT / "notebooks" / "05_chi_square_baseline.ipynb"


def verify_phase5():
    print("Verifying Phase 5 deliverables...")
    assert RES_CSV.exists(), f"Missing {RES_CSV}"
    assert CONT_CSV.exists(), f"Missing {CONT_CSV}"
    assert EXP_CSV.exists(), f"Missing {EXP_CSV}"
    assert HEATMAP.exists(), f"Missing {HEATMAP}"
    assert NOTEBOOK.exists(), f"Missing {NOTEBOOK}"
    
    res_df = pd.read_csv(RES_CSV)
    assert len(res_df) == 1, "Expected 1 row in results"
    row = res_df.iloc[0]
    
    assert row["Number of categories"] == 529
    assert row["Degrees of freedom"] == 528
    assert np.isclose(row["Chi-square statistic"], 51535.8043, rtol=1e-3)
    assert row["p-value"] == 0.0
    assert row["Number of cells with expected count below 5"] == 60
    assert row["Number of categories with fewer than 10 observations"] == 23
    assert row["Number of categories with fewer than 50 observations"] == 172
    assert row["Number of categories with fewer than 100 observations"] == 271
    
    print("All Phase 5 verification checks passed successfully!")


if __name__ == "__main__":
    verify_phase5()
