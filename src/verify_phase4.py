"""
Module: verify_phase4.py
Description: Validates Phase 4 high-cardinality deliverables, feature selection,
             sparsity counts, and plot outputs.
"""

from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

CARD_CSV = PROJECT_ROOT / "results" / "04_training_cardinality_summary.csv"
FEATURE_TXT = PROJECT_ROOT / "results" / "04_selected_high_cardinality_feature.txt"
FREQ_CSV = PROJECT_ROOT / "results" / "04_selected_feature_category_frequencies.csv"
FIG_TOP = PROJECT_ROOT / "figures" / "04_high_cardinality_top_categories.png"
FIG_DIST = PROJECT_ROOT / "figures" / "04_high_cardinality_frequency_distribution.png"
FIG_CUM = PROJECT_ROOT / "figures" / "04_high_cardinality_cumulative_frequency.png"


def verify_phase4():
    print("Verifying Phase 4 deliverables...")
    assert CARD_CSV.exists(), f"Missing {CARD_CSV}"
    assert FEATURE_TXT.exists(), f"Missing {FEATURE_TXT}"
    assert FREQ_CSV.exists(), f"Missing {FREQ_CSV}"
    assert FIG_TOP.exists(), f"Missing {FIG_TOP}"
    assert FIG_DIST.exists(), f"Missing {FIG_DIST}"
    assert FIG_CUM.exists(), f"Missing {FIG_CUM}"
    
    # Check feature name text file
    feature_name = FEATURE_TXT.read_text().strip()
    assert feature_name == "OCCP", f"Expected 'OCCP', got {feature_name}"
    
    # Check cardinality summary
    card_df = pd.read_csv(CARD_CSV)
    assert len(card_df) == 8, f"Expected 8 categorical columns, got {len(card_df)}"
    occp_row = card_df[card_df["column_name"] == "OCCP"].iloc[0]
    assert occp_row["unique_categories"] == 529
    assert occp_row["categories_lt_100"] == 271
    
    # Check detailed frequency table
    freq_df = pd.read_csv(FREQ_CSV)
    expected_cols = [
        "Category value",
        "Observation count",
        "Percentage of total observations",
        "Whether it is rare under the 100-observation threshold"
    ]
    assert list(freq_df.columns) == expected_cols, f"Columns mismatch: {list(freq_df.columns)}"
    assert len(freq_df) == 529, f"Expected 529 rows, got {len(freq_df)}"
    
    print("All Phase 4 verification checks passed successfully!")


if __name__ == "__main__":
    verify_phase4()
