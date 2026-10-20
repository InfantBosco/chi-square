"""
Module: verify_download.py
Description: Validates downloaded ACSIncome datasets against project requirements.
"""

from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

FEATURES_CSV = PROJECT_ROOT / "data" / "raw" / "acs_income_features.csv"
TARGET_CSV = PROJECT_ROOT / "data" / "raw" / "acs_income_target.csv"
SUMMARY_TXT = PROJECT_ROOT / "results" / "00_dataset_download_summary.txt"

EXPECTED_COLUMNS = [
    "AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P"
]


def verify():
    print("Verifying downloaded dataset and summary files...")
    
    assert FEATURES_CSV.exists(), f"Features file not found at {FEATURES_CSV}"
    assert TARGET_CSV.exists(), f"Target file not found at {TARGET_CSV}"
    assert SUMMARY_TXT.exists(), f"Summary file not found at {SUMMARY_TXT}"
    
    X = pd.read_csv(FEATURES_CSV)
    y = pd.read_csv(TARGET_CSV)
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    assert len(X) == len(y), "Row counts of features and target do not match"
    assert list(X.columns) == EXPECTED_COLUMNS, f"Columns mismatch: {list(X.columns)}"
    assert y.shape[1] == 1, "Target should have exactly 1 column"
    
    print("All verification checks passed successfully!")


if __name__ == "__main__":
    verify()
