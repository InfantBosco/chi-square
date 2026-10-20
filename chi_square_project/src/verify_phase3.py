"""
Module: verify_phase3.py
Description: Validates Phase 3 train-test split outputs, row counts, and stratification balance.
"""

from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

X_TRAIN = PROJECT_ROOT / "data" / "processed" / "X_train.csv"
X_TEST = PROJECT_ROOT / "data" / "processed" / "X_test.csv"
Y_TRAIN = PROJECT_ROOT / "data" / "processed" / "y_train.csv"
Y_TEST = PROJECT_ROOT / "data" / "processed" / "y_test.csv"
SUMMARY_CSV = PROJECT_ROOT / "results" / "03_train_test_split_summary.csv"


def verify_phase3():
    print("Verifying Phase 3 deliverables...")
    assert X_TRAIN.exists(), f"Missing {X_TRAIN}"
    assert X_TEST.exists(), f"Missing {X_TEST}"
    assert Y_TRAIN.exists(), f"Missing {Y_TRAIN}"
    assert Y_TEST.exists(), f"Missing {Y_TEST}"
    assert SUMMARY_CSV.exists(), f"Missing {SUMMARY_CSV}"
    
    X_tr = pd.read_csv(X_TRAIN)
    X_te = pd.read_csv(X_TEST)
    y_tr = pd.read_csv(Y_TRAIN).squeeze()
    y_te = pd.read_csv(Y_TEST).squeeze()
    
    assert len(X_tr) == 154204, f"Unexpected X_train rows: {len(X_tr)}"
    assert len(X_te) == 38551, f"Unexpected X_test rows: {len(X_te)}"
    assert len(y_tr) == 154204, f"Unexpected y_train rows: {len(y_tr)}"
    assert len(y_te) == 38551, f"Unexpected y_test rows: {len(y_te)}"
    
    train_pct1 = (y_tr == 1).mean()
    test_pct1 = (y_te == 1).mean()
    assert abs(train_pct1 - test_pct1) < 0.001, "Stratification failure"
    
    summary_df = pd.read_csv(SUMMARY_CSV)
    assert len(summary_df) == 3, "Summary should have 3 rows"
    print("All Phase 3 verification assertions passed successfully!")


if __name__ == "__main__":
    verify_phase3()
