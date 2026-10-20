"""
Verification script for Phase 9 deliverables
"""

import os
import json
import pandas as pd

def verify_phase9():
    print("=" * 60)
    print("VERIFYING PHASE 9 DELIVERABLES")
    print("=" * 60)

    # 1. Check results CSV
    results_path = "results/09_model_comparison.csv"
    assert os.path.exists(results_path), f"Missing {results_path}"
    df = pd.read_csv(results_path)
    expected_cols = [
        "Model name",
        "Number of features",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
        "Training time (seconds)",
        "Prediction time (seconds)"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column {col} in {results_path}"
    assert len(df) == 2, f"Expected 2 rows in {results_path}, got {len(df)}"
    print(f"[OK] {results_path} contains all required columns and rows:")
    print(df.to_string(index=False))

    # 2. Check figures
    fig1 = "figures/09_model_performance_comparison.png"
    fig2 = "figures/09_encoded_feature_count_comparison.png"
    assert os.path.exists(fig1) and os.path.getsize(fig1) > 1000, f"Invalid or missing {fig1}"
    assert os.path.exists(fig2) and os.path.getsize(fig2) > 1000, f"Invalid or missing {fig2}"
    print(f"[OK] {fig1} exists ({os.path.getsize(fig1):,} bytes)")
    print(f"[OK] {fig2} exists ({os.path.getsize(fig2):,} bytes)")

    # 3. Check notebook
    nb_path = "notebooks/09_model_comparison.ipynb"
    assert os.path.exists(nb_path), f"Missing {nb_path}"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    executed_cells = [c for c in code_cells if len(c.get("outputs", [])) > 0]
    print(f"[OK] {nb_path} has {len(code_cells)} code cells, {len(executed_cells)} executed with outputs")
    assert len(executed_cells) >= 6, "Expected at least 6 executed code cells with outputs"

    # 4. Check mirrored files
    mirrored_files = [
        "chi_square_project/results/09_model_comparison.csv",
        "chi_square_project/figures/09_model_performance_comparison.png",
        "chi_square_project/figures/09_encoded_feature_count_comparison.png",
        "chi_square_project/notebooks/09_model_comparison.ipynb"
    ]
    for mf in mirrored_files:
        assert os.path.exists(mf), f"Missing mirrored file {mf}"
        print(f"[OK] Mirrored file exists: {mf}")

    print("\nALL PHASE 9 VERIFICATIONS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    verify_phase9()
