"""
Verification script for Phase 10 deliverables
"""

import os
import json
import pandas as pd

def verify_phase10():
    print("=" * 60)
    print("VERIFYING PHASE 10 DELIVERABLES")
    print("=" * 60)

    # 1. Check generalization results CSV
    results_path = "results/10_generalization_results.csv"
    assert os.path.exists(results_path), f"Missing {results_path}"
    df_results = pd.read_csv(results_path)
    expected_cols = ["Evaluation subset", "Model name", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    for col in expected_cols:
        assert col in df_results.columns, f"Missing column {col} in {results_path}"
    assert len(df_results) == 4, f"Expected 4 rows, got {len(df_results)}"
    print(f"[OK] {results_path} contains 4 rows and all expected columns:")
    print(df_results.to_string(index=False))

    # 2. Check rare/unseen summary CSV
    summary_path = "results/10_rare_unseen_category_summary.csv"
    assert os.path.exists(summary_path), f"Missing {summary_path}"
    df_summary = pd.read_csv(summary_path)
    expected_summary_cols = [
        "Number of rare test categories",
        "Number of unseen test categories",
        "Number of test rows in rare/unseen subset",
        "Percentage of test rows in rare/unseen subset"
    ]
    for col in expected_summary_cols:
        assert col in df_summary.columns, f"Missing column {col} in {summary_path}"
    print(f"\n[OK] {summary_path} contains required summary metrics:")
    print(df_summary.to_string(index=False))

    # 3. Check figure
    fig_path = "figures/10_generalization_comparison.png"
    assert os.path.exists(fig_path) and os.path.getsize(fig_path) > 1000, f"Invalid or missing {fig_path}"
    print(f"\n[OK] {fig_path} exists ({os.path.getsize(fig_path):,} bytes)")

    # 4. Check notebook
    nb_path = "notebooks/10_generalization_analysis.ipynb"
    assert os.path.exists(nb_path), f"Missing {nb_path}"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    executed_cells = [c for c in code_cells if len(c.get("outputs", [])) > 0]
    print(f"[OK] {nb_path} has {len(code_cells)} code cells, {len(executed_cells)} executed with outputs")
    assert len(executed_cells) >= 5, "Expected at least 5 executed code cells with outputs"

    # 5. Check mirrored files
    mirrored_files = [
        "chi_square_project/results/10_generalization_results.csv",
        "chi_square_project/results/10_rare_unseen_category_summary.csv",
        "chi_square_project/figures/10_generalization_comparison.png",
        "chi_square_project/notebooks/10_generalization_analysis.ipynb"
    ]
    for mf in mirrored_files:
        assert os.path.exists(mf), f"Missing mirrored file {mf}"
        print(f"[OK] Mirrored file exists: {mf}")

    print("\nALL PHASE 10 VERIFICATIONS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    verify_phase10()
