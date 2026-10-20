"""
Verification script for Phase 11 deliverables
"""

import os
import json
import pandas as pd

def verify_phase11():
    print("=" * 60)
    print("VERIFYING PHASE 11 DELIVERABLES")
    print("=" * 60)

    # 1. Check summary CSV
    summary_path = "results/11_final_project_summary.csv"
    assert os.path.exists(summary_path), f"Missing {summary_path}"
    df_summary = pd.read_csv(summary_path)
    assert len(df_summary) >= 25, f"Expected >=25 metrics, got {len(df_summary)}"
    print(f"[OK] {summary_path} exists with {len(df_summary)} metrics:")
    print(df_summary.to_string(index=False))

    # Required metric checks
    req_metrics = [
        "Original number of categories",
        "Transformed number of categories",
        "Cardinality reduction percentage",
        "Baseline chi-square statistic",
        "Baseline chi-square p-value",
        "Transformed chi-square statistic",
        "Transformed chi-square p-value",
        "Original encoded-feature count",
        "Transformed encoded-feature count",
        "Model A Full-test Accuracy",
        "Model B Full-test Accuracy",
        "Model A Rare/unseen-subset Accuracy",
        "Model B Rare/unseen-subset Accuracy"
    ]
    existing_metrics = set(df_summary["Metric"])
    for m in req_metrics:
        assert m in existing_metrics, f"Missing metric: {m}"
    print(f"\n[OK] All key required metrics verified in {summary_path}!")

    # 2. Check figure
    fig_path = "figures/11_final_results_summary.png"
    assert os.path.exists(fig_path) and os.path.getsize(fig_path) > 1000, f"Missing or invalid {fig_path}"
    print(f"[OK] {fig_path} exists ({os.path.getsize(fig_path):,} bytes)")

    # 3. Check outline
    outline_path = "report/final_report_outline.md"
    assert os.path.exists(outline_path), f"Missing {outline_path}"
    with open(outline_path, "r", encoding="utf-8") as f:
        outline_content = f.read()
    
    sections = [
        "1. Introduction",
        "2. Problem statement",
        "3. Literature review",
        "4. Dataset description",
        "5. Methodology",
        "6. Mathematical background",
        "7. High-cardinality analysis",
        "8. Baseline chi-square analysis",
        "9. Cardinality-reduction strategy",
        "10. Transformed chi-square analysis",
        "11. Machine-learning evaluation",
        "12. Generalization testing",
        "13. Results and discussion",
        "14. Limitations",
        "15. Conclusion",
        "16. References"
    ]
    for s in sections:
        assert s.lower() in outline_content.lower(), f"Missing section '{s}' in {outline_path}"
    print(f"[OK] {outline_path} contains all 16 required sections!")

    # 4. Check notebook
    nb_path = "notebooks/11_final_analysis_and_report.ipynb"
    assert os.path.exists(nb_path), f"Missing {nb_path}"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    executed_cells = [c for c in code_cells if len(c.get("outputs", [])) > 0]
    print(f"[OK] {nb_path} has {len(code_cells)} code cells, {len(executed_cells)} executed with outputs")
    assert len(executed_cells) >= 3, "Expected at least 3 executed code cells"

    # 5. Check mirrored files
    mirrored_files = [
        "chi_square_project/results/11_final_project_summary.csv",
        "chi_square_project/figures/11_final_results_summary.png",
        "chi_square_project/report/final_report_outline.md",
        "chi_square_project/notebooks/11_final_analysis_and_report.ipynb"
    ]
    for mf in mirrored_files:
        assert os.path.exists(mf), f"Missing mirrored file {mf}"
        print(f"[OK] Mirrored file exists: {mf}")

    print("\nALL PHASE 11 VERIFICATIONS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    verify_phase11()
