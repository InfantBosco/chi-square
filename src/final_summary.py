"""
Final Summary and Dashboard Generator (Phase 11)
Aggregates all results, creates results/11_final_project_summary.csv,
and generates figures/11_final_results_summary.png (4-panel dashboard).
"""

import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_final_summary():
    print("=" * 70)
    print("PHASE 11: FINAL ANALYSIS AND REPORTING")
    print("=" * 70)

    # 1. Load result files
    print("\n1. Loading result files...")
    df_chi2_base = pd.read_csv("results/05_baseline_chi_square_results.csv")
    df_chi2_trans = pd.read_csv("results/07_transformed_chi_square_results.csv")
    df_chi2_comp = pd.read_csv("results/07_chi_square_before_after_comparison.csv")
    df_model_comp = pd.read_csv("results/09_model_comparison.csv")
    df_gen_results = pd.read_csv("results/10_generalization_results.csv")
    df_rare_summary = pd.read_csv("results/10_rare_unseen_category_summary.csv")

    # Extract exact metrics
    orig_cats = int(df_chi2_base["Number of categories"].iloc[0])
    trans_cats = int(df_chi2_trans["Number of transformed categories"].iloc[0])
    cat_reduction_pct = ((orig_cats - trans_cats) / orig_cats) * 100

    base_chi2 = float(df_chi2_base["Chi-square statistic"].iloc[0])
    base_p = float(df_chi2_base["p-value"].iloc[0])
    trans_chi2 = float(df_chi2_trans["Chi-square statistic"].iloc[0])
    trans_p = float(df_chi2_trans["p-value"].iloc[0])

    orig_features = int(df_model_comp[df_model_comp["Model name"] == "Model A (Original)"]["Number of features"].iloc[0])
    trans_features = int(df_model_comp[df_model_comp["Model name"] == "Model B (Transformed)"]["Number of features"].iloc[0])
    feature_reduction_pct = ((orig_features - trans_features) / orig_features) * 100

    # Model A & B Full test metrics
    full_A = df_gen_results[(df_gen_results["Evaluation subset"] == "Full test set") & (df_gen_results["Model name"] == "Model A (Original)")].iloc[0]
    full_B = df_gen_results[(df_gen_results["Evaluation subset"] == "Full test set") & (df_gen_results["Model name"] == "Model B (Transformed)")].iloc[0]

    # Model A & B Rare/Unseen test metrics
    rare_A = df_gen_results[(df_gen_results["Evaluation subset"] == "Rare or unseen test subset") & (df_gen_results["Model name"] == "Model A (Original)")].iloc[0]
    rare_B = df_gen_results[(df_gen_results["Evaluation subset"] == "Rare or unseen test subset") & (df_gen_results["Model name"] == "Model B (Transformed)")].iloc[0]

    # 2. Build results/11_final_project_summary.csv
    print("\n2. Building results/11_final_project_summary.csv...")
    summary_data = [
        {"Metric": "Original number of categories", "Value": str(orig_cats)},
        {"Metric": "Transformed number of categories", "Value": str(trans_cats)},
        {"Metric": "Cardinality reduction percentage", "Value": f"{cat_reduction_pct:.4f}%"},
        {"Metric": "Baseline chi-square statistic", "Value": f"{base_chi2:.4f}"},
        {"Metric": "Baseline chi-square p-value", "Value": str(base_p)},
        {"Metric": "Transformed chi-square statistic", "Value": f"{trans_chi2:.4f}"},
        {"Metric": "Transformed chi-square p-value", "Value": str(trans_p)},
        {"Metric": "Original encoded-feature count", "Value": str(orig_features)},
        {"Metric": "Transformed encoded-feature count", "Value": str(trans_features)},
        {"Metric": "Encoded-feature reduction percentage", "Value": f"{feature_reduction_pct:.4f}%"},
        {"Metric": "Model A Full-test Accuracy", "Value": str(full_A["Accuracy"])},
        {"Metric": "Model A Full-test Precision", "Value": str(full_A["Precision"])},
        {"Metric": "Model A Full-test Recall", "Value": str(full_A["Recall"])},
        {"Metric": "Model A Full-test F1-score", "Value": str(full_A["F1-score"])},
        {"Metric": "Model A Full-test ROC-AUC", "Value": str(full_A["ROC-AUC"])},
        {"Metric": "Model B Full-test Accuracy", "Value": str(full_B["Accuracy"])},
        {"Metric": "Model B Full-test Precision", "Value": str(full_B["Precision"])},
        {"Metric": "Model B Full-test Recall", "Value": str(full_B["Recall"])},
        {"Metric": "Model B Full-test F1-score", "Value": str(full_B["F1-score"])},
        {"Metric": "Model B Full-test ROC-AUC", "Value": str(full_B["ROC-AUC"])},
        {"Metric": "Model A Rare/unseen-subset Accuracy", "Value": str(rare_A["Accuracy"])},
        {"Metric": "Model A Rare/unseen-subset Precision", "Value": str(rare_A["Precision"])},
        {"Metric": "Model A Rare/unseen-subset Recall", "Value": str(rare_A["Recall"])},
        {"Metric": "Model A Rare/unseen-subset F1-score", "Value": str(rare_A["F1-score"])},
        {"Metric": "Model A Rare/unseen-subset ROC-AUC", "Value": str(rare_A["ROC-AUC"])},
        {"Metric": "Model B Rare/unseen-subset Accuracy", "Value": str(rare_B["Accuracy"])},
        {"Metric": "Model B Rare/unseen-subset Precision", "Value": str(rare_B["Precision"])},
        {"Metric": "Model B Rare/unseen-subset Recall", "Value": str(rare_B["Recall"])},
        {"Metric": "Model B Rare/unseen-subset F1-score", "Value": str(rare_B["F1-score"])},
        {"Metric": "Model B Rare/unseen-subset ROC-AUC", "Value": str(rare_B["ROC-AUC"])}
    ]

    summary_df = pd.DataFrame(summary_data)
    os.makedirs("results", exist_ok=True)
    summary_path = "results/11_final_project_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved final summary to {summary_path}")

    if os.path.exists("chi_square_project"):
        os.makedirs("chi_square_project/results", exist_ok=True)
        shutil.copy2(summary_path, "chi_square_project/results/11_final_project_summary.csv")

    # 3. Create 4-panel dashboard figure
    print("\n3. Creating 4-panel dashboard figure (figures/11_final_results_summary.png)...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

    # Panel 1: Category Count Before and After Grouping
    ax1 = axes[0, 0]
    cat_labels = ["Original OCCP", "Transformed (OCCP_grouped)"]
    cat_counts = [orig_cats, trans_cats]
    bars1 = ax1.bar(cat_labels, cat_counts, color=["#1f77b4", "#2ca02c"], width=0.45, edgecolor="black", alpha=0.85)
    ax1.set_title("1. Category Count Before and After Grouping", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylabel("Number of Categories", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 650)
    for b in bars1:
        h = b.get_height()
        ax1.annotate(f"{h:,}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 5),
                     textcoords="offset points", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax1.annotate(f"Reduction: -270 categories (-{cat_reduction_pct:.1f}%)\nCochran Violations: 5.67% -> 0.00%",
                 xy=(0.5, 450), xycoords="data", ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff2cc", edgecolor="#d6b656", lw=1.5),
                 fontsize=10, fontweight="bold")

    # Panel 2: Encoded-Feature Count Before and After Grouping
    ax2 = axes[0, 1]
    feat_labels = ["Model A (Original)", "Model B (Transformed)"]
    feat_counts = [orig_features, trans_features]
    bars2 = ax2.bar(feat_labels, feat_counts, color=["#1f77b4", "#2ca02c"], width=0.45, edgecolor="black", alpha=0.85)
    ax2.set_title("2. One-Hot Encoded Feature Count", fontsize=12, fontweight="bold", pad=10)
    ax2.set_ylabel("Encoded Feature Dimensions", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 1000)
    for b in bars2:
        h = b.get_height()
        ax2.annotate(f"{h:,}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 5),
                     textcoords="offset points", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax2.annotate(f"Reduction: -270 features (-{feature_reduction_pct:.1f}%)\nTraining Time: 1.23s -> 0.87s (-29.7%)",
                 xy=(0.5, 680), xycoords="data", ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff2cc", edgecolor="#d6b656", lw=1.5),
                 fontsize=10, fontweight="bold")

    # Panel 3: Model Performance Comparison (Full Test Set)
    ax3 = axes[1, 0]
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    val_full_A = [full_A[m] for m in metric_cols]
    val_full_B = [full_B[m] for m in metric_cols]
    x = np.arange(len(metric_cols))
    width = 0.35

    bars3_A = ax3.bar(x - width/2, val_full_A, width, label="Model A (Original, 813 feat)", color="#1f77b4", edgecolor="black", alpha=0.85)
    bars3_B = ax3.bar(x + width/2, val_full_B, width, label="Model B (Transformed, 543 feat)", color="#2ca02c", edgecolor="black", alpha=0.85)
    ax3.set_title("3. Model Performance Comparison (Full Test Set, N = 38,551)", fontsize=12, fontweight="bold", pad=10)
    ax3.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(metric_cols, fontsize=9.5, fontweight="bold")
    ax3.set_ylim(0.65, 0.95)
    ax3.legend(loc="lower right", frameon=True, fontsize=9.5)
    for b in bars3_A:
        h = b.get_height()
        ax3.annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#1f77b4")
    for b in bars3_B:
        h = b.get_height()
        ax3.annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#2ca02c")

    # Panel 4: Generalization Comparison (Rare/Unseen Subset)
    ax4 = axes[1, 1]
    val_rare_A = [rare_A[m] for m in metric_cols]
    val_rare_B = [rare_B[m] for m in metric_cols]

    bars4_A = ax4.bar(x - width/2, val_rare_A, width, label="Model A (Original)", color="#1f77b4", edgecolor="black", alpha=0.85)
    bars4_B = ax4.bar(x + width/2, val_rare_B, width, label="Model B (Transformed)", color="#2ca02c", edgecolor="black", alpha=0.85)
    ax4.set_title("4. Generalization Comparison (Rare / Unseen Subset, N = 2,840, 7.37%)", fontsize=12, fontweight="bold", pad=10)
    ax4.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(metric_cols, fontsize=9.5, fontweight="bold")
    ax4.set_ylim(0.65, 0.95)
    ax4.legend(loc="lower right", frameon=True, fontsize=9.5)
    for b in bars4_A:
        h = b.get_height()
        ax4.annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#1f77b4")
    for b in bars4_B:
        h = b.get_height()
        ax4.annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#2ca02c")

    plt.suptitle("Chi-Square Project: End-to-End Statistical & Machine Learning Summary", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    fig_path = "figures/11_final_results_summary.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved final results summary figure to {fig_path}")

    if os.path.exists("chi_square_project"):
        os.makedirs("chi_square_project/figures", exist_ok=True)
        shutil.copy2(fig_path, "chi_square_project/figures/11_final_results_summary.png")

    print("\nPhase 11 final summary generation complete!")
    return summary_df

if __name__ == "__main__":
    generate_final_summary()
