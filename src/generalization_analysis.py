"""
Generalization Analysis Module (Phase 10)
Analyzes Model A vs. Model B performance on:
- Full test set
- Rare or unseen test subset
"""

import os
import json
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

def run_generalization_analysis():
    print("=" * 70)
    print("PHASE 10: GENERALIZATION ANALYSIS (FULL VS. RARE/UNSEEN SUBSET)")
    print("=" * 70)

    # 1. Load data
    print("\n1. Loading datasets and metadata...")
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    X_train_grouped = pd.read_csv("data/processed/X_train_grouped.csv")
    X_test_grouped = pd.read_csv("data/processed/X_test_grouped.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    with open("results/04_selected_high_cardinality_feature.txt", "r") as f:
        high_card_feat = f.read().strip()

    with open("results/06_category_mapping.json", "r") as f:
        mapping = json.load(f)

    # 2. Rebuild and train Model A and Model B
    print("\n2. Rebuilding and training Model A and Model B on training set...")
    cat_cols_A = ["COW", "SCHL", "MAR", high_card_feat, "POBP", "RELP", "SEX", "RAC1P"]
    cat_cols_B = [c if c != high_card_feat else f"{high_card_feat}_grouped" for c in cat_cols_A]

    # Model A
    ohe_A = OneHotEncoder(handle_unknown="ignore")
    X_tr_enc_A = ohe_A.fit_transform(X_train[cat_cols_A].astype(str))
    clf_A = LogisticRegression(max_iter=1000, random_state=42)
    clf_A.fit(X_tr_enc_A, y_train)

    # Model B
    ohe_B = OneHotEncoder(handle_unknown="ignore")
    X_tr_enc_B = ohe_B.fit_transform(X_train_grouped[cat_cols_B].astype(str))
    clf_B = LogisticRegression(max_iter=1000, random_state=42)
    clf_B.fit(X_tr_enc_B, y_train)

    print("Model A and Model B trained successfully.")

    # 3. Identify rare or unseen test rows
    print("\n3. Identifying rare or unseen test categories and rows...")
    train_counts = X_train[high_card_feat].astype(str).value_counts()
    train_cats = set(train_counts.index)
    test_cats = set(X_test[high_card_feat].astype(str).unique())

    rare_train_cats = set(train_counts[train_counts < 100].index)
    unseen_test_cats = test_cats - train_cats
    rare_test_cats = (test_cats & train_cats) & rare_train_cats

    n_rare_cats = len(rare_test_cats)
    n_unseen_cats = len(unseen_test_cats)

    # Mask for test rows where OCCP is rare (<100 in train) or unseen
    rare_or_unseen_mask = X_test[high_card_feat].astype(str).apply(
        lambda c: (c in rare_train_cats) or (c in unseen_test_cats)
    )

    n_test_rare_rows = int(rare_or_unseen_mask.sum())
    total_test_rows = len(X_test)
    pct_test_rare_rows = (n_test_rare_rows / total_test_rows) * 100

    print(f"  Number of rare test categories (< 100 train obs): {n_rare_cats}")
    print(f"  Number of unseen test categories:                {n_unseen_cats}")
    print(f"  Number of test rows in rare/unseen subset:       {n_test_rare_rows:,}")
    print(f"  Percentage of test rows in rare/unseen subset:   {pct_test_rare_rows:.4f}%")

    # Save summary to results/10_rare_unseen_category_summary.csv
    rare_unseen_summary_df = pd.DataFrame([{
        "Number of rare test categories": n_rare_cats,
        "Number of unseen test categories": n_unseen_cats,
        "Number of test rows in rare/unseen subset": n_test_rare_rows,
        "Percentage of test rows in rare/unseen subset": round(pct_test_rare_rows, 4)
    }])
    summary_path = "results/10_rare_unseen_category_summary.csv"
    rare_unseen_summary_df.to_csv(summary_path, index=False)
    print(f"Saved rare/unseen category summary to {summary_path}")

    # Mirror summary
    if os.path.exists("chi_square_project"):
        os.makedirs("chi_square_project/results", exist_ok=True)
        shutil.copy2(summary_path, "chi_square_project/results/10_rare_unseen_category_summary.csv")

    # 4. Predict on Full Test Set and Rare/Unseen Subset
    print("\n4. Evaluating Model A and Model B on Full Test and Rare/Unseen Subsets...")
    X_te_enc_A = ohe_A.transform(X_test[cat_cols_A].astype(str))
    preds_full_A = clf_A.predict(X_te_enc_A)
    probs_full_A = clf_A.predict_proba(X_te_enc_A)[:, 1]

    X_te_enc_B = ohe_B.transform(X_test_grouped[cat_cols_B].astype(str))
    preds_full_B = clf_B.predict(X_te_enc_B)
    probs_full_B = clf_B.predict_proba(X_te_enc_B)[:, 1]

    # Subset predictions
    preds_rare_A = preds_full_A[rare_or_unseen_mask]
    probs_rare_A = probs_full_A[rare_or_unseen_mask]

    preds_rare_B = preds_full_B[rare_or_unseen_mask]
    probs_rare_B = probs_full_B[rare_or_unseen_mask]

    y_test_rare = y_test[rare_or_unseen_mask]

    def compute_metrics(y_true, preds, probs):
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds)
        rec = recall_score(y_true, preds)
        f1 = f1_score(y_true, preds)
        auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else np.nan
        return {
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-score": round(f1, 4),
            "ROC-AUC": round(auc, 4)
        }

    metrics_full_A = compute_metrics(y_test, preds_full_A, probs_full_A)
    metrics_full_B = compute_metrics(y_test, preds_full_B, probs_full_B)
    metrics_rare_A = compute_metrics(y_test_rare, preds_rare_A, probs_rare_A)
    metrics_rare_B = compute_metrics(y_test_rare, preds_rare_B, probs_rare_B)

    # 5. Build and save results/10_generalization_results.csv
    print("\n5. Constructing Generalization Results DataFrame...")
    results_df = pd.DataFrame([
        {"Evaluation subset": "Full test set", "Model name": "Model A (Original)", **metrics_full_A},
        {"Evaluation subset": "Full test set", "Model name": "Model B (Transformed)", **metrics_full_B},
        {"Evaluation subset": "Rare or unseen test subset", "Model name": "Model A (Original)", **metrics_rare_A},
        {"Evaluation subset": "Rare or unseen test subset", "Model name": "Model B (Transformed)", **metrics_rare_B},
    ])

    print("\n" + results_df.to_string(index=False))

    results_path = "results/10_generalization_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved generalization results to {results_path}")

    if os.path.exists("chi_square_project"):
        shutil.copy2(results_path, "chi_square_project/results/10_generalization_results.csv")

    # 6. Visualization
    print("\n6. Generating Generalization Comparison Chart...")
    metrics_names = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    val_full_A = [metrics_full_A[m] for m in metrics_names]
    val_full_B = [metrics_full_B[m] for m in metrics_names]
    val_rare_A = [metrics_rare_A[m] for m in metrics_names]
    val_rare_B = [metrics_rare_B[m] for m in metrics_names]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300, sharey=True)
    sns.set_theme(style="whitegrid")

    x = np.arange(len(metrics_names))
    width = 0.35

    # Subplot 1: Full Test Set
    ax1 = axes[0]
    b1_A = ax1.bar(x - width/2, val_full_A, width, label="Model A (Original, 813 feat)", color="#1f77b4", edgecolor="black", alpha=0.85)
    b1_B = ax1.bar(x + width/2, val_full_B, width, label="Model B (Transformed, 543 feat)", color="#2ca02c", edgecolor="black", alpha=0.85)
    ax1.set_title(f"Full Test Set (N = {total_test_rows:,})", fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names, fontsize=10, fontweight="bold")
    ax1.set_ylim(0.65, 0.92)
    ax1.legend(loc="lower right", frameon=True, fontsize=10)

    for b in b1_A:
        h = b.get_height()
        ax1.annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1f77b4")
    for b in b1_B:
        h = b.get_height()
        ax1.annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#2ca02c")

    # Subplot 2: Rare/Unseen Test Subset
    ax2 = axes[1]
    b2_A = ax2.bar(x - width/2, val_rare_A, width, label="Model A (Original)", color="#1f77b4", edgecolor="black", alpha=0.85)
    b2_B = ax2.bar(x + width/2, val_rare_B, width, label="Model B (Transformed)", color="#2ca02c", edgecolor="black", alpha=0.85)
    ax2.set_title(f"Rare / Unseen Category Subset (N = {n_test_rare_rows:,}, {pct_test_rare_rows:.1f}%)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_names, fontsize=10, fontweight="bold")
    ax2.legend(loc="lower right", frameon=True, fontsize=10)

    for b in b2_A:
        h = b.get_height()
        ax2.annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1f77b4")
    for b in b2_B:
        h = b.get_height()
        ax2.annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#2ca02c")

    plt.suptitle("Generalization Performance Comparison: Full Test vs. Rare/Unseen Subset", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()

    fig_path = "figures/10_generalization_comparison.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved generalization comparison chart to {fig_path}")

    if os.path.exists("chi_square_project"):
        os.makedirs("chi_square_project/figures", exist_ok=True)
        shutil.copy2(fig_path, "chi_square_project/figures/10_generalization_comparison.png")

    print("\nPhase 10 generalization analysis complete!")
    return results_df, rare_unseen_summary_df

if __name__ == "__main__":
    run_generalization_analysis()
