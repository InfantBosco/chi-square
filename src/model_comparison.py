"""
Model Comparison Module (Phase 9)
Compares Logistic Regression performance using:
- Model A: Original high-cardinality categorical features
- Model B: Grouped high-cardinality categorical features
"""

import os
import time
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
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def run_model_comparison():
    print("=" * 70)
    print("PHASE 9: MODEL COMPARISON (ORIGINAL VS. TRANSFORMED)")
    print("=" * 70)

    # 1. Load data
    print("\n1. Loading datasets and artifacts...")
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    X_train_grouped = pd.read_csv("data/processed/X_train_grouped.csv")
    X_test_grouped = pd.read_csv("data/processed/X_test_grouped.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    with open("results/04_selected_high_cardinality_feature.txt", "r") as f:
        high_card_feat = f.read().strip()
    
    print(f"Selected high-cardinality feature: {high_card_feat}")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Identify categorical columns
    categorical_cols = ["COW", "SCHL", "MAR", high_card_feat, "POBP", "RELP", "SEX", "RAC1P"]
    categorical_cols_grouped = [
        c if c != high_card_feat else f"{high_card_feat}_grouped" for c in categorical_cols
    ]

    print(f"Model A categorical features: {categorical_cols}")
    print(f"Model B categorical features: {categorical_cols_grouped}")

    # Ensure categorical features are string-typed for consistent OHE
    X_tr_cat_A = X_train[categorical_cols].astype(str)
    X_te_cat_A = X_test[categorical_cols].astype(str)

    X_tr_cat_B = X_train_grouped[categorical_cols_grouped].astype(str)
    X_te_cat_B = X_test_grouped[categorical_cols_grouped].astype(str)

    # 2. Model A — Original representation
    print("\n2. Training Model A (Original Representation)...")
    ohe_A = OneHotEncoder(handle_unknown="ignore")
    t0 = time.time()
    X_tr_enc_A = ohe_A.fit_transform(X_tr_cat_A)
    clf_A = LogisticRegression(max_iter=1000, random_state=42)
    clf_A.fit(X_tr_enc_A, y_train)
    train_time_A = time.time() - t0

    t0 = time.time()
    X_te_enc_A = ohe_A.transform(X_te_cat_A)
    preds_A = clf_A.predict(X_te_enc_A)
    probs_A = clf_A.predict_proba(X_te_enc_A)[:, 1]
    pred_time_A = time.time() - t0

    n_features_A = X_tr_enc_A.shape[1]
    acc_A = accuracy_score(y_test, preds_A)
    prec_A = precision_score(y_test, preds_A)
    rec_A = recall_score(y_test, preds_A)
    f1_A = f1_score(y_test, preds_A)
    auc_A = roc_auc_score(y_test, probs_A)

    print(f"  Model A Encoded Features: {n_features_A}")
    print(f"  Model A Accuracy:  {acc_A:.4f}")
    print(f"  Model A Precision: {prec_A:.4f}")
    print(f"  Model A Recall:    {rec_A:.4f}")
    print(f"  Model A F1-score:  {f1_A:.4f}")
    print(f"  Model A ROC-AUC:   {auc_A:.4f}")
    print(f"  Model A Train Time: {train_time_A:.4f}s | Predict Time: {pred_time_A:.4f}s")

    # 3. Model B — Transformed representation
    print("\n3. Training Model B (Transformed / Grouped Representation)...")
    ohe_B = OneHotEncoder(handle_unknown="ignore")
    t0 = time.time()
    X_tr_enc_B = ohe_B.fit_transform(X_tr_cat_B)
    clf_B = LogisticRegression(max_iter=1000, random_state=42)
    clf_B.fit(X_tr_enc_B, y_train)
    train_time_B = time.time() - t0

    t0 = time.time()
    X_te_enc_B = ohe_B.transform(X_te_cat_B)
    preds_B = clf_B.predict(X_te_enc_B)
    probs_B = clf_B.predict_proba(X_te_enc_B)[:, 1]
    pred_time_B = time.time() - t0

    n_features_B = X_tr_enc_B.shape[1]
    acc_B = accuracy_score(y_test, preds_B)
    prec_B = precision_score(y_test, preds_B)
    rec_B = recall_score(y_test, preds_B)
    f1_B = f1_score(y_test, preds_B)
    auc_B = roc_auc_score(y_test, probs_B)

    print(f"  Model B Encoded Features: {n_features_B}")
    print(f"  Model B Accuracy:  {acc_B:.4f}")
    print(f"  Model B Precision: {prec_B:.4f}")
    print(f"  Model B Recall:    {rec_B:.4f}")
    print(f"  Model B F1-score:  {f1_B:.4f}")
    print(f"  Model B ROC-AUC:   {auc_B:.4f}")
    print(f"  Model B Train Time: {train_time_B:.4f}s | Predict Time: {pred_time_B:.4f}s")

    # 4. Construct Comparison Summary Table
    print("\n4. Building Comparison Summary DataFrame...")
    comparison_df = pd.DataFrame([
        {
            "Model name": "Model A (Original)",
            "Number of features": n_features_A,
            "Accuracy": round(acc_A, 4),
            "Precision": round(prec_A, 4),
            "Recall": round(rec_A, 4),
            "F1-score": round(f1_A, 4),
            "ROC-AUC": round(auc_A, 4),
            "Training time (seconds)": round(train_time_A, 4),
            "Prediction time (seconds)": round(pred_time_A, 4),
        },
        {
            "Model name": "Model B (Transformed)",
            "Number of features": n_features_B,
            "Accuracy": round(acc_B, 4),
            "Precision": round(prec_B, 4),
            "Recall": round(rec_B, 4),
            "F1-score": round(f1_B, 4),
            "ROC-AUC": round(auc_B, 4),
            "Training time (seconds)": round(train_time_B, 4),
            "Prediction time (seconds)": round(pred_time_B, 4),
        }
    ])

    print("\n" + comparison_df.to_string(index=False))

    # Save to results/09_model_comparison.csv
    results_path = "results/09_model_comparison.csv"
    os.makedirs("results", exist_ok=True)
    comparison_df.to_csv(results_path, index=False)
    print(f"\nSaved comparison summary to {results_path}")

    # Mirror to chi_square_project
    if os.path.exists("chi_square_project"):
        os.makedirs("chi_square_project/results", exist_ok=True)
        shutil.copy2(results_path, "chi_square_project/results/09_model_comparison.csv")
        print("Mirrored comparison summary to chi_square_project/results/09_model_comparison.csv")

    # 5. Visualizations
    print("\n5. Generating Visualizations...")
    os.makedirs("figures", exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Figure 1: Grouped bar chart comparing performance metrics
    metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    val_A = [acc_A, prec_A, rec_A, f1_A, auc_A]
    val_B = [acc_B, prec_B, rec_B, f1_B, auc_B]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    bars_A = ax.bar(x - width/2, val_A, width, label=f"Model A (Original, {n_features_A} features)", color="#1f77b4", edgecolor="black", alpha=0.85)
    bars_B = ax.bar(x + width/2, val_B, width, label=f"Model B (Transformed, {n_features_B} features)", color="#2ca02c", edgecolor="black", alpha=0.85)

    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Out-of-Sample Performance Comparison: Model A vs. Model B", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11, fontweight="bold")
    ax.set_ylim(0.65, 0.95)
    ax.legend(loc="lower right", frameon=True, fontsize=11)

    # Add value labels
    for bar in bars_A:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1f77b4")

    for bar in bars_B:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold", color="#2ca02c")

    plt.tight_layout()
    fig1_path = "figures/09_model_performance_comparison.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved performance comparison figure to {fig1_path}")

    # Figure 2: Bar chart comparing the number of encoded features
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    model_labels = ["Model A\n(Original OCCP)", "Model B\n(OCCP Grouped)"]
    feat_counts = [n_features_A, n_features_B]
    colors = ["#1f77b4", "#2ca02c"]

    bars = ax.bar(model_labels, feat_counts, color=colors, width=0.5, edgecolor="black", alpha=0.85)
    ax.set_ylabel("Number of Encoded Features", fontsize=12, fontweight="bold")
    ax.set_title("One-Hot Encoded Dimensionality Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, 1000)

    # Reduction calculation
    reduction_count = n_features_A - n_features_B
    reduction_pct = (reduction_count / n_features_A) * 100

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:,} features",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Annotate reduction
    ax.annotate(
        f"Dimensionality Reduction:\n-270 features (-{reduction_pct:.1f}%)\nZero sparse categories < 100 obs",
        xy=(0.5, 650),
        xycoords="data",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#fff2cc", edgecolor="#d6b656", lw=1.5),
        fontsize=10, fontweight="bold"
    )

    plt.tight_layout()
    fig2_path = "figures/09_encoded_feature_count_comparison.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved feature count comparison figure to {fig2_path}")

    # Mirror figures to chi_square_project
    if os.path.exists("chi_square_project"):
        os.makedirs("chi_square_project/figures", exist_ok=True)
        shutil.copy2(fig1_path, "chi_square_project/figures/09_model_performance_comparison.png")
        shutil.copy2(fig2_path, "chi_square_project/figures/09_encoded_feature_count_comparison.png")
        print("Mirrored figures to chi_square_project/figures/")

    print("\nPhase 9 model comparison complete!")
    return comparison_df

if __name__ == "__main__":
    run_model_comparison()
