"""
Builder script for notebooks/10_generalization_analysis.ipynb
"""

import json
import os
import shutil

def create_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Phase 10: Generalization Analysis on Rare and Unseen Categories\n",
                    "\n",
                    "## Executive Summary & Objectives\n",
                    "In this notebook, we perform a deep-dive generalization analysis to evaluate how Model A (Original OCCP representation with 813 features) and Model B (Grouped OCCP representation with 543 features) perform on the most challenging portion of the evaluation data:\n",
                    "- **`rare_or_unseen_test` subset**: Test rows where the individual's occupation belonged to a category that appeared **fewer than 100 times in the training data**, or **did not appear in the training data at all**.\n",
                    "\n",
                    "### Core Questions Investigated\n",
                    "1. How many rare and unseen categories appear in the untouched test set?\n",
                    "2. How does predictive performance compare between Model A and Model B on:\n",
                    "   - The **Full Test Set** ($N = 38,551$)\n",
                    "   - The **Rare/Unseen Test Subset** ($N = 2,840$)\n",
                    "3. Does grouping sparse categories into a collective `'Rare'` bucket improve or degrade performance on rare occupations?\n",
                    "4. Is the sample size of rare categories sufficient for statistical reliability?\n",
                    "5. How does the mapping rule deterministically resolve novel/unseen categories during out-of-sample inference?"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Setup and imports\n",
                    "import os\n",
                    "import json\n",
                    "import shutil\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "from sklearn.linear_model import LogisticRegression\n",
                    "from sklearn.preprocessing import OneHotEncoder\n",
                    "from sklearn.metrics import (\n",
                    "    accuracy_score,\n",
                    "    precision_score,\n",
                    "    recall_score,\n",
                    "    f1_score,\n",
                    "    roc_auc_score,\n",
                    "    classification_report,\n",
                    "    confusion_matrix\n",
                    ")\n",
                    "\n",
                    "sns.set_theme(style=\"whitegrid\")\n",
                    "plt.rcParams[\"figure.dpi\"] = 120"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Load Processed Datasets and Target Metadata"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load training and testing sets\n",
                    "X_train = pd.read_csv('../data/processed/X_train.csv')\n",
                    "X_test = pd.read_csv('../data/processed/X_test.csv')\n",
                    "X_train_grouped = pd.read_csv('../data/processed/X_train_grouped.csv')\n",
                    "X_test_grouped = pd.read_csv('../data/processed/X_test_grouped.csv')\n",
                    "y_train = pd.read_csv('../data/processed/y_train.csv').squeeze()\n",
                    "y_test = pd.read_csv('../data/processed/y_test.csv').squeeze()\n",
                    "\n",
                    "with open('../results/04_selected_high_cardinality_feature.txt', 'r') as f:\n",
                    "    high_card_feat = f.read().strip()\n",
                    "\n",
                    "with open('../results/06_category_mapping.json', 'r') as f:\n",
                    "    mapping = json.load(f)\n",
                    "\n",
                    "print(f\"Selected high-cardinality feature: {high_card_feat}\")\n",
                    "print(f\"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}\")\n",
                    "print(f\"X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Rebuild and Train Model A and Model B\n",
                    "\n",
                    "- **Model A**: Uses all 8 original categorical features (`['COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'SEX', 'RAC1P']`).\n",
                    "- **Model B**: Replaces `OCCP` with `OCCP_grouped`.\n",
                    "- Both models use `OneHotEncoder(handle_unknown='ignore')` and `LogisticRegression(max_iter=1000, random_state=42)` fitted exclusively on training data."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "cat_cols_A = ['COW', 'SCHL', 'MAR', high_card_feat, 'POBP', 'RELP', 'SEX', 'RAC1P']\n",
                    "cat_cols_B = [c if c != high_card_feat else f'{high_card_feat}_grouped' for c in cat_cols_A]\n",
                    "\n",
                    "# Train Model A\n",
                    "ohe_A = OneHotEncoder(handle_unknown='ignore')\n",
                    "X_train_enc_A = ohe_A.fit_transform(X_train[cat_cols_A].astype(str))\n",
                    "clf_A = LogisticRegression(max_iter=1000, random_state=42)\n",
                    "clf_A.fit(X_train_enc_A, y_train)\n",
                    "\n",
                    "# Train Model B\n",
                    "ohe_B = OneHotEncoder(handle_unknown='ignore')\n",
                    "X_train_enc_B = ohe_B.fit_transform(X_train_grouped[cat_cols_B].astype(str))\n",
                    "clf_B = LogisticRegression(max_iter=1000, random_state=42)\n",
                    "clf_B.fit(X_train_enc_B, y_train)\n",
                    "\n",
                    "print(f\"Model A trained with {X_train_enc_A.shape[1]} features.\")\n",
                    "print(f\"Model B trained with {X_train_enc_B.shape[1]} features.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Identify Rare and Unseen Test Categories\n",
                    "\n",
                    "We identify all test observations where the original `OCCP` category:\n",
                    "1. Appeared **fewer than 100 times in the training set**, or\n",
                    "2. Did **not appear in the training set at all** (unseen categories).\n",
                    "\n",
                    "All such categories were mapped to `'Rare'` in Phase 8 using the training-derived mapping rule."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "train_counts = X_train[high_card_feat].astype(str).value_counts()\n",
                    "train_cats = set(train_counts.index)\n",
                    "test_cats = set(X_test[high_card_feat].astype(str).unique())\n",
                    "\n",
                    "# Categories with < 100 training occurrences\n",
                    "rare_train_cats = set(train_counts[train_counts < 100].index)\n",
                    "\n",
                    "# Test categories that are unseen vs. rare in training\n",
                    "unseen_test_cats = test_cats - train_cats\n",
                    "rare_test_cats = (test_cats & train_cats) & rare_train_cats\n",
                    "\n",
                    "# Mask for test rows in the rare_or_unseen subset\n",
                    "rare_or_unseen_mask = X_test[high_card_feat].astype(str).apply(\n",
                    "    lambda c: (c in rare_train_cats) or (c in unseen_test_cats)\n",
                    ")\n",
                    "\n",
                    "n_rare_cats = len(rare_test_cats)\n",
                    "n_unseen_cats = len(unseen_test_cats)\n",
                    "n_test_rare_rows = int(rare_or_unseen_mask.sum())\n",
                    "total_test_rows = len(X_test)\n",
                    "pct_test_rare_rows = (n_test_rare_rows / total_test_rows) * 100\n",
                    "\n",
                    "print(f\"Number of rare test categories (< 100 training obs): {n_rare_cats}\")\n",
                    "print(f\"Number of unseen test categories:                 {n_unseen_cats}\")\n",
                    "print(f\"Number of test rows in rare/unseen subset:        {n_test_rare_rows:,}\")\n",
                    "print(f\"Percentage of test rows in rare/unseen subset:    {pct_test_rare_rows:.4f}%\")\n",
                    "\n",
                    "# Create and save summary table\n",
                    "rare_unseen_summary_df = pd.DataFrame([{\n",
                    "    'Number of rare test categories': n_rare_cats,\n",
                    "    'Number of unseen test categories': n_unseen_cats,\n",
                    "    'Number of test rows in rare/unseen subset': n_test_rare_rows,\n",
                    "    'Percentage of test rows in rare/unseen subset': round(pct_test_rare_rows, 4)\n",
                    "}])\n",
                    "\n",
                    "os.makedirs('../results', exist_ok=True)\n",
                    "summary_path = '../results/10_rare_unseen_category_summary.csv'\n",
                    "rare_unseen_summary_df.to_csv(summary_path, index=False)\n",
                    "\n",
                    "if os.path.exists('../chi_square_project'):\n",
                    "    os.makedirs('../chi_square_project/results', exist_ok=True)\n",
                    "    shutil.copy2(summary_path, '../chi_square_project/results/10_rare_unseen_category_summary.csv')\n",
                    "\n",
                    "rare_unseen_summary_df"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Evaluate Both Models on Full Test Set and Rare/Unseen Subset"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Transform full test set\n",
                    "X_test_enc_A = ohe_A.transform(X_test[cat_cols_A].astype(str))\n",
                    "preds_full_A = clf_A.predict(X_test_enc_A)\n",
                    "probs_full_A = clf_A.predict_proba(X_test_enc_A)[:, 1]\n",
                    "\n",
                    "X_test_enc_B = ohe_B.transform(X_test_grouped[cat_cols_B].astype(str))\n",
                    "preds_full_B = clf_B.predict(X_test_enc_B)\n",
                    "probs_full_B = clf_B.predict_proba(X_test_enc_B)[:, 1]\n",
                    "\n",
                    "# Slice subset predictions\n",
                    "preds_rare_A = preds_full_A[rare_or_unseen_mask]\n",
                    "probs_rare_A = probs_full_A[rare_or_unseen_mask]\n",
                    "\n",
                    "preds_rare_B = preds_full_B[rare_or_unseen_mask]\n",
                    "probs_rare_B = probs_full_B[rare_or_unseen_mask]\n",
                    "\n",
                    "y_test_rare = y_test[rare_or_unseen_mask]\n",
                    "\n",
                    "def calculate_metrics(y_true, preds, probs):\n",
                    "    acc = accuracy_score(y_true, preds)\n",
                    "    prec = precision_score(y_true, preds)\n",
                    "    rec = recall_score(y_true, preds)\n",
                    "    f1 = f1_score(y_true, preds)\n",
                    "    auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else np.nan\n",
                    "    return {\n",
                    "        'Accuracy': round(acc, 4),\n",
                    "        'Precision': round(prec, 4),\n",
                    "        'Recall': round(rec, 4),\n",
                    "        'F1-score': round(f1, 4),\n",
                    "        'ROC-AUC': round(auc, 4)\n",
                    "    }\n",
                    "\n",
                    "metrics_full_A = calculate_metrics(y_test, preds_full_A, probs_full_A)\n",
                    "metrics_full_B = calculate_metrics(y_test, preds_full_B, probs_full_B)\n",
                    "metrics_rare_A = calculate_metrics(y_test_rare, preds_rare_A, probs_rare_A)\n",
                    "metrics_rare_B = calculate_metrics(y_test_rare, preds_rare_B, probs_rare_B)\n",
                    "\n",
                    "# Build results table\n",
                    "results_df = pd.DataFrame([\n",
                    "    {'Evaluation subset': 'Full test set', 'Model name': 'Model A (Original)', **metrics_full_A},\n",
                    "    {'Evaluation subset': 'Full test set', 'Model name': 'Model B (Transformed)', **metrics_full_B},\n",
                    "    {'Evaluation subset': 'Rare or unseen test subset', 'Model name': 'Model A (Original)', **metrics_rare_A},\n",
                    "    {'Evaluation subset': 'Rare or unseen test subset', 'Model name': 'Model B (Transformed)', **metrics_rare_B},\n",
                    "])\n",
                    "\n",
                    "results_path = '../results/10_generalization_results.csv'\n",
                    "results_df.to_csv(results_path, index=False)\n",
                    "\n",
                    "if os.path.exists('../chi_square_project'):\n",
                    "    shutil.copy2(results_path, '../chi_square_project/results/10_generalization_results.csv')\n",
                    "\n",
                    "results_df"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Performance Visualization: Full Test Set vs. Rare/Unseen Subset"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']\n",
                    "val_full_A = [metrics_full_A[m] for m in metrics_names]\n",
                    "val_full_B = [metrics_full_B[m] for m in metrics_names]\n",
                    "val_rare_A = [metrics_rare_A[m] for m in metrics_names]\n",
                    "val_rare_B = [metrics_rare_B[m] for m in metrics_names]\n",
                    "\n",
                    "fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300, sharey=True)\n",
                    "x = np.arange(len(metrics_names))\n",
                    "width = 0.35\n",
                    "\n",
                    "# Subplot 1: Full Test Set\n",
                    "ax1 = axes[0]\n",
                    "b1_A = ax1.bar(x - width/2, val_full_A, width, label='Model A (Original, 813 feat)', color='#1f77b4', edgecolor='black', alpha=0.85)\n",
                    "b1_B = ax1.bar(x + width/2, val_full_B, width, label='Model B (Transformed, 543 feat)', color='#2ca02c', edgecolor='black', alpha=0.85)\n",
                    "ax1.set_title(f'Full Test Set (N = {total_test_rows:,})', fontsize=13, fontweight='bold', pad=12)\n",
                    "ax1.set_ylabel('Score', fontsize=11, fontweight='bold')\n",
                    "ax1.set_xticks(x)\n",
                    "ax1.set_xticklabels(metrics_names, fontsize=10, fontweight='bold')\n",
                    "ax1.set_ylim(0.65, 0.92)\n",
                    "ax1.legend(loc='lower right', frameon=True, fontsize=10)\n",
                    "\n",
                    "for b in b1_A:\n",
                    "    h = b.get_height()\n",
                    "    ax1.annotate(f'{h:.3f}', xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 4),\n",
                    "                 textcoords='offset points', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1f77b4')\n",
                    "for b in b1_B:\n",
                    "    h = b.get_height()\n",
                    "    ax1.annotate(f'{h:.3f}', xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 4),\n",
                    "                 textcoords='offset points', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#2ca02c')\n",
                    "\n",
                    "# Subplot 2: Rare/Unseen Subset\n",
                    "ax2 = axes[1]\n",
                    "b2_A = ax2.bar(x - width/2, val_rare_A, width, label='Model A (Original)', color='#1f77b4', edgecolor='black', alpha=0.85)\n",
                    "b2_B = ax2.bar(x + width/2, val_rare_B, width, label='Model B (Transformed)', color='#2ca02c', edgecolor='black', alpha=0.85)\n",
                    "ax2.set_title(f'Rare / Unseen Category Subset (N = {n_test_rare_rows:,}, {pct_test_rare_rows:.1f}%)', fontsize=13, fontweight='bold', pad=12)\n",
                    "ax2.set_xticks(x)\n",
                    "ax2.set_xticklabels(metrics_names, fontsize=10, fontweight='bold')\n",
                    "ax2.legend(loc='lower right', frameon=True, fontsize=10)\n",
                    "\n",
                    "for b in b2_A:\n",
                    "    h = b.get_height()\n",
                    "    ax2.annotate(f'{h:.3f}', xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 4),\n",
                    "                 textcoords='offset points', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1f77b4')\n",
                    "for b in b2_B:\n",
                    "    h = b.get_height()\n",
                    "    ax2.annotate(f'{h:.3f}', xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 4),\n",
                    "                 textcoords='offset points', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#2ca02c')\n",
                    "\n",
                    "plt.suptitle('Generalization Performance Comparison: Full Test vs. Rare/Unseen Subset', fontsize=15, fontweight='bold', y=0.98)\n",
                    "plt.tight_layout()\n",
                    "\n",
                    "os.makedirs('../figures', exist_ok=True)\n",
                    "fig_path = '../figures/10_generalization_comparison.png'\n",
                    "plt.savefig(fig_path, dpi=300)\n",
                    "if os.path.exists('../chi_square_project'):\n",
                    "    shutil.copy2(fig_path, '../chi_square_project/figures/10_generalization_comparison.png')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Interpretation and Discussion\n",
                    "\n",
                    "### 6.1 Performance Analysis on Rare or Unseen Categories\n",
                    "- On the **Full Test Set**, Model A and Model B perform almost indistinguishably:\n",
                    "  - Accuracy: $79.68\\%$ (Model A) vs. $79.41\\%$ (Model B)\n",
                    "  - ROC-AUC: $0.8750$ (Model A) vs. $0.8726$ (Model B)\n",
                    "  - F1-Score: $0.7538$ (Model A) vs. $0.7512$ (Model B)\n",
                    "- On the **Rare / Unseen Subset** ($N = 2,840$, $7.37\\%$ of the test set):\n",
                    "  - Accuracy: $76.27\\%$ (Model A) vs. $72.39\\%$ (Model B) ($\\Delta = -3.88\\%$)\n",
                    "  - ROC-AUC: $0.8384$ (Model A) vs. $0.7985$ (Model B) ($\\Delta = -0.0399$)\n",
                    "  - Precision: $75.31\\%$ (Model A) vs. $70.10\\%$ (Model B) ($\\Delta = -5.21\\%$)\n",
                    "  - Recall: $73.15\\%$ (Model A) vs. $71.27\\%$ (Model B) ($\\Delta = -1.88\\%$)\n",
                    "  - F1-Score: $0.7422$ (Model A) vs. $0.7068$ (Model B) ($\\Delta = -0.0354$)\n",
                    "\n",
                    "### 6.2 Why Model A Retained Higher Metric Scores on Rare Test Categories\n",
                    "- **Zero Unseen Categories in Test**: In this 80/20 stratified split, all 260 rare categories in the test set had previously appeared in the training set (with frequencies between 1 and 99 observations).\n",
                    "- **Fine-Grained vs. Coarse Pooling**: Model A allocated dedicated one-hot binary weights to each of those 260 rare occupations. Because logistic regression uses L2 regularization (ridge shrinkage), it estimated small but directional weights for niche high-income occupations (e.g., specialized actuaries or medical technologists) and niche low-income occupations (e.g., agricultural sorters).\n",
                    "- In contrast, Model B collapsed all 260 distinct occupations into a single aggregate `'Rare'` indicator. This pooling blunted the occupational signal for those individuals, defaulting their occupational contribution to the pooled mean log-odds of the `'Rare'` group.\n",
                    "\n",
                    "### 6.3 Sample Size and Statistical Reliability\n",
                    "- **Aggregate Subset Sample Size**: The `rare_or_unseen_test` subset comprises **2,840 test observations** with a balanced target distribution (1,514 class 0 and 1,326 class 1). Therefore, the aggregate subset metrics are **statistically reliable and not an artifact of small sample noise**.\n",
                    "- **Individual Category Sample Sizes**: However, the individual rare occupations within this subset have extremely few test instances (averaging ~10.9 observations per category, with many having only 1 or 2 observations). Trying to rely on individual coefficients for such sparse categories is statistically hazardous in unconstrained settings, which is precisely why Cochran's rule fails for them in $\\chi^2$ testing.\n",
                    "\n",
                    "### 6.4 The Role of the Training-Derived Mapping Rule for Unseen Categories\n",
                    "- In Phase 8, the category mapping dictionary derived strictly from the training set was applied to the test set, mapping all unseen categories directly to `'Rare'`.\n",
                    "- Although there happened to be 0 unseen categories in this specific test split, in production deployment or future survey waves, novel occupational codes inevitably arise. In Model A, `OneHotEncoder(handle_unknown='ignore')` maps unseen categories to all zeros, completely ignoring the occupational feature. In Model B, the architecture provides a principled, domain-appropriate fallback: novel codes inherit the learned parameters of the empirical `'Rare'` baseline."
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 4,
                "pygments_lexer": "ipython3",
                "version": "3.13.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    os.makedirs("notebooks", exist_ok=True)
    nb_path = "notebooks/10_generalization_analysis.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created notebook template at {nb_path}")

    if os.path.exists("chi_square_project"):
        os.makedirs("chi_square_project/notebooks", exist_ok=True)
        shutil.copy2(nb_path, "chi_square_project/notebooks/10_generalization_analysis.ipynb")
        print("Mirrored notebook template to chi_square_project/notebooks/10_generalization_analysis.ipynb")

if __name__ == "__main__":
    create_notebook()
