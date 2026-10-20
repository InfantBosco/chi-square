# Phase 9: Machine Learning Model Comparison Report

## 1. Overview and Objectives
In Phase 9 of the Chi-Square Statistical Machine Learning Pipeline, we benchmarked the downstream predictive impact of the cardinality reduction transformation developed in Phase 6 and verified in Phase 7 and Phase 8.

The classification goal is to predict whether an individual's annual income exceeds $50,000 (`income_above_50k`) based on ACS demographic and employment features.

Two Logistic Regression classifiers were trained strictly on the training set ($N_{\text{train}} = 154,204$) and evaluated out-of-sample on the untouched test set ($N_{\text{test}} = 38,551$):
- **Model A (Original Representation)**: All original categorical features, including raw `OCCP` with 529 distinct categories.
- **Model B (Transformed Representation)**: All original categorical features, replacing `OCCP` with `OCCP_grouped` (infrequent occupations with $< 100$ observations mapped to `'Rare'`).

Both models were fitted using identical hyperparameters:
```python
LogisticRegression(max_iter=1000, random_state=42)
```
with one-hot encoding handled via:
```python
OneHotEncoder(handle_unknown="ignore")
```

---

## 2. Experimental Results Summary

The out-of-sample test results are recorded below (from `results/09_model_comparison.csv`):

| Model Name | Encoded Features | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time (s) | Prediction Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A (Original)** | 813 | 0.7968 | 0.7597 | 0.7480 | 0.7538 | 0.8750 | 1.2313 s | 0.0441 s |
| **Model B (Transformed)** | 543 | 0.7941 | 0.7548 | 0.7477 | 0.7512 | 0.8726 | 0.8659 s | 0.0403 s |
| **Absolute Difference** | **-270** | **-0.0027** | **-0.0049** | **-0.0003** | **-0.0026** | **-0.0024** | **-0.3654 s** | **-0.0038 s** |
| **Relative % Change** | **-33.21%** | **-0.34%** | **-0.65%** | **-0.04%** | **-0.34%** | **-0.27%** | **-29.68%** | **-8.62%** |

---

## 3. Visualizations Generated

1. **Performance Metrics Comparison**: `figures/09_model_performance_comparison.png`
   - Grouped bar chart comparing Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
2. **Encoded Feature Dimensionality Comparison**: `figures/09_encoded_feature_count_comparison.png`
   - Bar chart visualizing the 813 vs. 543 feature dimension reduction (-33.2% compression).

---

## 4. Key Interpretations & Scientific Insights

### 4.1 Did Grouping Reduce Dimensionality?
**Yes, substantially.**
- The raw `OCCP` feature contained 529 unique occupational codes in the training set, generating 529 one-hot binary columns.
- After applying the minimum threshold of 100 observations ($N_{\text{min}} = 100$), 271 sparse occupations were collapsed into a single `'Rare'` category, leaving 258 retained occupations plus 1 `'Rare'` category (259 categories total).
- Consequently, the total one-hot encoded dimensional space decreased from **813 features to 543 features**, representing an absolute reduction of **270 features (33.2% parameter compression)**.
- This directly reduces model memory footprint and accelerates training convergence (training time decreased by nearly $30\%$).

### 4.2 Was Predictive Performance Maintained or Improved?
**Predictive performance was exceptionally well-preserved.**
- **Accuracy**: $79.68\%$ vs. $79.41\%$ (difference of only $0.27\%$).
- **Recall**: $74.80\%$ vs. $74.77\%$ (difference of only $0.03\%$).
- **F1-Score**: $0.7538$ vs. $0.7512$ (difference of only $0.0026$).
- **ROC-AUC**: $0.8750$ vs. $0.8726$ (difference of only $0.0024$).

Although the raw representation (Model A) holds a negligible numerical margin ($<0.3\%$) due to having 270 additional degrees of freedom to capture subtle occupational nuances, the difference is practically imperceptible. The 271 sparse occupations accounted for only $4.94\%$ of the population. Grouping them prevented the model from estimating unstable coefficients on small cell counts while preserving almost the entirety of the discriminative signal.

### 4.3 The Trade-off Between Statistical Validity and Machine Learning Performance
- **Statistical Validity**:
  In Phase 5, the raw contingency table between `OCCP` and `income_above_50k` violated Cochran's rule severely: **44.05% of cells had expected counts below 5**, completely invalidating the asymptotic $\chi^2$ distribution assumption for hypothesis testing. In Phase 7, the transformed feature reduced Cochran violations to **0.00%**, restoring full statistical validity.
- **Machine Learning Regularization**:
  Linear machine learning algorithms (like regularized logistic regression) can technically optimize loss functions even in the presence of sparse binary indicators without numerical crashing. However, sparse indicators have high sample variance and risk overfitting.
- **The Synthesis**:
  By reducing cardinality through a principled frequency threshold grounded in contingency analysis, we achieve:
  1. Complete adherence to statistical distribution assumptions ($0\%$ Cochran violations).
  2. A $33.2\%$ reduction in model complexity.
  3. Stable, generalizable parameter estimates with virtually zero degradation in out-of-sample ML classification metrics.
  4. Robust handling of novel or unseen occupational categories in test or production environments.
