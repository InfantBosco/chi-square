# Phase 11: Final Analysis, Synthesis, and Project Report

## 1. Executive Project Summary

This project systematically investigated the theoretical and practical implications of high cardinality in categorical predictors within applied statistics and machine learning. Using the 2018 American Community Survey (ACS) Income microdata for California ($N = 192,755$), we analyzed the relationship between occupation codes (`OCCP`) and annual personal income exceeding $\$50,000$ (`income_above_50k`).

---

## 2. Master Project Metrics & Summary Table

Below is the verified summary table compiled in [`results/11_final_project_summary.csv`](file:///d:/Github%20Repo/C-Square/results/11_final_project_summary.csv):

| Pipeline Domain | Metric Description | Baseline (Original / Model A) | Transformed (Grouped / Model B) | Impact / $\Delta$ |
| :--- | :--- | :---: | :---: | :---: |
| **Categorical Cardinality** | Unique Categories ($OCCP$) | **529** | **259** | **-270 categories (-51.04%)** |
| **Machine Learning Space** | One-Hot Encoded Features | **813** | **543** | **-270 features (-33.21%)** |
| **Statistical Independence**| $\chi^2$ Test Statistic | **51,535.8043** | **48,444.0265** | **-3,091.7778 (-6.00%)** |
| | Degrees of Freedom ($\text{df}$) | **528** | **258** | **-270 df (-51.14%)** |
| | Asymptotic $p$-value | **0.0** ($p < 10^{-300}$) | **0.0** ($p < 10^{-300}$) | Preserved ($p \approx 0$) |
| **Cochran Rule Validity** | Cells with Expected Count $E_{ij} < 5$ | **60 (5.6711%)** | **0 (0.0000%)** | **100% Validated** |
| | Categories with $N < 10$ | **23** | **0** | Completely eliminated |
| | Categories with $N < 50$ | **172** | **0** | Completely eliminated |
| **Full Test Performance** | Test Set Accuracy | **0.7968** | **0.7941** | **-0.0027 (-0.34%)** |
| ($N_{\text{test}} = 38,551$) | Test Set Precision | **0.7597** | **0.7548** | **-0.0049 (-0.65%)** |
| | Test Set Recall | **0.7480** | **0.7477** | **-0.0003 (-0.04%)** |
| | Test Set F1-score | **0.7538** | **0.7512** | **-0.0026 (-0.34%)** |
| | Test Set ROC-AUC | **0.8750** | **0.8726** | **-0.0024 (-0.27%)** |
| **Computational Overhead** | Logistic Regression Training Time | **1.2313 s** | **0.8659 s** | **-0.3654 s (-29.68%)** |
| | Logistic Regression Prediction Time | **0.0441 s** | **0.0403 s** | **-0.0038 s (-8.62%)** |
| **Rare / Unseen Subset** | Rare Subset Accuracy | **0.7627** | **0.7239** | **-0.0388 (-5.09%)** |
| ($N_{\text{subset}} = 2,840$) | Rare Subset Precision | **0.7531** | **0.7010** | **-0.0521 (-6.92%)** |
| | Rare Subset Recall | **0.7315** | **0.7127** | **-0.0188 (-2.57%)** |
| | Rare Subset F1-score | **0.7422** | **0.7068** | **-0.0354 (-4.77%)** |
| | Rare Subset ROC-AUC | **0.8384** | **0.7985** | **-0.0399 (-4.76%)** |

---

## 3. Visual Dashboard
The complete 4-panel dashboard is stored at [`figures/11_final_results_summary.png`](file:///d:/Github%20Repo/C-Square/figures/11_final_results_summary.png):
1. **Panel 1**: Category count comparison before and after grouping ($529 \to 259$).
2. **Panel 2**: Encoded feature space dimensionality ($813 \to 543$).
3. **Panel 3**: Full test set model performance ($N = 38,551$).
4. **Panel 4**: Generalization performance on the rare/unseen subset ($N = 2,840$).

---

## 4. Final Scientific Answers to Core Questions

### 1. Did cardinality reduction reduce representation size?
**Yes.** Category count decreased by **$51.04\%$** ($529 \to 259$), and one-hot encoded dummy variables were reduced by **$33.21\%$** ($813 \to 543$). Training convergence speed improved by **$29.68\%$** ($1.23\text{ s} \to 0.87\text{ s}$).

### 2. Did it reduce sparse contingency-table cells?
**Yes, completely.** Baseline expected frequency violations of Cochran's rule ($E_{ij} < 5$) fell from **$5.6711\%$ (60 cells)** to **$0.0000\%$ (0 cells)**. Categories with fewer than 50 observations dropped from 172 to 0.

### 3. Did it preserve or change statistical association?
**Preserved.** The association between occupation and income remained decisively non-independent with $p \approx 0$ ($p < 10^{-300}$). The $\chi^2$ statistic changed from $51,535.8043$ to $48,444.0265$ (a slight $6.00\%$ reduction), reflecting the removal of spurious high chi-square components from sparse cells with tiny expected values.

### 4. Did it improve, maintain, or reduce classification performance?
**Maintained.** Global out-of-sample accuracy changed from $79.68\%$ to $79.41\%$ ($\Delta = -0.27\%$), F1-score from $0.7538$ to $0.7512$ ($\Delta = -0.0026$), and ROC-AUC from $0.8750$ to $0.8726$ ($\Delta = -0.0024$).

### 5. Did it improve performance on rare or unseen categories?
**No.** Model A outperformed Model B on the rare test subset ($76.27\%$ vs. $72.39\%$ accuracy, $0.8384$ vs. $0.7985$ ROC-AUC) because all 260 rare categories had between 1 and 99 training samples. Model A's $\ell_2$-regularized regression assigned small directional weights to distinct rare jobs, whereas Model B's aggregate `'Rare'` indicator forced all rare jobs to share a single pooled mean log-odds.

### 6. What are the limitations of the rare-category threshold?
- **Arbitrary cutoff**: The $N_{\text{min}} = 100$ threshold does not adapt to category-specific variances.
- **Heterogeneous pooling**: Disparate high-earning and low-earning niche jobs are merged into a single bucket.
- **Scale sensitivity**: A threshold of 100 observations represents different percentages depending on dataset scale ($0.065\%$ in ACS vs. much larger fractions in smaller datasets).

### 7. What future improvements could be made?
- **Supervised multi-bucket clustering**: Splitting rare categories into `'Rare_High_Income'` and `'Rare_Low_Income'` based on training empirical class probabilities.
- **Hierarchical SOC roll-up**: Mapping rare occupations to 2-digit Standard Occupational Classification major groups.
- **Empirical Bayes Target Encoding**: Utilizing smoothed target encoding with James-Stein or empirical Bayes priors to eliminate dummy variables entirely.
