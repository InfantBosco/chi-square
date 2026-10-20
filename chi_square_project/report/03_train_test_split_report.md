# Phase 3: Stratified Train-Test Splitting & Leakage Prevention Report

## 1. Overview
This report details the partitioning of the deduplicated, cleaned California (CA) 2018 ACSIncome dataset into isolated training (80%) and testing (20%) subsets.

## 2. Partition Summary
- **Total Initial Clean Observations**: 192,755
- **Train Proportion**: 80.0% (`test_size=0.20`, `random_state=42`)
- **Stratification Variable**: `y` (`income_above_50k`)

| Partition | Total Rows | Class 0 ($\le \$50\text{k}$) | % Class 0 | Class 1 ($> \$50\text{k}$) | % Class 1 |
|---|---|---|---|---|---|
| **Full Dataset** | 192,755 | 112,610 | 58.4213% | 80,145 | 41.5787% |
| **Training Set (`X_train`, `y_train`)** | 154,204 | 90,088 | 58.4213% | 64,116 | 41.5787% |
| **Testing Set (`X_test`, `y_test`)** | 38,551 | 22,522 | 58.4213% | 16,029 | 41.5787% |

## 3. Stratification Integrity
The proportion of positive labels is identical across training and testing partitions ($41.5787\%$), confirming that class distributions are completely preserved without sampling bias.

## 4. Strict Isolation & Leakage Prevention Protocol
To guarantee sound scientific and machine learning rigor:
- **No data preprocessing, scaling, category pooling, frequency thresholding, or encoding has been fitted on `X_test` or the full dataset.**
- All transformers and statistical encoders must learn their parameters (e.g., occupation category groupings, contingency tables, target encodings) **exclusively from `X_train`**.
- `X_test` is reserved strictly as holdout data to evaluate statistical validity and model generalization.
