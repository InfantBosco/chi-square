# Phase 6: Category Reduction & Rare Grouping Report

## 1. Executive Summary
This report documents the category pooling and dimensionality reduction pipeline implemented on the high-cardinality occupation feature (`OCCP`) using the training dataset (`X_train`, 154,204 rows).

## 2. Threshold Justification
- **Minimum-Frequency Threshold**: **100 training observations**.
- **Theoretical Rationale**:
  In Phase 5, the raw $529 \times 2$ contingency table had 60 cells with expected frequencies $E_{ij} < 5.0$.
  With binary target proportions of 58.42% (class 0) and 41.58% (class 1), any category having $\ge 100$ total observations is guaranteed to produce an expected frequency for the minority class of:
  $$E_{i,1} \ge 100 \times 0.4158 = 41.58 > 5.0$$
  This comfortably exceeds Cochran's threshold ($E_{ij} \ge 5$) by more than an eightfold margin, entirely eliminating small-cell asymptotic distortion.

## 3. Reduction Performance Metrics
| Metric | Value |
|---|---|
| Original Unique Categories | 529 |
| Transformed Unique Categories | 259 (258 preserved + 1 `"Rare"`) |
| Categories Grouped into `"Rare"` | 271 (51.23% of original categories) |
| Observations Assigned to `"Rare"` | 11,326 (7.3448% of training dataset) |
| Cardinality Reduction Percentage | **51.04%** |

## 4. Preservation of Schema & Original Data
- The original `OCCP` column was strictly preserved unchanged.
- A new column `OCCP_grouped` was created and appended to `data/processed/X_train_grouped.csv`.
- The category mapping dictionary was exported to `results/06_category_mapping.json`.

## 5. Leakage Prevention & Test Set Protocol
- The mapping was learned **strictly from `X_train`** without observing test data.
- During future evaluation, `X_test` will be transformed using the exact dictionary stored in `results/06_category_mapping.json`. Any occupation code unseen in training or appearing $< 100$ times will map directly to `"Rare"`, guaranteeing deterministic, leakage-free processing.
