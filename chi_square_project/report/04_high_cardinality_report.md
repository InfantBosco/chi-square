# Phase 4: High-Cardinality Feature Analysis Report

## 1. Executive Summary
This report analyzes categorical cardinality and sparsity across the isolated training partition (`X_train`, 154,204 rows) of the California 2018 ACSIncome dataset.

## 2. Categorical Cardinality Profile
| Column Name | Unique Categories | Missing Values | Most Frequent Category | Frequency of Top Cat | Categories $<10$ | Categories $<50$ | Categories $<100$ | Cardinality Ratio |
|---|---|---|---|---|---|---|---|---|
| `COW` | 8 | 0 | 1.0 | 100,737 | 0 | 0 | 0 | 0.000052 |
| `SCHL` | 24 | 0 | 21.0 | 36,970 | 0 | 0 | 3 | 0.000156 |
| `MAR` | 5 | 0 | 1.0 | 81,755 | 0 | 0 | 0 | 0.000032 |
| **`OCCP`** | **529** | **0** | **440.0** | **4,431** | **23** | **172** | **271** | **0.003431** |
| `POBP` | 218 | 0 | 6.0 | 74,068 | 35 | 92 | 116 | 0.001414 |
| `RELP` | 18 | 0 | 0.0 | 72,383 | 0 | 1 | 1 | 0.000117 |
| `SEX` | 2 | 0 | 1.0 | 81,459 | 0 | 0 | 0 | 0.000013 |
| `RAC1P` | 9 | 0 | 1.0 | 95,389 | 0 | 1 | 1 | 0.000058 |

## 3. High-Cardinality Feature Selection: `OCCP`
- **Feature Name**: `OCCP` (Occupation Code)
- **Unique Categories**: 529
- **Cardinality Ratio**: 0.003431
- Saved to: `results/04_selected_high_cardinality_feature.txt`

## 4. Sparsity & Long-Tail Distribution
- **Single-Occurrence Categories**: 2 (0.38%)
- **Categories $< 10$ Observations**: 23 (4.35%)
- **Categories $< 50$ Observations**: 172 (32.51%)
- **Categories $< 100$ Observations**: 271 (51.23%)
- **Total Observations in Rare Categories ($< 100$)**: 11,326 (7.34% of training set)

## 5. Statistical Independence Implications for $\chi^2$ Tests
1. **Contingency Table Dimensions**:
   - Testing `OCCP` (529 levels) against binary target `income_above_50k` (2 levels) creates a $529 \times 2 = 1,058$-cell table.
2. **Cochran's Rule Violation**:
   - Over half of the categories (271 out of 529) have $< 100$ observations. Under $H_0$, dozens of individual cells have expected counts $E_{ij} < 5$ (and some $< 1$), blatantly violating Cochran's condition.
3. **Statistical Distortion**:
   - Small expected frequencies cause the denominators in Pearson's $\chi^2 = \sum \frac{(O_{ij}-E_{ij})^2}{E_{ij}}$ to approach zero, artificially inflating the test statistic and rendering standard asymptotic p-values invalid.
4. **Resolution Strategy**:
   - Grouping rare categories (e.g. into an `"Other / Rare"` bucket or frequency-based clusters) is statistically mandatory prior to hypothesis testing.
