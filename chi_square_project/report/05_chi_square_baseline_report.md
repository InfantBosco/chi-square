# Phase 5: Baseline Chi-Square Test of Independence Report

## 1. Executive Summary
This report documents the baseline Pearson's Chi-Square ($\chi^2$) test of independence between the uncompressed high-cardinality occupation feature (`OCCP`) and binary income (`income_above_50k`) using the training set (`X_train`, 154,204 observations).

## 2. Hypothesis Testing Framework
- **Null Hypothesis ($H_0$)**: Occupation (`OCCP`) and high income status (`income_above_50k`) are statistically independent in the population.
- **Alternative Hypothesis ($H_1$)**: Occupation and income status are statistically associated (dependent).

## 3. Contingency Table & Chi-Square Metrics
- **Contingency Table Dimensions**: 529 occupation rows $\times$ 2 target columns (1,058 total cells)
- **Pearson's $\chi^2$ Statistic**: **51,535.8043** (Verified identical between manual computation and `scipy.stats.chi2_contingency`)
- **Degrees of Freedom**: **528** ($(529 - 1) \times (2 - 1)$)
- **p-value**: **0.0** ($p < 10^{-300}$)

| Metric | Value |
|---|---|
| Number of categories | 529 |
| Chi-square statistic | 51,535.8043 |
| Degrees of freedom | 528 |
| p-value | 0.0 |
| Cells with expected count $< 5$ | 60 (5.67%) |
| Categories with $< 10$ observations | 23 (4.35%) |
| Categories with $< 50$ observations | 172 (32.51%) |
| Categories with $< 100$ observations | 271 (51.23%) |

## 4. Methodological Interpretation & Statistical Caveats
1. **Strong Global Association**:
   The enormous $\chi^2$ value of $51,535.8$ overwhelmingly rejects the null hypothesis of independence ($p \approx 0.0$), indicating that occupation codes contain substantial explanatory power for predicting income bracket.
2. **Cochran's Assumption Violations in Sparse Tails**:
   - 60 cells have expected frequencies $E_{ij} < 5.0$, and multiple rare occupations have $E_{ij} < 1.0$.
   - When expected counts approach zero, the term $\frac{(O_{ij}-E_{ij})^2}{E_{ij}}$ has an unstable denominator, causing random sampling noise in rare categories to disproportionately inflate the $\chi^2$ test statistic.
3. **Degrees of Freedom Inflation**:
   - With 528 degrees of freedom, the test assesses global departure from independence across all 529 occupations simultaneously, but obscures which occupation clusters drive the association.
4. **Baseline Purpose**:
   - This baseline evaluation proves the statistical necessity of Phase 6: grouping low-frequency occupation categories to eliminate sparse cells, conform to Cochran's criteria, and achieve a robust, interpretable independence test.
