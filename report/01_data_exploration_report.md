# Phase 1: Data Exploration and Categorical Cardinality Report

## Executive Summary
This report documents the exploratory analysis of the California (CA) 2018 ACSIncome dataset from Fairlearn.

## 1. Dataset Dimensions & Integrity
- **Total Records**: 195,665 rows
- **Total Features**: 10 predictor columns + 1 target column (`PINCP`)
- **Missing Values**: 0 across all features and target
- **Duplicate Rows**: 2,910 rows (1.49% of dataset)
- **Data Status**: Kept in raw, unmodified condition without cleaning, transforming, or splitting

## 2. Categorical Variables & Cardinality
Treating integer-coded features as categorical yields 8 categorical variables:

| Column | Unique Categories | Missing | Most Frequent Category | Frequency of Top Cat | <10 Count | <50 Count | <100 Count |
|---|---|---|---|---|---|---|---|
| `COW` | 8 | 0 | 1.0 (Private For-Profit) | 127,815 | 0 | 0 | 0 |
| `SCHL` | 24 | 0 | 21.0 (Bachelor's Degree) | 46,526 | 0 | 0 | 2 |
| `MAR` | 5 | 0 | 1.0 (Married) | 102,442 | 0 | 0 | 0 |
| `OCCP` | 529 | 0 | 440.0 | 5,513 | 14 | 139 | 235 |
| `POBP` | 219 | 0 | 6.0 (California) | 94,450 | 29 | 85 | 112 |
| `RELP` | 18 | 0 | 0.0 (Reference Person) | 90,761 | 0 | 1 | 1 |
| `SEX` | 2 | 0 | 1.0 (Male) | 103,311 | 0 | 0 | 0 |
| `RAC1P` | 9 | 0 | 1.0 (White Alone) | 121,006 | 0 | 1 | 1 |

## 3. High-Cardinality Feature Analysis: OCCP
- `OCCP` exhibits extreme cardinality with **529 unique occupation codes**.
- 235 categories (44.4% of all unique codes) contain fewer than 100 observations.
- 139 categories contain fewer than 50 observations.
- 14 categories contain fewer than 10 observations.
- In contingency table analysis, these sparse categories will create cells with expected counts $E_{ij} < 5$, violating Cochran's condition and distorting asymptotic p-values.
- Therefore, `OCCP` is an optimal candidate for studying high-cardinality aggregation and dimensionality reduction methods.

## 4. Income Target Analysis: PINCP
- **Minimum**: $110.00
- **25th Percentile (Q1)**: $20,000.00
- **Median**: $40,000.00
- **Mean**: $65,192.05
- **75th Percentile (Q3)**: $80,000.00
- **Maximum**: $1,325,300.00
- **Standard Deviation**: $74,868.51

### Binarization Evaluation:
- The standard benchmark threshold of **$50,000** (used across Fairlearn and Ding et al., 2021) yields **41.1% positive cases** (`PINCP > 50000`), ensuring a well-balanced binary target suitable for logistic regression, decision trees, and chi-square independence testing.
