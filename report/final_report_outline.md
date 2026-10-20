# Statistical and Machine Learning Project Final Report Outline

**Project Title**: High-Cardinality Categorical Feature Reduction via $\chi^2$ Contingency Analysis: Balancing Statistical Validity and Machine Learning Generalization in ACS Income Prediction

---

## 1. Introduction
- 1.1 Context of categorical data in machine learning and applied statistics
- 1.2 The challenge of high cardinality in survey and administrative microdata
- 1.3 Overview of the US Census Bureau American Community Survey (ACS) Income benchmark
- 1.4 Objectives of this investigation: bridging asymptotic $\chi^2$ hypothesis testing with predictive regularized linear classification
- 1.5 Pipeline roadmap and phase progression (Phases 0 through 11)

## 2. Problem Statement
- 2.1 The curse of dimensionality in one-hot encoded representations
- 2.2 Violation of asymptotic distribution assumptions in contingency tables (Cochran's rule)
- 2.3 Statistical vs. computational trade-offs: hypothesis testing validity vs. predictive variance
- 2.4 Risks of information leakage during categorical transformation and out-of-sample inference
- 2.5 The open question: Does grouping sparse categories to satisfy statistical validity degrade downstream machine learning performance?

## 3. Literature Review
- 3.1 Categorical Encoding Techniques: One-Hot Encoding, Target/Bayesian Encoding, Frequency Encoding, and Category Clustering
- 3.2 Classical Independence Testing: Pearson (1900), Fisher (1922), and Cochran's asymptotic validity criteria (1952, 1954)
- 3.3 High Cardinality and Sparse Cell Handling in Applied Statistics: Agresti (2002), Yates correction, and category collapsing
- 3.4 Modern Benchmarking: Ding et al. (2021) "Retiring Adult: New Datasets for Fair Machine Learning" (Fairlearn ACSIncome)
- 3.5 Regularization in High Dimensions: Hastie, Tibshirani, and Friedman (2009) on Ridge ($\ell_2$) shrinkage with sparse dummy variables

## 4. Dataset Description
- 4.1 Data source: 2018 US Census Bureau American Community Survey (ACS) PUMS via Fairlearn
- 4.2 Geographical and temporal scope: California (state = "CA", year = 2018)
- 4.3 Observation volume: 195,665 initial survey records; 192,755 unique records post-deduplication
- 4.4 Target specification: Personal continuous income (`PINCP`), binarized at the median policy threshold of $\$50,000$ (`income_above_50k`)
- 4.5 Feature taxonomy:
  - Demographic & Employment features (10 total): `AGEP`, `COW`, `SCHL`, `MAR`, `OCCP`, `POBP`, `RELP`, `WKHP`, `SEX`, `RAC1P`
  - Identification of numeric features (`AGEP`, `WKHP`) vs. integer-coded categorical features (8 features)

## 5. Methodology
- 5.1 Strict data hygiene and anti-leakage architecture: Stratified 80/20 train/test split ($N_{\text{train}} = 154,204$, $N_{\text{test}} = 38,551$, random seed = 42)
- 5.2 Training-only discovery: Cardinality analysis, frequency estimation, threshold determination, and contingency table computation performed exclusively on training data
- 5.3 Deterministic out-of-sample mapping rule: Mapping dictionary serialization and execution on untouched test data
- 5.4 Controlled machine learning experiment: Dual-model comparison holding algorithm, solver, split, and non-target features identical
- 5.5 Subgroup generalization audit: Granular isolation of rare and novel category performance

## 6. Mathematical Background
- 6.1 Contingency Table Formulation:
  $$O_{ij} = \sum_{k=1}^N \mathbb{I}(X_k = c_i \land Y_k = j)$$
- 6.2 Expected Cell Frequency under Null Hypothesis of Independence:
  $$E_{ij} = \frac{R_i \cdot C_j}{N}$$
- 6.3 Pearson's $\chi^2$ Test Statistic:
  $$\chi^2 = \sum_{i=1}^r \sum_{j=1}^c \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$
- 6.4 Degrees of Freedom:
  $$\text{df} = (r - 1)(c - 1)$$
- 6.5 Cochran's Asymptotic Validity Conditions:
  - Minimum expected frequency: No cell has $E_{ij} < 1$
  - Sparsity ceiling: No more than $20\%$ of cells have $E_{ij} < 5$
- 6.6 Logistic Regression with $\ell_2$ Regularization:
  $$\min_{\mathbf{w}, b} \frac{1}{N} \sum_{k=1}^N \log(1 + \exp(-y_k (\mathbf{w}^T \mathbf{x}_k + b))) + \frac{1}{2C} \|\mathbf{w}\|_2^2$$

## 7. High-Cardinality Analysis
- 7.1 Cardinality survey across all categorical predictors in the training partition
- 7.2 Selection of the maximum-cardinality predictor: Occupation Code (`OCCP`, 529 unique categories, cardinality ratio = 0.00343)
- 7.3 Distributional tail analysis:
  - 23 categories with $< 10$ observations
  - 172 categories with $< 50$ observations
  - 271 categories with $< 100$ observations (51.2% of all distinct categories)
- 7.4 Skewness and long-tail concentration: The top 25 occupations account for over 38% of employment

## 8. Baseline Chi-Square Analysis
- 8.1 Contingency table construction: $529 \times 2$ matrix (1,058 cells) between raw `OCCP` and `income_above_50k`
- 8.2 Expected frequencies and Cochran's rule audit:
  - 60 cells with $E_{ij} < 5$ ($5.6711\%$ of cells)
  - 23 categories with $N < 10$
- 8.3 Computed statistic: $\chi^2 = 51,535.8043$, $\text{df} = 528$, $p < 10^{-300}$
- 8.4 Verification: Exact match between manual matrix derivation and `scipy.stats.chi2_contingency`
- 8.5 Diagnostic conclusion: While the global p-value is nominally zero, sparse cells violate the continuous asymptotic approximation, inflating local cell contributions

## 9. Cardinality-Reduction Strategy
- 9.1 Rationale for the minimum-frequency threshold ($N_{\text{min}} = 100$ observations)
- 9.2 Derivation of the category mapping: Retaining 258 robust categories and collapsing 271 sparse categories into `'Rare'`
- 9.3 Creation and serialization of `06_category_mapping.json`
- 9.4 Verification of feature alignment: $100\%$ row preservation with zero data loss ($N = 154,204$)
- 9.5 Out-of-sample mapping enforcement on test data: Transforming test features without fitting on test statistics

## 10. Transformed Chi-Square Analysis
- 10.1 Contingency table construction: $259 \times 2$ matrix (518 cells) between `OCCP_grouped` and `income_above_50k`
- 10.2 Post-transformation Cochran audit:
  - 0 cells with $E_{ij} < 5$ ($0.0000\%$ violation rate)
  - Complete elimination of sparse cells
- 10.3 Statistical comparison:
  - Transformed $\chi^2 = 48,444.0265$, $\text{df} = 258$, $p < 10^{-300}$
  - Absolute $\chi^2$ change: $-3,091.7778$ ($-6.00\%$)
  - Degrees of freedom reduction: $-51.14\%$
- 10.4 Conclusion: Preservation of strong statistical association with $100\%$ statistical test validity

## 11. Machine-Learning Evaluation
- 11.1 Experimental protocol: Logistic Regression (`max_iter=1000`, `random_state=42`) with `OneHotEncoder(handle_unknown='ignore')`
- 11.2 Model A (Original OCCP, 813 encoded features) vs. Model B (Grouped OCCP, 543 encoded features)
- 11.3 Out-of-sample evaluation on untouched test set ($N_{\text{test}} = 38,551$):
  - Accuracy: $79.68\%$ (Model A) vs. $79.41\%$ (Model B) [$\Delta = -0.27\%$]
  - Precision: $75.97\%$ (Model A) vs. $75.48\%$ (Model B) [$\Delta = -0.49\%$]
  - Recall: $74.80\%$ (Model A) vs. $74.77\%$ (Model B) [$\Delta = -0.03\%$]
  - F1-Score: $0.7538$ (Model A) vs. $0.7512$ (Model B) [$\Delta = -0.0026$]
  - ROC-AUC: $0.8750$ (Model A) vs. $0.8726$ (Model B) [$\Delta = -0.0024$]
- 11.4 Computational efficiency:
  - Feature space reduction: $-270$ features ($-33.21\%$)
  - Training wall-clock time reduction: $1.2313\text{ s} \to 0.8659\text{ s}$ ($-29.68\%$)
  - Prediction latency reduction: $0.0441\text{ s} \to 0.0403\text{ s}$ ($-8.62\%$)

## 12. Generalization Testing
- 12.1 Isolation of the `rare_or_unseen_test` subset ($N = 2,840$, $7.37\%$ of test instances)
- 12.2 Rare and unseen category audit: 260 rare test categories, 0 unseen test categories
- 12.3 Performance on rare/unseen subset:
  - Accuracy: $76.27\%$ (Model A) vs. $72.39\%$ (Model B) [$\Delta = -3.88\%$]
  - Precision: $75.31\%$ (Model A) vs. $70.10\%$ (Model B) [$\Delta = -5.21\%$]
  - Recall: $73.15\%$ (Model A) vs. $71.27\%$ (Model B) [$\Delta = -1.88\%$]
  - F1-Score: $0.7422$ (Model A) vs. $0.7068$ (Model B) [$\Delta = -0.0354$]
  - ROC-AUC: $0.8384$ (Model A) vs. $0.7985$ (Model B) [$\Delta = -0.0399$]
- 12.4 Mechanism: Granular $\ell_2$-regularized shrinkage vs. uniform pooling into the `'Rare'` cluster mean

## 13. Results and Discussion
- 13.1 Synthesis of findings across statistical and machine learning domains
- 13.2 The trade-off spectrum: Statistical purity vs. loss of granular subgroup variance
- 13.3 Practical utility: Why 33.2% parameter compression with $<0.3\%$ accuracy loss is highly desirable in production systems
- 13.4 Deterministic robustness: Handling novel categories without silent pipeline failure

## 14. Limitations
- 14.1 Arbitrary thresholding: Sensitivity of the $N_{\text{min}} = 100$ cutoff
- 14.2 Coarse pooling: Loss of nuance among heterogeneous rare occupations grouped into a single bucket
- 14.3 Linear model focus: Absence of tree-based models (e.g. LightGBM, CatBoost) or target encoding benchmarks
- 14.4 Geographical confinement: Findings specific to California ACS 2018 microdata

## 15. Conclusion
- 15.1 Core conclusions addressing the central research questions
- 15.2 Recommendations for applied data scientists and statisticians
- 15.3 Summary of final project deliverables and reproducible code base

## 16. References
- 16.1 Primary statistical literature (Pearson, Cochran, Agresti)
- 16.2 Machine learning and algorithmic fair representation references (Ding et al., Hastie et al., Pedregosa et al.)
- 16.3 U.S. Census Bureau documentation for ACS PUMS 2018
