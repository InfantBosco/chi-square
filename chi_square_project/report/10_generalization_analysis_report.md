# Phase 10: Generalization Analysis Report (Rare and Unseen Categories)

## 1. Executive Summary & Purpose
In Phase 10 of the Chi-Square Statistical Machine Learning Pipeline, we investigated how dimensionality reduction via $\chi^2$-guided category grouping impacts out-of-sample generalization on the most vulnerable partition of the evaluation data:
- **`rare_or_unseen_test` subset**: All test observations where the original high-cardinality occupational feature (`OCCP`) either:
  1. Appeared fewer than 100 times in the training partition ($< 100$ observations), or
  2. Did not appear in the training partition at all ($0$ observations).

We re-trained both:
- **Model A (Original Representation)**: Raw categorical features with 529 unique `OCCP` occupational dummy variables (813 total encoded features).
- **Model B (Transformed Representation)**: Grouped categorical features with infrequent occupations mapped to `'Rare'` (543 total encoded features).

Both models were evaluated on both the **Full Test Set** ($N = 38,551$) and the **Rare/Unseen Test Subset** ($N = 2,840$).

---

## 2. Rare and Unseen Category Audit

From [`results/10_rare_unseen_category_summary.csv`](file:///d:/Github%20Repo/C-Square/results/10_rare_unseen_category_summary.csv):

| Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **Number of rare test categories** | **260** | Occupations present in training with $< 100$ occurrences. |
| **Number of unseen test categories** | **0** | Occupations completely absent from training. |
| **Test rows in rare/unseen subset** | **2,840** | Total test individuals belonging to rare occupations. |
| **Percentage of test rows in subset** | **7.3669%** | Fraction of the test set impacted by rare category grouping. |

---

## 3. Generalization Performance Results

From [`results/10_generalization_results.csv`](file:///d:/Github%20Repo/C-Square/results/10_generalization_results.csv):

| Evaluation Partition | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Full Test Set** ($N = 38,551$) | Model A (Original) | **0.7968** | **0.7597** | **0.7480** | **0.7538** | **0.8750** |
| | Model B (Transformed) | **0.7941** | **0.7548** | **0.7477** | **0.7512** | **0.8726** |
| | *Difference ($\Delta$)* | *-0.0027* | *-0.0049* | *-0.0003* | *-0.0026* | *-0.0024* |
| **Rare/Unseen Subset** ($N = 2,840$) | Model A (Original) | **0.7627** | **0.7531** | **0.7315** | **0.7422** | **0.8384** |
| | Model B (Transformed) | **0.7239** | **0.7010** | **0.7127** | **0.7068** | **0.7985** |
| | *Difference ($\Delta$)* | *-0.0388* | *-0.0521* | *-0.0188* | *-0.0354* | *-0.0399* |

---

## 4. Key Interpretations & Scientific Insights

### 4.1 Did Grouping Improve Performance on Rare or Unseen Categories?
**No, grouping into a single `'Rare'` category yielded slightly lower performance on the rare subset than the unregularized multi-dummy representation.**
- On the full test set, Model A and Model B are virtually indistinguishable ($\Delta \text{Accuracy} = -0.27\%$, $\Delta \text{ROC-AUC} = -0.0024$).
- However, when isolating the 2,840 individuals in the rare occupational categories, Model A achieved higher accuracy ($76.27\%$ vs. $72.39\%$) and higher ROC-AUC ($0.8384$ vs. $0.7985$).

### 4.2 Is the Difference Consistent Across Metrics?
**Yes, the difference is strictly consistent across all five metrics:**
- Accuracy: $-3.88\%$
- Precision: $-5.21\%$
- Recall: $-1.88\%$
- F1-Score: $-3.54\%$
- ROC-AUC: $-0.0399$

### 4.3 Why Did Model A Outperform Model B on the Rare Subset?
1. **Zero Unseen Categories in the Split**: In this stratified split, all 260 rare categories in the test set had appeared in the training set (with counts between 1 and 99).
2. **Individual vs. Pooled Weight Estimation**:
   - In Model A, `OneHotEncoder` gave each of the 260 rare occupations its own individual weight. Because Scikit-Learn's `LogisticRegression` applies L2 penalty ($\ell_2$ ridge regularization), the model shrank these coefficients toward zero without forcing them to be identical. Thus, high-income niche jobs (e.g. specialized engineering consultants, actuaries) received positive log-odds, while low-income niche jobs (e.g. specialized craftspersons, sorters) received negative log-odds.
   - In Model B, all 260 distinct occupations were collapsed into a single uniform `'Rare'` bucket. This pooling coerced every rare job to share the exact same occupational weight (the population average of rare jobs), blurring the distinguishing signals between high-earning rare jobs and low-earning rare jobs.

### 4.4 Sample Size and Statistical Reliability
- **Aggregate Reliability**: The `rare_or_unseen_test` subset contains **2,840 observations** (1,514 negative class and 1,326 positive class). With thousands of samples and a balanced target proportion (46.7% positive class), the aggregate metrics are **highly statistically reliable** and not an artifact of small-sample variance.
- **Individual Category Fragility**: While the *aggregate* subset is large, individual occupations within it are fragile (averaging just 10.9 test observations per category). In classical hypothesis testing, this sparsity severely violates Cochran's rule (44.05% of cells $<5$).

### 4.5 The Principled Role of the Training-Derived Mapping Rule for Unseen Categories
- All unseen categories in the pipeline were strictly mapped to `'Rare'` using the rule established in Phase 6:
  $$\text{OCCP}_{\text{test}} \notin \text{Categories}_{\text{train}} \implies \text{OCCP}_{\text{grouped}} = \text{'Rare'}$$
- Although 0 unseen categories occurred in this particular test set, in future inference or external populations, unseen categories inevitably appear.
- While Model A's `OneHotEncoder(handle_unknown='ignore')` maps unseen categories to all zeros (defaulting to the global intercept), Model B's mapping provides a domain-informed empirical prior: assigning the observation to the learned `'Rare'` group weight.

---

## 5. Artifacts and Visualizations
- Summary CSV: [`results/10_rare_unseen_category_summary.csv`](file:///d:/Github%20Repo/C-Square/results/10_rare_unseen_category_summary.csv)
- Generalization CSV: [`results/10_generalization_results.csv`](file:///d:/Github%20Repo/C-Square/results/10_generalization_results.csv)
- Comparison Chart: [`figures/10_generalization_comparison.png`](file:///d:/Github%20Repo/C-Square/figures/10_generalization_comparison.png)
- Jupyter Notebook: [`notebooks/10_generalization_analysis.ipynb`](file:///d:/Github%20Repo/C-Square/notebooks/10_generalization_analysis.ipynb)
