# Cryptocurrency Fraud Detection on the Elliptic Dataset
## DA3 Final Report — 100% Project Completion

**Course:** Programming for Data Science
**Date:** 15 September 2026
**Team size:** 2

---

## 1. Executive Summary

This project develops and evaluates machine learning models that classify Bitcoin transactions in the Elliptic dataset as licit or illicit. Eleven classification algorithms were trained on a SMOTE-balanced training set covering time steps 1–34 and evaluated on a held-out test set of 16,670 transactions covering time steps 35–49, of which 1,083 (6.5%) are illicit.

The recommended final model is the **tuned LightGBM**, with an F1-score of **0.8269** (95% CI [0.8086, 0.8448]), ROC-AUC of 0.9571 and PR-AUC of 0.8264. Pairwise McNemar tests confirm that its advantage over XGBoost and Random Forest is statistically significant after correction for multiple comparisons, and it trains in 8 seconds against 133 seconds for Random Forest.

The most consequential finding of DA3 is one that the aggregate metrics conceal. A per-time_step breakdown shows that LightGBM's recall is **0.861 over time steps 35–42** but **0.036 over time steps 43–49** — it detects 787 of 914 illicit transactions in the first period and only 6 of 169 in the second. The headline F1 of 0.83 is therefore not a stable estimate of future performance; it is an average over a period in which the model works well and a period in which it has effectively stopped working. This is consistent with the documented "dark market shutdown" event in the Elliptic dataset and is the single most important limitation of the work.

---

## 2. Methodology Recap

**DA1 — data and preprocessing.** The Elliptic dataset provides 203,769 Bitcoin transactions described by 165 anonymised numeric features plus a time step, of which only 46,564 carry a licit/illicit label. DA1 restricted the data to labelled transactions, split it by time rather than at random (time steps 1–34 for training, 35–49 for testing), and applied SMOTE to the training portion to address the severe class imbalance, producing a balanced training file of 39,648 rows (13,216 illicit, 26,432 licit) and an untouched test file of 16,670 rows retaining the real-world 6.5% illicit rate.

**DA2 — database, features, models and tuning.** The cleaned files were loaded into a SQLite database (`db/fraud_detection.db`) accessed from R through `DBI`/`RSQLite`, with the raw tables (`transactions_train`, `transactions_test`) preserved alongside reduced ones. Correlation-based feature selection removed one member of every feature pair correlated above 0.9, reducing 165 predictors to 95 and writing `train_selected`/`test_selected`. Eleven algorithms were then trained and compared on precision, recall, F1, ROC-AUC and PR-AUC. The top four by F1 were submitted for 3-fold cross-validated grid search: LightGBM and Random Forest tuned successfully, while XGBoost and Extra Trees failed inside `caret::train()` for reasons discussed in Section 6.

---

## 3. Final Performance Comparison and Statistical Analysis

### 3.1 All eleven models (baseline configurations)

| Rank | Model | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| 1 | Extra Trees (ranger) | 0.9511 | 0.7184 | 0.8185 | 0.9357 | 0.7261 |
| 2 | Random Forest | 0.9444 | 0.7221 | 0.8184 | 0.9351 | 0.7150 |
| 3 | XGBoost | 0.9178 | 0.7322 | 0.8146 | 0.9475 | 0.8185 |
| 4 | LightGBM | 0.9227 | 0.7276 | 0.8136 | 0.9562 | 0.8220 |
| 5 | AdaBoost | 0.9145 | 0.7211 | 0.8064 | NA | NA |
| 6 | GBM | 0.5370 | 0.7378 | 0.6215 | 0.8896 | 0.7650 |
| 7 | Decision Tree | 0.4738 | 0.7175 | 0.5707 | 0.8639 | 0.4047 |
| 8 | SVM | 0.4629 | 0.6399 | 0.5372 | 0.8821 | 0.5199 |
| 9 | KNN (k = 5) | 0.3414 | 0.7470 | 0.4686 | 0.8697 | 0.1319 |
| 10 | Logistic Regression | 0.1504 | 0.7156 | 0.2486 | 0.7174 | 0.0306 |
| 11 | Naive Bayes | 0.1262 | 0.7922 | 0.2178 | 0.7413 | 0.0263 |

The five ensemble methods occupy the top five positions. The separation from the remaining models is almost entirely in precision: recall lies between 0.64 and 0.79 for nearly every algorithm, while precision falls from 0.95 to 0.13. Logistic Regression and Naive Bayes flag so many licit transactions that their output would be unusable operationally, which indicates that the licit/illicit boundary in this feature space is not linear and does not satisfy conditional independence.

### 3.2 Final four models with bootstrap confidence intervals

The four strongest models were re-evaluated using their final objects — the tuned object for LightGBM and Random Forest, the original for XGBoost and Extra Trees. Confidence intervals come from 1,000 bootstrap resamples of the test set, with a shared set of resample indices so that every model is scored on an identical resampled test set in each replicate.

| Model | Object | F1 [95% CI] | ROC-AUC [95% CI] | PR-AUC [95% CI] | Train time |
|---|---|---|---|---|---|
| **LightGBM** | tuned | **0.8269** [0.8086, 0.8448] | **0.9571** [0.9516, 0.9624] | **0.8264** [0.8075, 0.8445] | 8.2 s |
| Extra Trees | original | 0.8181 [0.7993, 0.8356] | 0.9357 [0.9285, 0.9432] | 0.7973 [0.7762, 0.8179] | 19.4 s |
| XGBoost | original | 0.8146 [0.7955, 0.8321] | 0.9475 [0.9405, 0.9543] | 0.8194 [0.8002, 0.8383] | 2.4 s |
| Random Forest | tuned | 0.8085 [0.7895, 0.8262] | 0.9366 [0.9290, 0.9443] | 0.7976 [0.7768, 0.8186] | 133.4 s |

The confidence intervals overlap substantially. On F1 the four models span a range of roughly 0.018, while each individual interval is about 0.036 wide, so the interval widths alone do not establish a winner — which is why the paired tests below are necessary.

### 3.3 Pairwise McNemar tests

McNemar's test is the appropriate test here because every model is evaluated on exactly the same 16,670 transactions, making the correct/incorrect outcomes paired. The test uses only the discordant cases — transactions one model classifies correctly and the other does not — and asks whether the split between the two directions of disagreement is compatible with chance. p-values are Holm-adjusted across the six simultaneous comparisons.

| Comparison | Discordant (A only / B only) | χ² | p (raw) | p (Holm) | Result |
|---|---|---|---|---|---|
| LightGBM vs Random Forest | 62 (50 / 12) | 22.081 | 2.61e-06 | 1.57e-05 | **LightGBM better** |
| XGBoost vs LightGBM | 63 (17 / 46) | 12.444 | 0.00042 | 0.00193 | **LightGBM better** |
| Extra Trees vs Random Forest | 42 (33 / 9) | 12.595 | 0.00039 | 0.00193 | **Extra Trees better** |
| LightGBM vs Extra Trees | 58 (36 / 22) | 2.914 | 0.0878 | 0.2635 | not significant |
| XGBoost vs Extra Trees | 77 (31 / 46) | 2.545 | 0.1106 | 0.2635 | not significant |
| XGBoost vs Random Forest | 79 (44 / 35) | 0.810 | 0.3681 | 0.3681 | not significant |

Three of six comparisons are significant at α = 0.05 after correction. LightGBM is significantly better than both XGBoost and Random Forest; its advantage over Extra Trees is **not** statistically significant.

A paired bootstrap of the F1 difference, computed from the same shared replicates, gives a slightly different picture for one pair:

| Comparison | ΔF1 | 95% CI | Excludes zero |
|---|---|---|---|
| LightGBM − Random Forest | +0.0184 | [+0.0115, +0.0259] | yes |
| XGBoost − LightGBM | −0.0123 | [−0.0196, −0.0057] | yes |
| Extra Trees − Random Forest | +0.0096 | [+0.0040, +0.0151] | yes |
| LightGBM − Extra Trees | +0.0088 | [+0.0021, +0.0164] | yes |
| XGBoost − Extra Trees | −0.0035 | [−0.0117, +0.0046] | no |
| XGBoost − Random Forest | +0.0061 | [−0.0021, +0.0143] | no |

LightGBM vs Extra Trees is the one disagreement: McNemar finds no significant difference (p = 0.26) while the bootstrap F1 interval excludes zero. The two tests are answering different questions. McNemar assesses the **overall error rate**, which on this test set is dominated by the 15,587 licit transactions that both models classify correctly; F1 concerns only the positive class. LightGBM's advantage is concentrated in illicit-class decisions, so it registers in the F1 comparison and is diluted in the overall error rate. Where the two disagree, the F1-based comparison is the more relevant one for fraud detection, but the honest summary is that LightGBM and Extra Trees are close and the evidence separating them is mixed.

### 3.4 Recommended final model

**LightGBM (tuned) is the recommended model**, on the following grounds:

1. **Performance.** It has the highest F1 (0.8269), ROC-AUC (0.9571) and PR-AUC (0.8264) of the four, and its advantage is statistically significant against two of the other three under McNemar and against all three under the paired bootstrap of F1.
2. **False-positive cost.** At the 0.5 threshold its precision is 0.9497 — the highest of the four — producing only 42 false alarms out of 15,587 licit transactions. In a fraud-detection setting each false positive consumes analyst time and, if acted upon, freezes a legitimate user's funds. Extra Trees has comparable precision but lower recall; XGBoost catches the same number of illicit transactions (793) but raises 71 false alarms, a 69% higher investigative load for identical detection.
3. **PR-AUC under imbalance.** LightGBM and XGBoost hold a clear PR-AUC advantage (0.826 and 0.819) over the bagged ensembles (both ≈0.797). PR-AUC is the threshold-independent metric that matters here, so this indicates the boosting models rank illicit transactions better across all possible operating points, not just at 0.5.
4. **Training cost.** LightGBM trains in 8.2 seconds versus 133.4 seconds for the tuned Random Forest — a 16× difference that matters because concept drift (Section 5) implies this model would need frequent retraining in any real deployment.
5. **Interpretability.** This is the one dimension on which LightGBM does not win outright. All four are ensembles of hundreds of trees and none is directly interpretable; all four support the same gain- or impurity-based importance rankings, and none provides per-transaction explanations without an additional tool such as SHAP. Interpretability therefore does not discriminate between them, and is noted as a gap rather than a reason to prefer any one.

The recommendation is conditional on the caveat in Section 4.3: no model in this comparison, LightGBM included, performs acceptably in the final third of the test period.

---

## 4. Result Interpretation and Discussion

### 4.1 What the important features indicate

Across all four models the importance rankings are dominated by a single predictor, `feat_53`, with `feat_14` and `feat_41` consistently in the next tier. The four algorithms broadly agree on which predictors carry signal, though they differ in concentration: the boosting models place far more relative weight on the top two features than the bagged ensembles, which spread importance more evenly.

Interpretation must stop short of semantics. Elliptic's features are **anonymised by the dataset publisher** and no data dictionary is released, so it is not possible to state what `feat_53` measures. What *is* known is the structure: features 1–93 are local transaction properties (such as fee, volume and input/output counts) while 94–165 are aggregates of the same quantities over a transaction's one-hop neighbours. On that basis the top predictors for every model — `feat_53`, `feat_14`, `feat_41`, `feat_40`, `feat_22` — fall in the **local** block, with aggregated features (`feat_163`, `feat_138`, `feat_100`) appearing only further down. The defensible reading is that a transaction's own characteristics carry more of the discriminative signal than its immediate neighbourhood does, which is consistent with the Elliptic literature reporting that local features alone recover most of the achievable accuracy. Any claim about *which* transaction property matters would be speculation.

### 4.2 What the curve shapes say about the precision–recall trade-off

The ROC curves for all four models rise very steeply and sit close to the top-left corner, with AUCs of 0.94–0.96. Taken alone this suggests near-solved classification, and that impression is misleading: with 93.5% of transactions licit, the false-positive rate on the x-axis is divided by a large denominator, so even a substantial number of false alarms barely moves the curve.

The PR curves are more informative. They hold a precision above 0.95 while recall climbs to roughly 0.7, then fall away sharply — precision collapses as recall is pushed beyond about 0.75. The practical meaning is that roughly three-quarters of illicit transactions can be caught at very high precision, and the remaining quarter cannot be reached without a disproportionate rise in false alarms. The default 0.5 threshold sits just before the knee of that curve, which is a defensible operating point but a choice rather than a given: an investigative team willing to tolerate more false alarms could move the threshold down to trade precision for recall, and the PR curve is the correct instrument for making that decision. No threshold optimisation was performed in this project; 0.5 was used throughout.

### 4.3 Where the model fails

The per-time_step breakdown for LightGBM (`outputs/final_per_timestep_performance.csv`) shows a sharp and total degradation:

| time_step | Transactions | Illicit | Detected | Recall | Precision |
|---|---|---|---|---|---|
| 35 | 1,341 | 182 | 176 | 0.967 | 0.972 |
| 36 | 1,708 | 33 | 33 | 1.000 | 0.917 |
| 37 | 498 | 40 | 27 | 0.675 | 1.000 |
| 38 | 756 | 111 | 101 | 0.910 | 0.981 |
| 39 | 1,183 | 81 | 75 | 0.926 | 1.000 |
| 40 | 1,211 | 112 | 75 | 0.670 | 0.962 |
| 41 | 1,132 | 116 | 110 | 0.948 | 0.991 |
| 42 | 2,154 | 239 | 190 | 0.795 | 0.979 |
| **43** | 1,370 | 24 | **0** | **0.000** | — |
| 44 | 1,591 | 24 | 2 | 0.083 | 0.182 |
| 45 | 1,221 | 5 | 0 | 0.000 | — |
| 46 | 712 | 2 | 1 | 0.500 | 0.200 |
| 47 | 846 | 22 | 0 | 0.000 | — |
| 48 | 471 | 36 | 2 | 0.056 | 1.000 |
| 49 | 476 | 56 | 1 | 0.018 | 0.250 |

Aggregating: recall is **0.861** (787/914) over time steps 35–42 and **0.036** (6/169) over 43–49. At time step 43 detection stops almost completely and never recovers.

This pattern matches the event documented in the original Elliptic paper: a large dark marketplace was shut down partway through the observation window, after which the illicit transactions that remain in the data are of a different character from those the model was trained on. The model has not degraded gradually — it has encountered a population it was never shown. Two observations follow. First, a single aggregate F1 computed over a time-ordered test set can hide a complete failure in part of that period, which is an argument for always reporting performance over time when the split is temporal. Second, the fact that all four models were trained on data ending at time step 34 means none of them can be expected to handle this shift; this is a property of the problem, not a defect of LightGBM specifically.

---

## 5. Limitations

1. **Performance collapses after time step 43.** As shown above, recall falls from 0.861 to 0.036. The headline metrics in Section 3 are averages across both regimes and overstate what the model would achieve on the later period. This is the most serious limitation of the project.
2. **XGBoost and Extra Trees were never tuned.** Both failed inside `caret::train()` with `trainControl(classProbs = TRUE, summaryFunction = twoClassSummary)` and terminated before any grid point was scored. The failure lies in the caret configuration, not in the underlying packages, which train successfully outside caret in `scripts/04_train_models.R`. The cause was not isolated within the available time and the models are reported untuned. Two of the four final models are therefore evaluated at a possible disadvantage, and the comparison in Section 3 should be read with that asymmetry in mind.
3. **Reliance on SMOTE.** The training set is balanced with synthetic minority examples interpolated between real illicit transactions. Such points need not correspond to transactions that could actually occur, and the model may have learned partly from artefacts of the interpolation. No comparison against alternatives (class weighting, undersampling, or training on the raw imbalanced data) was run, so the contribution of SMOTE to the final results is unmeasured.
4. **The time-aware split reduces usable training data.** Restricting training to time steps 1–34 leaves 34 of 49 periods, and only labelled transactions are usable at all — 46,564 of 203,769. A random split would have yielded more training data and higher headline scores, but would have leaked future information into training and produced numbers that could not be trusted. The time-aware split is the methodologically correct choice; the cost is less data and lower apparent performance.
5. **A single fixed historical window.** The dataset covers roughly two weeks of Bitcoin activity from 2016. Fraud tactics, mixing services and the blockchain population itself have changed considerably since, so performance here is not evidence of performance on current data. The time-step collapse observed in Section 4.3 is a concrete demonstration of how quickly this kind of model can go stale.
6. **No threshold tuning and no cost model.** Every metric is computed at a 0.5 probability threshold, chosen by convention rather than by any analysis of the relative cost of a missed fraud versus a false alarm.
7. **Anonymised features limit interpretation.** Because Elliptic does not publish a data dictionary, the importance analysis can identify which features matter but not what they mean, which limits how far the results can be translated into operational guidance.
8. **Graph structure unused.** The Elliptic edge list (234,355 transaction-to-transaction edges) was not used at all; the models treat each transaction as an independent row.

---

## 6. Future Scope

1. **Use the graph structure.** The unused edge list supports features such as neighbour counts, degree, clustering coefficient, and distance to known illicit nodes, and supports graph neural network approaches (GCN, GAT) that the Elliptic literature reports as competitive. This is the largest untapped source of signal in the dataset, and neighbourhood context may be exactly what is needed to recognise the post-shutdown illicit transactions the current models miss entirely.
2. **Train time-aware and retrain continuously.** Given the collapse after time step 43, the natural next step is a rolling-origin evaluation — train on time steps 1–k, test on k+1, advance — which would measure how fast performance decays and how often retraining is required. A deployed system would need scheduled retraining and drift monitoring rather than a single fixed model.
3. **Cost-sensitive learning and threshold optimisation.** Missing an illicit transaction and freezing a legitimate one are not equally costly. Attaching explicit costs and selecting the threshold that minimises expected cost — or training with class weights and a cost-sensitive objective — would replace the arbitrary 0.5 cut-off with a justified operating point.
4. **Ensemble or stack the top models.** LightGBM, XGBoost and Extra Trees make partly independent errors (58–79 discordant transactions per pair in Section 3.3), which is the precondition for a useful ensemble. A soft-voting or stacked combination is a cheap next experiment.
5. **Resolve the caret tuning failure and complete the comparison.** Diagnosing the `trainControl`/`classProbs` interaction, or replacing caret with manual cross-validated grid loops of the kind already used successfully for LightGBM, would allow XGBoost and Extra Trees to be tuned and the four-way comparison to be made on equal footing.
6. **Add per-prediction explanations.** SHAP values would give analysts a reason for each flag, which matters both for investigator trust and for any regulatory requirement to justify an automated decision.

---

## 7. Conclusion

Eleven classification algorithms were implemented, compared and statistically evaluated for detecting illicit Bitcoin transactions in the Elliptic dataset. Tree ensembles clearly outperform linear and distance-based methods, and the tuned LightGBM is the recommended final model on the combined evidence of F1 (0.8269), PR-AUC (0.8264), precision (0.9497), statistically significant McNemar advantages over two of three competitors, and a training time an order of magnitude below the bagged ensembles. The statistical analysis also shows how narrow the top group is: the four best models lie within 0.018 F1 of one another and half the pairwise comparisons are not significant, so the ranking should not be over-read. The project's most important result is a negative one — performance is not stable across the test period, collapsing from 0.861 recall before time step 43 to 0.036 after it — and any future work on this problem should treat temporal robustness, not incremental gains in aggregate F1, as the primary objective.

---

## 8. Project Completion Status

**This project is at 100% project completion.**

| Phase | Deliverable | Status |
|---|---|---|
| DA1 | Data acquisition, preprocessing, time-aware split, SMOTE balancing | Complete |
| DA2 | SQLite database and retrieval layer | Complete |
| DA2 | Correlation-based feature selection (165 → 95) | Complete |
| DA2 | 11 ML algorithms implemented and evaluated | Complete |
| DA2 | Hyperparameter tuning | Partial — 2 of 4 tuned; XGBoost and Extra Trees documented as failed (Section 5.2) |
| DA2 | Static comparative visualisations | Complete |
| DA3 | Statistical significance testing and bootstrap confidence intervals | Complete |
| DA3 | Interactive R Shiny dashboard | Complete |
| DA3 | Result interpretation and discussion | Complete |
| DA3 | Limitations, future scope, conclusion | Complete |
| DA3 | Final report and code organisation | Complete |

All rubric items are delivered. The one incomplete element within them — tuning for two of the four top models — is reported as a known limitation rather than omitted.

---

## 9. Code Organisation and Reproducibility

The pipeline is a numbered sequence of R scripts, each of which reads from the database, writes its outputs to `outputs/`, and can be run independently once its predecessors have been run. Set the working directory to `project-folder/` and execute in order:

| Script | Purpose | Key outputs |
|---|---|---|
| `01_build_database.R` | Load the DA1 CSV files into SQLite; create `transactions_train` / `transactions_test` | `db/fraud_detection.db` |
| `02_fix_test_labels.R` | Recode test labels from numeric 1/2 to `illicit`/`licit` so both tables share one encoding | updated `transactions_test` |
| `03_feature_selection.R` | Drop one feature from every pair correlated above 0.9 (165 → 95) | `train_selected`, `test_selected` |
| `04_train_models.R` | Train and evaluate 11 algorithms; persist every fitted model | `model_comparison_results.csv`, `models/*.rds` |
| `05_hyperparameter_tuning.R` | 3-fold CV grid search on the top 4 by F1 | `tuning_results.csv`, `models/tuned/*.rds` |
| `06_visualizations.R` | ROC, PR, confusion-matrix and feature-importance figures at 300 DPI | `plots/*.png` |
| `07_final_statistical_analysis.R` | McNemar tests, bootstrap CIs, paired F1 differences, per-time_step breakdown | `final_statistical_comparison.csv` and three companion CSVs |
| `app.R` | Interactive Shiny dashboard — run with `shiny::runApp()` | caches to `outputs/dashboard_data.rds` |

Conventions applied throughout for reproducibility:

- Every script derives predictors as the set difference of the column names against `class`, `time_step` and `txId`, so a change to the selected feature set propagates automatically.
- Every script sets `factor(class, levels = c("licit", "illicit"))` immediately after retrieval, so `illicit` is the positive class in every metric, in every script.
- Model training, tuning and statistical tests are wrapped in `tryCatch`, so one failure is recorded and reported rather than halting the pipeline — this is how the DA2 tuning failures came to be documented instead of silently lost.
- `scripts/07` sets `set.seed(42)` before bootstrapping, so the confidence intervals reproduce exactly.
- Scripts 06, 07 and `app.R` share one model-selection rule: use the tuned `.rds` where it exists, otherwise the original, and print which was used.

```
project-folder/
├── app.R                  Shiny dashboard
├── data/                  elliptic_train_balanced.csv, elliptic_test.csv
├── db/                    fraud_detection.db (4 tables)
├── scripts/               01 … 07 (see table above)
├── outputs/
│   ├── model_comparison_results.csv, tuning_results.csv
│   ├── final_statistical_comparison.csv, final_pairwise_mcnemar.csv,
│   │   final_bootstrap_f1_differences.csv, final_per_timestep_performance.csv
│   ├── models/            11 baseline model objects
│   ├── models/tuned/      lightgbm.rds, random_forest.rds
│   └── plots/             4 comparative figures (300 DPI)
└── docs/                  DA2_progress_report.md, DA3_final_report.md
```

**Environment:** R 4.6.1. Packages: `DBI`, `RSQLite`, `caret`, `randomForest`, `ranger`, `xgboost`, `lightgbm`, `adabag`, `gbm`, `rpart`, `e1071`, `class`, `naivebayes`, `pROC`, `PRROC`, `MLmetrics`, `ggplot2`, `RColorBrewer`, `shiny`, `bslib`, `plotly`, `DT`.

---

## 10. Team Contribution

This project was completed by a two-person team; both members contributed across all three assessment phases.

*[Placeholder — replace with the actual split, for example:*
*Member 1 (Name, Reg. No.): DA1 preprocessing and SMOTE balancing; database construction and feature selection (scripts 01–03); model implementation (script 04); statistical analysis (script 07).*
*Member 2 (Name, Reg. No.): hyperparameter tuning (script 05); static and interactive visualisations (script 06, app.R); DA2 and DA3 report writing; viva preparation.]*
