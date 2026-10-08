# Cryptocurrency Fraud Detection on the Elliptic Dataset
## DA2 Progress Report — 75% Project Completion

**Course:** Programming for Data Science
**Date:** 15 September 2026
**Team size:** 2

---

## 1. Project Recap

This project builds and evaluates machine learning models that classify Bitcoin transactions in the Elliptic dataset as licit or illicit. DA1 established the preprocessing pipeline: loading the raw Elliptic feature and class files, resolving the labelled/unlabelled split, handling the severe class imbalance in the training data through balancing, and producing the cleaned `elliptic_train_balanced.csv` and `elliptic_test.csv` files. DA2 extends that pipeline into a persistent SQLite database, a feature-selection stage, a comparative study of eleven classification algorithms, hyperparameter tuning of the strongest models, and a set of comparative visualisations.

---

## 2. Database Connectivity and Data Retrieval

**What was built.** A SQLite database at `db/fraud_detection.db` holds all modelling data, created and queried from R through the `DBI` and `RSQLite` packages (`scripts/01_build_database.R`). It contains four tables:

| Table | Contents |
|---|---|
| `transactions_train` | Full balanced training set as produced by the DA1 pipeline (165 features) |
| `transactions_test` | Full held-out test set (165 features) |
| `train_selected` | Training set after correlation-based feature selection (95 features) |
| `test_selected` | Test set after the same selection, with identical column ordering |

The `class` column stores the labels as the strings `"illicit"` and `"licit"`. Every downstream script converts this to a factor with levels `c("licit", "illicit")` immediately after retrieval, so that `"illicit"` is consistently treated as the positive (minority) class by all metric calculations.

**Why SQLite.** SQLite was chosen over a client–server database (MySQL, PostgreSQL) and over flat CSV files for three reasons. First, it is serverless and file-based, so the entire database travels with the project directory and requires no installation or credentials on a second team member's machine. Second, it is fully supported in R through `RSQLite`, which implements the standard `DBI` interface, so the retrieval code is portable if the backend is ever changed. Third, unlike CSV files it enforces a schema and column types, and it allows the intermediate `*_selected` tables to be stored alongside the originals without duplicating raw files — the original 165-feature tables remain available for comparison, which would be wasteful to maintain as separate CSVs.

**How retrieval works.** Each analysis script opens a connection with `DBI::dbConnect(RSQLite::SQLite(), dbname = "db/fraud_detection.db")`, reads the required tables with `DBI::dbReadTable()`, and closes the connection with `DBI::dbDisconnect()` before any model training begins. Closing early avoids holding a file lock during long-running training jobs. Predictor columns are derived as the set difference between all column names and the non-predictor columns (`class`, `time_step`, `txId`), so the scripts adapt automatically if the selected feature set changes. Each script also verifies that all training features are present in the test table before proceeding.

---

## 3. Feature Engineering and Feature Selection

The Elliptic feature set contains 165 numeric predictors (local transaction features plus aggregated neighbourhood features). Many of these are near-duplicates of one another — the aggregated features are, by construction, functions of the local features of neighbouring nodes — which produces substantial multicollinearity.

**Methodology** (`scripts/03_feature_selection.R`): a Pearson correlation matrix was computed across all 165 numeric predictors on the training data. For every pair of features with an absolute correlation above the threshold of **0.9**, one member of the pair was removed, retaining the other. This reduced the predictor set from **165 to 95 features**, a 42% reduction.

**Why this was done.** Highly correlated predictors carry largely redundant information. Removing them reduces multicollinearity, which destabilises coefficient estimates in linear models such as logistic regression; it reduces training time and memory for the tree ensembles; and it makes tree-based feature importance easier to interpret, because importance is no longer arbitrarily split between two near-identical columns. A threshold of 0.9 was chosen as a conservative value that removes clear redundancy without discarding features that are merely related. The reduced tables were written back to the database as `train_selected` and `test_selected` so that all models train on an identical feature set.

---

## 4. Model Implementation and Comparative Performance

### 4.1 Models trained

Eleven classification algorithms were trained on `train_selected` and evaluated on `test_selected` (`scripts/04_train_models.R`):

1. Extra Trees (`ranger`)
2. Random Forest (`randomForest`)
3. XGBoost (`xgboost`)
4. LightGBM (`lightgbm`)
5. AdaBoost (`adabag`)
6. Gradient Boosting Machine (`gbm`)
7. Decision Tree (`rpart`)
8. Support Vector Machine (`e1071`)
9. K-Nearest Neighbours, k = 5 (`class`)
10. Logistic Regression (`stats::glm`)
11. Naive Bayes (`naivebayes`)

Each model is trained inside a `tryCatch` block, so a failure in one algorithm does not halt the remaining ones, and every fitted model object is persisted to `outputs/models/` as an `.rds` file for reuse by the tuning and visualisation stages.

### 4.2 Evaluation approach

All models were evaluated on the held-out test set using precision, recall, F1-score, ROC-AUC and PR-AUC, with `"illicit"` as the positive class. The test set contains **16,670 transactions, of which 1,083 (6.5%) are illicit** and 15,587 are licit.

**Why accuracy is not the headline metric here.** With 93.5% of test transactions labelled licit, a trivial classifier that predicts "licit" for every transaction achieves 93.5% accuracy while detecting no fraud at all. Accuracy is therefore uninformative under this degree of class imbalance. The metrics used instead are:

- **Precision** — of the transactions flagged as illicit, what fraction really are. Low precision means investigators waste effort on false alarms.
- **Recall** — of the genuinely illicit transactions, what fraction the model catches. Low recall means fraud goes undetected.
- **F1-score** — the harmonic mean of the two, used as the primary ranking metric because fraud detection requires both to be acceptable simultaneously.
- **PR-AUC** — the area under the precision–recall curve. Unlike ROC-AUC, it is computed entirely from the positive class and does not benefit from the large pool of true negatives. Its no-skill baseline equals the positive-class prevalence (0.065 here), so a PR-AUC of 0.82 represents a genuine improvement over chance, whereas ROC-AUC values are optimistically high for every model under imbalance — even Naive Bayes reaches 0.741 ROC-AUC while managing only 0.027 PR-AUC.
- **ROC-AUC** — retained for completeness and for threshold-independent comparison against published results on this dataset.

### 4.3 Results (all 11 models, sorted by F1-score)

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

Full results are stored in `outputs/model_comparison_results.csv`.

**Observations.** The four tree ensembles (Extra Trees, Random Forest, XGBoost, LightGBM) form a clear top group with F1-scores between 0.814 and 0.819, followed closely by AdaBoost. ROC-AUC and PR-AUC were not recorded for AdaBoost because the `adabag` prediction object did not return class probabilities in the expected format; its confusion-based metrics are unaffected. The main separation between the top group and the rest is in precision, not recall: recall sits between 0.64 and 0.79 for almost every model including the weakest, but precision collapses from ~0.95 to ~0.13. Logistic Regression and Naive Bayes flag a large fraction of licit transactions as illicit, which is consistent with a linear or independence-assuming decision boundary being unable to separate the classes in this feature space. Notably, the two best models by F1 (Extra Trees, Random Forest) are not the two best by PR-AUC (LightGBM, XGBoost) — the boosting models rank illicit transactions better overall, while the bagged ensembles are better calibrated at the default 0.5 threshold.

---

## 5. Hyperparameter Tuning and Model Optimisation

### 5.1 Methodology

The top four models by F1-score were selected for tuning (`scripts/05_hyperparameter_tuning.R`). The intended approach was grid search with **3-fold cross-validation** on the training set, using `caret::train()` with `trainControl(method = "cv", number = 3, classProbs = TRUE)` for the caret-supported methods, and a manual grid loop for LightGBM, which has no reliable native caret method. Three folds rather than five or ten were used to keep total runtime feasible on the available hardware given the size of the grids.

Search spaces:

| Model | Grid |
|---|---|
| XGBoost | `nrounds` {100, 300} × `max_depth` {3, 5, 7} × `eta` {0.01, 0.05, 0.1}, with `subsample` 0.8, `colsample_bytree` 0.8 |
| LightGBM | `num_leaves` {15, 31, 63} × `learning_rate` {0.01, 0.05, 0.1} × `max_depth` {5, 7, −1}, 300 rounds |
| Extra Trees | `mtry` {5, 10, 15, 20} × `min.node.size` {1, 5, 10}, 500 trees |
| Random Forest | `mtry` {5, 10, 15, 20, 25}, 500 trees |

### 5.2 Results

| Model | Original F1 | Tuned F1 | Change | Tuned precision | Tuned recall | Tuned ROC-AUC | Tuned PR-AUC | Best hyperparameters |
|---|---|---|---|---|---|---|---|---|
| LightGBM | 0.8136 | **0.8269** | +0.0133 | 0.9497 | 0.7322 | 0.9571 | 0.8264 | `num_leaves = 63`, `learning_rate = 0.05`, `max_depth = -1` |
| Random Forest | 0.8184 | 0.8089 | −0.0095 | 0.9210 | 0.7211 | 0.9366 | 0.7976 | `mtry = 25` |
| XGBoost | 0.8146 | — | — | — | — | — | — | Tuning did not complete |
| Extra Trees | 0.8185 | — | — | — | — | — | — | Tuning did not complete |

**LightGBM** improved from 0.8136 to 0.8269 F1, driven mainly by an increase in precision (0.9227 → 0.9497) at effectively unchanged recall. Its tuned configuration is the best single result obtained in DA2 on both F1 and PR-AUC.

**Random Forest** decreased slightly, from 0.8184 to 0.8089 F1. This is a legitimate result rather than an error: the tuning grid was scored on cross-validated accuracy, whereas the reported figure is test-set F1, and under class imbalance the `mtry` value that maximises accuracy in cross-validation is not necessarily the one that maximises F1 on held-out data. The untuned Random Forest is therefore retained as the better configuration of that model, and the mismatch between the tuning criterion and the reporting criterion is noted as something to correct in DA3 by scoring the grid on an imbalance-aware metric.

### 5.3 Known limitation: XGBoost and Extra Trees tuning did not complete

Tuning for XGBoost and Extra Trees did not produce results. Both were routed through `caret::train()` with `trainControl(classProbs = TRUE, summaryFunction = twoClassSummary)` and `metric = "ROC"`, and both terminated with a caret error before any grid point was scored; the error was caught by the surrounding `tryCatch`, so the script continued and recorded the failure in `outputs/tuning_results.csv` as `"Tuning failed: Stopping"`. The failure is in the `trainControl`/`classProbs` configuration rather than in the underlying `xgboost` and `ranger` packages, both of which train successfully outside caret in `scripts/04_train_models.R`. The specific cause was not isolated within the time available for DA2, and no results are reported for these two models rather than substituting untuned figures.

The practical effect is that two of the four intended tuning runs are incomplete. The untuned XGBoost and Extra Trees models remain available and are used unchanged in the comparative analysis and visualisations. The plan for DA3 is to diagnose the caret configuration — the likely candidates being the interaction between `classProbs`, the custom `summaryFunction` and the factor level naming — and, if it cannot be resolved, to replace caret with manual cross-validated grid loops of the kind already used successfully for LightGBM.

---

## 6. Comparative Visualisations

`scripts/06_visualizations.R` loads the four top models — using the tuned object where one exists (LightGBM, Random Forest) and the original baseline object otherwise (XGBoost, Extra Trees), printing which version was used for each — and writes four figures to `outputs/plots/` at 300 DPI. A single colour per model, taken from the RColorBrewer "Set2" palette, is used consistently across all four figures.

| File | Contents |
|---|---|
| `roc_curves_comparison.png` | ROC curves for all four models overlaid, AUC per model in the legend |
| `pr_curves_comparison.png` | Precision–recall curves overlaid, PR-AUC per model in the legend, with the no-skill prevalence baseline marked |
| `confusion_matrices_grid.png` | 2×2 grid of confusion-matrix heatmaps, each cell labelled with count and percentage of the actual class |
| `feature_importance_grid.png` | 2×2 grid of the top 15 features per model as horizontal bar charts, importance rescaled to 0–100 within each model |

**What the visualisations show.** The feature importance panels are dominated by a single predictor, `feat_53`, which is the top-ranked feature for all four models by a wide margin, with `feat_14` and `feat_41` consistently in the next tier — the four algorithms largely agree on which features carry the signal, despite differing in how sharply importance is concentrated (the boosting models place far more weight on the top two features than the bagged ensembles do). The ROC and PR curves separate the models more clearly than the F1 table does: LightGBM and XGBoost achieve the highest ROC-AUC (0.957 and 0.947) and PR-AUC (0.826 and 0.819), while Extra Trees and Random Forest sit around 0.936 ROC-AUC and 0.80 PR-AUC, indicating that the boosting models rank illicit transactions better across all thresholds even though their F1 at the default 0.5 threshold is comparable. The confusion matrices show that the dominant error mode is shared by all four models: each correctly identifies 99.5–99.7% of licit transactions but misses roughly 27–28% of illicit ones (290 of 1,083 for LightGBM and XGBoost, 302–305 for the bagged ensembles), confirming that recall, not precision, is the binding constraint at the current threshold.

*Note on PR-AUC values: the figures quoted above are computed by `PRROC::pr.curve()`, whereas the PR-AUC column in Section 4.3 was computed by `MLmetrics::PRAUC()` in `scripts/04_train_models.R`. The two use different interpolation rules, so the values differ slightly for the boosting models (e.g. XGBoost 0.8185 vs 0.8194) and more noticeably for the bagged ensembles, whose probability outputs are heavily tied (Extra Trees 0.7261 vs 0.7973). The model ordering is unaffected. DA3 will standardise on a single PR-AUC implementation across all scripts.*

*Note: the baseline Extra Trees model was fitted without `importance = "impurity"` and therefore stores no importance vector; the script refits an equivalent `ranger` model on `train_selected` solely to obtain impurity importances for that panel. This refit is used for the visualisation only and contributes to no reported metric.*

---

## 7. Current Completion Status

**The project stands at 75% completion**, matching the DA2 requirement.

**Complete (DA2):**
- Feature engineering and correlation-based feature selection (165 → 95 features)
- SQLite database connectivity and data retrieval in R via `DBI`/`RSQLite`
- Implementation and evaluation of 11 ML algorithms
- Hyperparameter tuning: 2 of the top 4 models tuned successfully (LightGBM improved, Random Forest slightly decreased); 2 incomplete with the technical cause documented in Section 5.3
- Comparative performance analysis across precision, recall, F1, ROC-AUC and PR-AUC
- Comparative visualisations: ROC curves, PR curves, confusion matrix grid, feature importance grid

**Remaining (DA3):**
- Formal statistical comparison of model performance (significance testing of the differences between the top models rather than raw metric ordering)
- Interactive dashboard for exploring predictions and model comparisons
- Interpretation of results, including what the dominant features correspond to in the Elliptic feature schema
- Limitations and future scope, including the threshold-selection question raised by the confusion matrices and the unresolved caret tuning issue
- Resolution or documented workaround for the XGBoost and Extra Trees tuning failure
- Final consolidated project report
- Viva preparation and demonstration

---

## 8. Team Contribution

This work was carried out by a two-person team. Both members contributed to the DA2 deliverables described above.

*[Placeholder — task split to be filled in: e.g. Member 1 (Name, Reg. No.) — database construction, feature selection, model implementation; Member 2 (Name, Reg. No.) — hyperparameter tuning, visualisations, documentation.]*

---

## Appendix: Repository Structure

```
project-folder/
├── data/                  elliptic_train_balanced.csv, elliptic_test.csv
├── db/                    fraud_detection.db (4 tables)
├── scripts/
│   ├── 01_build_database.R
│   ├── 02_fix_test_labels.R
│   ├── 03_feature_selection.R
│   ├── 04_train_models.R
│   ├── 05_hyperparameter_tuning.R
│   └── 06_visualizations.R
├── outputs/
│   ├── model_comparison_results.csv
│   ├── tuning_results.csv
│   ├── models/            11 baseline model objects (.rds)
│   ├── models/tuned/      lightgbm.rds, random_forest.rds
│   └── plots/             4 comparative figures (300 DPI)
└── docs/                  DA2_progress_report.md
```
