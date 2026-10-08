# DA3 Viva Preparation
## Cryptocurrency Fraud Detection on the Elliptic Dataset

Fifteen questions an examiner is likely to ask, each with a short answer written to be spoken aloud in 15–25 seconds. Numbers are from the final results; keep the key ones in mind: **LightGBM F1 0.8269, precision 0.9497, recall 0.7322, ROC-AUC 0.9571, PR-AUC 0.8264**; test set **16,670 transactions, 1,083 illicit (6.5%)**; recall **0.861 before time step 43, 0.036 after**.

---

### 1. Why SQLite? Why not just use CSV files, or a full database server like MySQL?

SQLite is serverless and file-based, so the whole database travels with the project folder and needs no installation or credentials on a second machine — which matters for a two-person team. Unlike CSV it enforces a schema and lets us keep the original 165-feature tables alongside the reduced 95-feature ones without duplicating files. A server like MySQL would add setup and access management for no benefit at this data size, and we access it through R's standard `DBI` interface, so the backend could be swapped later without rewriting the analysis code.

### 2. Why these particular eleven algorithms?

They were chosen to span the main families of classifiers rather than to stack the deck: linear (logistic regression), probabilistic (naive Bayes), distance-based (KNN), margin-based (SVM), a single tree (decision tree), bagged ensembles (random forest, extra trees) and boosted ensembles (XGBoost, LightGBM, AdaBoost, GBM). The point was to establish which *family* suits this problem before optimising within one. The result was decisive — all five ensembles took the top five places, and the gap to everything else is large.

### 3. Why did you use SMOTE, and what are the risks?

The labelled training data is about 90% licit, and on imbalanced data most learners simply predict the majority class. SMOTE rebalances by interpolating new minority examples between existing illicit transactions, which gave us a 13,216 / 26,432 training split. The risk, which we state as a limitation, is that an interpolated point need not correspond to a transaction that could actually occur — so the model may partly learn artefacts of the interpolation. We also applied it only to the training set, never the test set, so the test data keeps its real 6.5% illicit rate.

### 4. Why a time-aware split instead of a random split?

Because a random split would leak the future into the past. Transactions in this dataset are ordered by time step and connected to each other, so training on a random sample would let the model see transactions from the same period — sometimes the same cluster — that it is later tested on, inflating the scores. We trained on time steps 1–34 and tested on 35–49, which is how the model would actually be used: fit on history, predict what comes next. It gives lower headline numbers than a random split, but they are numbers you can trust.

### 5. What does PR-AUC tell you that ROC-AUC does not?

With 93.5% of transactions licit, the false-positive rate in ROC space is divided by a very large denominator, so even a lot of false alarms barely moves the curve — which is why every model here, even naive Bayes, posts a respectable ROC-AUC. PR-AUC uses precision instead, so it only ever looks at the positive class and is not flattered by the true negatives. The contrast is stark: naive Bayes scores 0.741 ROC-AUC but 0.026 PR-AUC. The no-skill baseline for PR-AUC is the prevalence, 0.065, so LightGBM's 0.826 is a real result.

### 6. Why is accuracy not in your results table at all?

Because it would be actively misleading. A model that labels every transaction licit is 93.5% accurate on this test set and catches zero fraud. We ranked on F1 because it requires precision and recall to be acceptable at the same time, and we report PR-AUC as the threshold-independent companion.

### 7. Why did the XGBoost and Extra Trees tuning fail, and what does that mean for your comparison?

Both were routed through `caret::train()` with `classProbs = TRUE` and a custom `twoClassSummary` function, and both terminated before any grid point was scored. The failure is in that caret configuration, not in the packages themselves — XGBoost and ranger train fine outside caret in script 04, and LightGBM tuned successfully through a manual grid loop. We could not isolate the cause in the time available, so we report those two models untuned rather than substituting figures. It means two of the four finalists are evaluated at a possible disadvantage, and the comparison should be read with that asymmetry in mind.

### 8. Your Random Forest got *worse* after tuning. How do you explain that?

The caret grid was scored on cross-validated accuracy while we report test-set F1, and under class imbalance those two do not optimise to the same hyperparameters — the `mtry` that maximises accuracy is not the one that maximises F1. So the tuned model is genuinely slightly worse on our reporting metric, F1 0.8089 against 0.8184. It is a real result, not an error, and the fix for future work is to score the grid on an imbalance-aware metric in the first place.

### 9. What do the top features tell you about how fraud is detected here?

All four models agree that `feat_53` dominates, with `feat_14` and `feat_41` next. I have to be careful here: Elliptic anonymises its features and publishes no data dictionary, so I cannot say what `feat_53` measures. What I can say is structural — features 1 to 93 are local transaction properties and 94 to 165 are aggregates over neighbouring transactions, and every model's top predictors fall in the local block. So a transaction's own characteristics carry more signal than its immediate neighbourhood, which matches what the Elliptic literature reports.

### 10. You used McNemar's test. Why that test, and what did it show?

Because the comparison is paired: every model is scored on exactly the same 16,670 transactions, so an unpaired test would throw away that structure. McNemar looks only at the discordant cases — where one model is right and the other wrong — and asks whether the split between the two directions is compatible with chance. After Holm correction for six comparisons, three were significant: LightGBM beats XGBoost and Random Forest, Extra Trees beats Random Forest. LightGBM versus Extra Trees was *not* significant.

### 11. Your McNemar test and your bootstrap disagree on LightGBM versus Extra Trees. Which do you believe?

They answer different questions, so both are right. McNemar tests the overall error rate, which on this test set is dominated by the 15,587 licit transactions both models get right, so a positive-class advantage gets diluted. The bootstrap interval is on F1, which looks only at the illicit class, and it excludes zero. For fraud detection the F1 comparison is the more relevant one — but the honest summary is that those two models are close and the evidence separating them is mixed.

### 12. So which model do you recommend, and why that one?

Tuned LightGBM. It leads on all three headline metrics, and its precision of 0.9497 means only 42 false alarms across 15,587 legitimate transactions — XGBoost catches exactly the same 793 frauds but raises 71 false alarms, which is 69% more analyst time for identical detection. It also trains in 8 seconds against 133 for random forest, which matters because this model would need frequent retraining. On interpretability it has no advantage — all four are opaque ensembles — so I treat that as a gap rather than a tiebreaker.

### 13. Where does your model fail, and did you find that before or after you picked it?

We found it in DA3, by breaking recall down per time step — and it changed how we report the whole project. LightGBM catches 86% of illicit transactions up to time step 42, then almost nothing after: 6 out of 169 across time steps 43 to 49, with outright zero at several steps. That matches the dark-market shutdown documented in the original Elliptic paper — after it, the remaining illicit transactions look different from anything in training. The lesson is that a single aggregate F1 over a time-ordered test set can conceal a total failure in part of the period.

### 14. What would you do differently if you started again?

Three things. First, use the edge list — we never touched the 234,355 transaction-to-transaction edges, and neighbourhood structure is probably what is needed to catch the post-shutdown transactions we miss entirely. Second, evaluate with a rolling origin from the start, so the temporal collapse would have shown up in week one rather than at the end. Third, tune against an imbalance-aware metric and settle the caret problem early instead of letting it become a documented limitation.

### 15. How would this work in production, and what would break first?

It would be a scheduled batch scorer: pull new transactions, score them with the saved LightGBM model, and queue anything above the threshold for analyst review — inference is fast, about 0.07 seconds for 16,670 transactions, so throughput is not the constraint. What breaks first is drift, and our own results prove it: performance collapsed within roughly ten time steps of the training window ending. So a real deployment needs continuous monitoring of the illicit-detection rate, scheduled retraining, and an alert when the flag rate moves — plus a chosen threshold based on the actual cost of a missed fraud versus a false alarm, which we did not model.

---

## Likely follow-up probes

- **"Is 0.5 the right threshold?"** — No, it is a convention, not a decision. The PR curve holds precision above 0.95 up to about 0.7 recall and falls away after; 0.5 sits just before that knee. Choosing properly needs a cost for a missed fraud versus a false alarm, which we did not model.
- **"Why is AdaBoost missing ROC-AUC and PR-AUC?"** — The `adabag` prediction object did not return class probabilities in the expected format, so only the confusion-matrix metrics are valid for it. Its F1 of 0.8064 is unaffected.
- **"Why 0.9 as the correlation cut-off?"** — A conservative value that removes clear redundancy — mostly aggregated features that are near-duplicates of local ones — without discarding merely related predictors. It took 165 features to 95.
- **"Why do your PR-AUC numbers differ slightly between DA2 and DA3?"** — DA2's script 04 used `MLmetrics::PRAUC` and the later scripts use `PRROC`, which interpolate differently; the gap is largest for the bagged ensembles because their vote-proportion probabilities are heavily tied. The ranking is unaffected, and DA3 standardises on PRROC.
- **"How do you know your bootstrap intervals are comparable across models?"** — All four models are scored on the same resampled index set in each of the 1,000 replicates, with `set.seed(42)`, so the differences are paired rather than independent draws.
