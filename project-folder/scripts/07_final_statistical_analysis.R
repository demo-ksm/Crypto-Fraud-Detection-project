# scripts/07_final_statistical_analysis.R
# Purpose: Final statistical comparison of the four best models from DA2.
#
#   1. Load test_selected and the final model set (tuned object where one
#      exists, original baseline object otherwise).
#   2. Pairwise McNemar tests on the paired predictions of every model pair,
#      to establish whether the observed differences are distinguishable from
#      chance on this test set.
#   3. Bootstrap 95% confidence intervals (1000 resamples) for F1, ROC-AUC and
#      PR-AUC, using a shared set of resample indices so that the intervals
#      (and the paired differences) are comparable across models.
#   4. Per-time_step recall for the best model, to show where it degrades.
#
# Outputs:
#   outputs/final_statistical_comparison.csv  - main summary table
#   outputs/final_pairwise_mcnemar.csv        - full pairwise test detail
#   outputs/final_per_timestep_performance.csv- per-time_step breakdown
#
# Every statistical test is wrapped in tryCatch so that one failure does not
# stop the remaining analysis.

script_start <- Sys.time()
set.seed(42)

N_BOOTSTRAP <- 1000
ALPHA <- 0.05

# ---------------------------------------------------------------------------
# 1. Packages
# ---------------------------------------------------------------------------

safe_require <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message(sprintf("Package '%s' is unavailable; dependent output will be skipped.", pkg))
    return(FALSE)
  }
  TRUE
}

dbi_available          <- safe_require("DBI")
rsqlite_available      <- safe_require("RSQLite")
proc_available         <- safe_require("pROC")
prroc_available        <- safe_require("PRROC")
xgboost_available      <- safe_require("xgboost")
lightgbm_available     <- safe_require("lightgbm")
ranger_available       <- safe_require("ranger")
randomforest_available <- safe_require("randomForest")
caret_available        <- safe_require("caret")

if (!dbi_available || !rsqlite_available) {
  stop("DBI and RSQLite are required to read the test data.")
}

# ---------------------------------------------------------------------------
# 2. Test data
# ---------------------------------------------------------------------------

db_path <- file.path("db", "fraud_detection.db")
if (!file.exists(db_path)) stop("Database not found at: ", db_path)

conn <- DBI::dbConnect(RSQLite::SQLite(), dbname = db_path)
test_df <- DBI::dbReadTable(conn, "test_selected")
DBI::dbDisconnect(conn)

test_df$class <- factor(test_df$class, levels = c("licit", "illicit"))

feature_cols <- setdiff(names(test_df), c("time_step", "class", "txId"))
X_test <- as.data.frame(lapply(test_df[, feature_cols, drop = FALSE],
                               function(x) as.numeric(as.character(x))))
names(X_test) <- feature_cols
y_test <- test_df$class
y_binary <- as.numeric(y_test == "illicit")
time_step <- test_df$time_step

message(sprintf("Test set: %d transactions, %d illicit (%.1f%%), time_step %d-%d.",
                nrow(X_test), sum(y_binary), 100 * mean(y_binary),
                min(time_step), max(time_step)))

if (!dir.exists("outputs")) dir.create("outputs", recursive = TRUE)

# ---------------------------------------------------------------------------
# 3. Final model set (mirrors the selection logic of script 06)
# ---------------------------------------------------------------------------

model_spec <- list(
  list(label = "XGBoost",
       tuned = file.path("outputs", "models", "tuned", "xgboost.rds"),
       original = file.path("outputs", "models", "xgboost.rds")),
  list(label = "LightGBM",
       tuned = file.path("outputs", "models", "tuned", "lightgbm.rds"),
       original = file.path("outputs", "models", "lightgbm.rds")),
  list(label = "Extra Trees",
       tuned = file.path("outputs", "models", "tuned", "extra_trees_ranger.rds"),
       original = file.path("outputs", "models", "extra_trees_(ranger).rds")),
  list(label = "Random Forest",
       tuned = file.path("outputs", "models", "tuned", "random_forest.rds"),
       original = file.path("outputs", "models", "random_forest.rds"))
)

predict_illicit_probability <- function(model, newdata) {
  if (inherits(model, "xgb.Booster")) {
    cols <- model$feature_names
    if (is.null(cols)) cols <- names(newdata)
    return(as.numeric(predict(model, as.matrix(newdata[, cols, drop = FALSE]))))
  }
  if (inherits(model, "lgb.Booster")) {
    return(as.numeric(predict(model, as.matrix(newdata))))
  }
  if (inherits(model, "ranger")) {
    predictions <- predict(model, data = newdata)$predictions
    if (is.null(dim(predictions))) return(as.numeric(predictions))
    if ("illicit" %in% colnames(predictions)) return(as.numeric(predictions[, "illicit"]))
    return(as.numeric(predictions[, ncol(predictions)]))
  }
  if (inherits(model, "train") || inherits(model, "randomForest")) {
    probabilities <- predict(model, newdata = newdata, type = "prob")
    if ("illicit" %in% colnames(probabilities)) return(as.numeric(probabilities[, "illicit"]))
    return(as.numeric(probabilities[, ncol(probabilities)]))
  }
  stop("Unsupported model class: ", paste(class(model), collapse = "/"))
}

load_model <- function(spec) {
  attempts <- list()
  if (file.exists(spec$tuned)) {
    attempts[[length(attempts) + 1]] <- list(version = "tuned", path = spec$tuned)
  }
  if (file.exists(spec$original)) {
    attempts[[length(attempts) + 1]] <- list(version = "original", path = spec$original)
  }
  for (attempt in attempts) {
    loaded <- tryCatch({
      model <- readRDS(attempt$path)
      probabilities <- predict_illicit_probability(model, X_test)
      if (length(probabilities) != nrow(X_test) || all(is.na(probabilities))) {
        stop("Predictions were empty or all NA.")
      }
      list(probabilities = probabilities, version = attempt$version, path = attempt$path)
    }, error = function(e) {
      warning(sprintf("Could not use %s model for %s: %s",
                      attempt$version, spec$label, conditionMessage(e)))
      NULL
    })
    if (!is.null(loaded)) return(loaded)
  }
  NULL
}

models <- list()
for (spec in model_spec) {
  loaded <- load_model(spec)
  if (is.null(loaded)) {
    warning(sprintf("No usable model object for %s; excluded from the analysis.", spec$label))
    next
  }
  loaded$predicted_positive <- as.numeric(loaded$probabilities >= 0.5)
  loaded$correct <- as.numeric(loaded$predicted_positive == y_binary)
  models[[spec$label]] <- loaded
  message(sprintf("%-14s -> %-8s version  (%s)", spec$label, loaded$version, loaded$path))
}

if (length(models) < 2) stop("At least two models are required for a pairwise comparison.")
model_labels <- names(models)

# ---------------------------------------------------------------------------
# 4. Metric helpers
# ---------------------------------------------------------------------------

# F1 for the positive ("illicit") class at a 0.5 decision threshold.
compute_f1 <- function(truth, predicted_positive) {
  tp <- sum(predicted_positive == 1 & truth == 1)
  fp <- sum(predicted_positive == 1 & truth == 0)
  fn <- sum(predicted_positive == 0 & truth == 1)
  if ((tp + fp) == 0 || (tp + fn) == 0) return(NA_real_)
  precision <- tp / (tp + fp)
  recall <- tp / (tp + fn)
  if ((precision + recall) == 0) return(NA_real_)
  2 * precision * recall / (precision + recall)
}

compute_roc_auc <- function(truth, probabilities) {
  if (!proc_available || length(unique(truth)) < 2) return(NA_real_)
  as.numeric(pROC::auc(pROC::roc(truth, probabilities, quiet = TRUE,
                                 direction = "<", levels = c(0, 1))))
}

compute_pr_auc <- function(truth, probabilities) {
  if (!prroc_available || length(unique(truth)) < 2) return(NA_real_)
  PRROC::pr.curve(scores.class0 = probabilities[truth == 1],
                  scores.class1 = probabilities[truth == 0],
                  curve = FALSE)$auc.integral
}

compute_precision <- function(truth, predicted_positive) {
  tp <- sum(predicted_positive == 1 & truth == 1)
  fp <- sum(predicted_positive == 1 & truth == 0)
  if ((tp + fp) == 0) return(NA_real_)
  tp / (tp + fp)
}

compute_recall <- function(truth, predicted_positive) {
  tp <- sum(predicted_positive == 1 & truth == 1)
  fn <- sum(predicted_positive == 0 & truth == 1)
  if ((tp + fn) == 0) return(NA_real_)
  tp / (tp + fn)
}

# ---------------------------------------------------------------------------
# 5. Point estimates
# ---------------------------------------------------------------------------

point_estimates <- list()
for (label in model_labels) {
  m <- models[[label]]
  point_estimates[[label]] <- list(
    precision = compute_precision(y_binary, m$predicted_positive),
    recall    = compute_recall(y_binary, m$predicted_positive),
    f1        = compute_f1(y_binary, m$predicted_positive),
    roc_auc   = compute_roc_auc(y_binary, m$probabilities),
    pr_auc    = compute_pr_auc(y_binary, m$probabilities)
  )
}

message("\n---------------- POINT ESTIMATES (test set) ----------------")
message(sprintf("%-14s %9s %9s %9s %9s %9s", "Model", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"))
for (label in model_labels) {
  p <- point_estimates[[label]]
  message(sprintf("%-14s %9.4f %9.4f %9.4f %9.4f %9.4f",
                  label, p$precision, p$recall, p$f1, p$roc_auc, p$pr_auc))
}

# ---------------------------------------------------------------------------
# 6. Pairwise McNemar tests
# ---------------------------------------------------------------------------
#
# McNemar's test is the appropriate paired test here: every model is evaluated
# on exactly the same transactions, so the correct/incorrect outcomes are
# paired observations. The test uses only the discordant pairs -- cases one
# model classifies correctly and the other does not -- and asks whether the
# split between the two kinds of disagreement is compatible with chance.

pairwise_results <- NULL
pairwise_results <- tryCatch({
  pairs <- utils::combn(model_labels, 2, simplify = FALSE)
  rows <- list()

  for (pair in pairs) {
    a <- pair[1]
    b <- pair[2]
    correct_a <- models[[a]]$correct
    correct_b <- models[[b]]$correct

    # Discordant counts: b_only = A wrong & B right, a_only = A right & B wrong.
    both_right <- sum(correct_a == 1 & correct_b == 1)
    a_only     <- sum(correct_a == 1 & correct_b == 0)
    b_only     <- sum(correct_a == 0 & correct_b == 1)
    both_wrong <- sum(correct_a == 0 & correct_b == 0)

    contingency <- matrix(c(both_right, a_only, b_only, both_wrong), nrow = 2, byrow = TRUE,
                          dimnames = list(paste(a, c("correct", "incorrect")),
                                          paste(b, c("correct", "incorrect"))))

    test_result <- tryCatch(stats::mcnemar.test(contingency, correct = TRUE),
                            error = function(e) {
                              warning(sprintf("McNemar test failed for %s vs %s: %s",
                                              a, b, conditionMessage(e)))
                              NULL
                            })

    rows[[length(rows) + 1]] <- data.frame(
      model_a = a,
      model_b = b,
      both_correct = both_right,
      only_a_correct = a_only,
      only_b_correct = b_only,
      both_incorrect = both_wrong,
      discordant_pairs = a_only + b_only,
      mcnemar_chi_squared = if (is.null(test_result)) NA_real_ else as.numeric(test_result$statistic),
      p_value = if (is.null(test_result)) NA_real_ else as.numeric(test_result$p.value),
      accuracy_a = mean(correct_a),
      accuracy_b = mean(correct_b),
      f1_a = point_estimates[[a]]$f1,
      f1_b = point_estimates[[b]]$f1,
      stringsAsFactors = FALSE
    )
  }

  out <- do.call(rbind, rows)
  # Holm correction across the six simultaneous comparisons.
  out$p_value_holm <- stats::p.adjust(out$p_value, method = "holm")
  out$significant_raw <- !is.na(out$p_value) & out$p_value < ALPHA
  out$significant_holm <- !is.na(out$p_value_holm) & out$p_value_holm < ALPHA
  out$better_model <- ifelse(is.na(out$p_value) | !out$significant_holm, "no significant difference",
                             ifelse(out$only_a_correct > out$only_b_correct, out$model_a, out$model_b))
  out
}, error = function(e) {
  warning("Pairwise McNemar analysis failed: ", conditionMessage(e))
  NULL
})

if (!is.null(pairwise_results)) {
  message("\n---------------- PAIRWISE McNEMAR TESTS ----------------")
  message("H0: the two models have the same error rate on this test set.")
  for (i in seq_len(nrow(pairwise_results))) {
    r <- pairwise_results[i, ]
    message(sprintf("%-14s vs %-14s  discordant %4d (%4d / %4d)  chi2 = %7.3f  p = %.4g  p(Holm) = %.4g  %s",
                    r$model_a, r$model_b, r$discordant_pairs,
                    r$only_a_correct, r$only_b_correct,
                    r$mcnemar_chi_squared, r$p_value, r$p_value_holm,
                    if (r$significant_holm) paste("SIGNIFICANT ->", r$better_model) else "not significant"))
  }
  write.csv(pairwise_results, file.path("outputs", "final_pairwise_mcnemar.csv"), row.names = FALSE)
  message("Saved: outputs/final_pairwise_mcnemar.csv")
}

# ---------------------------------------------------------------------------
# 7. Bootstrap confidence intervals
# ---------------------------------------------------------------------------
#
# A single matrix of resample indices is shared by all models, so each bootstrap
# replicate scores every model on an identical resampled test set. This makes
# the intervals directly comparable and allows paired differences to be read
# off the same replicates.

bootstrap_results <- NULL
bootstrap_differences <- NULL

bootstrap_results <- tryCatch({
  n <- length(y_binary)
  message(sprintf("\nRunning %d bootstrap resamples for %d models ...", N_BOOTSTRAP, length(model_labels)))

  boot_f1  <- matrix(NA_real_, nrow = N_BOOTSTRAP, ncol = length(model_labels),
                     dimnames = list(NULL, model_labels))
  boot_roc <- boot_f1
  boot_pr  <- boot_f1

  for (b in seq_len(N_BOOTSTRAP)) {
    idx <- sample.int(n, size = n, replace = TRUE)
    truth_b <- y_binary[idx]
    # A resample containing only one class cannot be scored; skip it.
    if (length(unique(truth_b)) < 2) next

    for (label in model_labels) {
      probabilities_b <- models[[label]]$probabilities[idx]
      predicted_b <- models[[label]]$predicted_positive[idx]
      boot_f1[b, label]  <- compute_f1(truth_b, predicted_b)
      boot_roc[b, label] <- compute_roc_auc(truth_b, probabilities_b)
      boot_pr[b, label]  <- compute_pr_auc(truth_b, probabilities_b)
    }

    if (b %% 100 == 0) message(sprintf("  ... %d / %d resamples", b, N_BOOTSTRAP))
  }

  percentile_ci <- function(values) {
    values <- values[is.finite(values)]
    if (length(values) < 2) return(c(NA_real_, NA_real_))
    stats::quantile(values, probs = c(ALPHA / 2, 1 - ALPHA / 2), names = FALSE)
  }

  rows <- list()
  for (label in model_labels) {
    for (metric in c("f1_score", "roc_auc", "pr_auc")) {
      values <- switch(metric,
                       f1_score = boot_f1[, label],
                       roc_auc  = boot_roc[, label],
                       pr_auc   = boot_pr[, label])
      estimate <- switch(metric,
                         f1_score = point_estimates[[label]]$f1,
                         roc_auc  = point_estimates[[label]]$roc_auc,
                         pr_auc   = point_estimates[[label]]$pr_auc)
      ci <- percentile_ci(values)
      rows[[length(rows) + 1]] <- data.frame(
        model = label,
        metric = metric,
        point_estimate = estimate,
        ci_lower = ci[1],
        ci_upper = ci[2],
        bootstrap_se = stats::sd(values[is.finite(values)]),
        n_valid_resamples = sum(is.finite(values)),
        stringsAsFactors = FALSE
      )
    }
  }

  # Paired differences in F1 between every model pair, from the shared replicates.
  diff_rows <- list()
  for (pair in utils::combn(model_labels, 2, simplify = FALSE)) {
    a <- pair[1]; b <- pair[2]
    deltas <- boot_f1[, a] - boot_f1[, b]
    deltas <- deltas[is.finite(deltas)]
    ci <- percentile_ci(deltas)
    diff_rows[[length(diff_rows) + 1]] <- data.frame(
      model_a = a, model_b = b,
      f1_difference = point_estimates[[a]]$f1 - point_estimates[[b]]$f1,
      diff_ci_lower = ci[1], diff_ci_upper = ci[2],
      ci_excludes_zero = !is.na(ci[1]) && !is.na(ci[2]) && (ci[1] > 0 || ci[2] < 0),
      stringsAsFactors = FALSE
    )
  }
  bootstrap_differences <<- do.call(rbind, diff_rows)

  do.call(rbind, rows)
}, error = function(e) {
  warning("Bootstrap confidence interval estimation failed: ", conditionMessage(e))
  NULL
})

if (!is.null(bootstrap_results)) {
  message("\n---------------- BOOTSTRAP 95% CONFIDENCE INTERVALS ----------------")
  message(sprintf("%-14s %-10s %10s %10s %10s", "Model", "Metric", "Estimate", "CI lower", "CI upper"))
  for (i in seq_len(nrow(bootstrap_results))) {
    r <- bootstrap_results[i, ]
    message(sprintf("%-14s %-10s %10.4f %10.4f %10.4f",
                    r$model, r$metric, r$point_estimate, r$ci_lower, r$ci_upper))
  }
}

if (!is.null(bootstrap_differences)) {
  message("\n---------------- PAIRED BOOTSTRAP F1 DIFFERENCES ----------------")
  for (i in seq_len(nrow(bootstrap_differences))) {
    r <- bootstrap_differences[i, ]
    message(sprintf("%-14s - %-14s  dF1 = %+7.4f  95%% CI [%+7.4f, %+7.4f]  %s",
                    r$model_a, r$model_b, r$f1_difference,
                    r$diff_ci_lower, r$diff_ci_upper,
                    if (r$ci_excludes_zero) "excludes 0" else "includes 0"))
  }
  write.csv(bootstrap_differences, file.path("outputs", "final_bootstrap_f1_differences.csv"),
            row.names = FALSE)
  message("Saved: outputs/final_bootstrap_f1_differences.csv")
}

# ---------------------------------------------------------------------------
# 8. Final summary table
# ---------------------------------------------------------------------------

summary_table <- tryCatch({
  if (is.null(bootstrap_results)) stop("bootstrap results are unavailable.")

  # For each model, the set of opponents it significantly beats / loses to.
  significant_against <- setNames(rep("", length(model_labels)), model_labels)
  if (!is.null(pairwise_results)) {
    for (label in model_labels) {
      involved <- pairwise_results[(pairwise_results$model_a == label |
                                    pairwise_results$model_b == label) &
                                   pairwise_results$significant_holm, , drop = FALSE]
      if (nrow(involved) == 0) {
        significant_against[label] <- "none"
        next
      }
      descriptions <- character(0)
      for (i in seq_len(nrow(involved))) {
        r <- involved[i, ]
        opponent <- if (r$model_a == label) r$model_b else r$model_a
        direction <- if (r$better_model == label) "better than" else "worse than"
        descriptions <- c(descriptions, sprintf("%s %s (p=%.3g)", direction, opponent, r$p_value_holm))
      }
      significant_against[label] <- paste(descriptions, collapse = "; ")
    }
  } else {
    significant_against[] <- NA_character_
  }

  out <- bootstrap_results
  out$model_version <- vapply(out$model, function(l) models[[l]]$version, character(1))
  out$significant_pairwise_differences <- significant_against[out$model]
  out <- out[order(match(out$model, model_labels),
                   match(out$metric, c("f1_score", "roc_auc", "pr_auc"))), , drop = FALSE]
  out <- out[, c("model", "model_version", "metric", "point_estimate", "ci_lower", "ci_upper",
                 "bootstrap_se", "n_valid_resamples", "significant_pairwise_differences")]
  row.names(out) <- NULL

  write.csv(out, file.path("outputs", "final_statistical_comparison.csv"), row.names = FALSE)
  message("\nSaved: outputs/final_statistical_comparison.csv")
  out
}, error = function(e) {
  warning("Final summary table could not be assembled: ", conditionMessage(e))
  NULL
})

# ---------------------------------------------------------------------------
# 9. Per-time_step performance for the best model
# ---------------------------------------------------------------------------
#
# The test set spans time_step 35-49 while training covers 1-34, so performance
# is expected to vary over time. This breakdown shows where it degrades.

per_timestep <- tryCatch({
  f1_values <- vapply(model_labels, function(l) point_estimates[[l]]$f1, numeric(1))
  best_label <- model_labels[which.max(f1_values)]
  best <- models[[best_label]]

  steps <- sort(unique(time_step))
  rows <- list()
  for (s in steps) {
    in_step <- time_step == s
    truth_s <- y_binary[in_step]
    predicted_s <- best$predicted_positive[in_step]
    rows[[length(rows) + 1]] <- data.frame(
      model = best_label,
      time_step = s,
      n_transactions = sum(in_step),
      n_illicit = sum(truth_s == 1),
      illicit_rate = mean(truth_s == 1),
      detected = sum(predicted_s == 1 & truth_s == 1),
      missed = sum(predicted_s == 0 & truth_s == 1),
      false_alarms = sum(predicted_s == 1 & truth_s == 0),
      recall = compute_recall(truth_s, predicted_s),
      precision = compute_precision(truth_s, predicted_s),
      stringsAsFactors = FALSE
    )
  }
  out <- do.call(rbind, rows)

  message(sprintf("\n---------------- PER-TIME_STEP PERFORMANCE (%s) ----------------", best_label))
  message(sprintf("%9s %8s %8s %8s %9s %9s", "time_step", "n", "illicit", "detected", "recall", "precision"))
  for (i in seq_len(nrow(out))) {
    r <- out[i, ]
    message(sprintf("%9d %8d %8d %8d %9s %9s", r$time_step, r$n_transactions, r$n_illicit, r$detected,
                    ifelse(is.na(r$recall), "-", sprintf("%.3f", r$recall)),
                    ifelse(is.na(r$precision), "-", sprintf("%.3f", r$precision))))
  }

  write.csv(out, file.path("outputs", "final_per_timestep_performance.csv"), row.names = FALSE)
  message("Saved: outputs/final_per_timestep_performance.csv")
  out
}, error = function(e) {
  warning("Per-time_step breakdown failed: ", conditionMessage(e))
  NULL
})

# ---------------------------------------------------------------------------
# 10. Console summary
# ---------------------------------------------------------------------------

message("\n==================== FINAL STATISTICAL ANALYSIS SUMMARY ====================")
message(sprintf("Models compared: %s", paste(model_labels, collapse = ", ")))
if (!is.null(summary_table)) {
  best_f1 <- summary_table[summary_table$metric == "f1_score", ]
  best_f1 <- best_f1[which.max(best_f1$point_estimate), ]
  message(sprintf("Highest F1:      %s (%.4f, 95%% CI [%.4f, %.4f])",
                  best_f1$model, best_f1$point_estimate, best_f1$ci_lower, best_f1$ci_upper))
}
if (!is.null(pairwise_results)) {
  n_sig <- sum(pairwise_results$significant_holm, na.rm = TRUE)
  message(sprintf("Pairwise tests:  %d of %d comparisons significant at alpha = %.2f (Holm-adjusted)",
                  n_sig, nrow(pairwise_results), ALPHA))
}
message("\nFiles written:")
for (f in c("final_statistical_comparison.csv", "final_pairwise_mcnemar.csv",
            "final_bootstrap_f1_differences.csv", "final_per_timestep_performance.csv")) {
  path <- file.path("outputs", f)
  message(sprintf("  %-40s %s", f, if (file.exists(path)) "[OK]" else "[NOT CREATED]"))
}
message(sprintf("\nTotal runtime: %s", format(Sys.time() - script_start)))
message("===========================================================================")
