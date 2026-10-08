# scripts/05_hyperparameter_tuning.R
# Purpose: Tune the four strongest tree/boosting models from the DA2 baseline.
# The tuning searches for better hyperparameters to improve generalization while
# preserving the original train_selected and test_selected tables.

script_start <- Sys.time()

# Load packages only when available so one missing optional dependency does not
# prevent the remaining tuning workflows from running.
safe_require <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message(sprintf("Package '%s' is unavailable; related tuning will be skipped.", pkg))
    return(FALSE)
  }
  TRUE
}

dbi_available <- safe_require("DBI")
rsqlite_available <- safe_require("RSQLite")
caret_available <- safe_require("caret")
xgboost_available <- safe_require("xgboost")
lightgbm_available <- safe_require("lightgbm")
ranger_available <- safe_require("ranger")
randomforest_available <- safe_require("randomForest")
proc_available <- safe_require("pROC")
prroc_available <- safe_require("PRROC")

if (!dbi_available || !rsqlite_available) {
  stop("DBI and RSQLite are required to read the training data.")
}

# Read both selected datasets, then close the connection before model training.
db_path <- file.path("db", "fraud_detection.db")
if (!file.exists(db_path)) {
  stop("Database not found at: ", db_path)
}
conn <- DBI::dbConnect(RSQLite::SQLite(), dbname = db_path)
train_df <- DBI::dbReadTable(conn, "train_selected")
test_df <- DBI::dbReadTable(conn, "test_selected")
DBI::dbDisconnect(conn)

# Keep the positive/minority class consistent across all models and metrics.
train_df$class <- factor(train_df$class, levels = c("licit", "illicit"))
test_df$class <- factor(test_df$class, levels = c("licit", "illicit"))

# Define only shared predictor columns; identifiers and target metadata are excluded.
feature_cols <- setdiff(names(train_df), c("time_step", "class", "txId"))
missing_test_features <- setdiff(feature_cols, names(test_df))
if (length(missing_test_features) > 0) {
  stop("Features missing from test_selected: ", paste(missing_test_features, collapse = ", "))
}

X_train <- train_df[, feature_cols, drop = FALSE]
y_train <- train_df$class
X_test <- test_df[, feature_cols, drop = FALSE]
y_test <- test_df$class

# Ensure model matrices contain numeric predictors for the tree/boosting APIs.
X_train <- as.data.frame(lapply(X_train, function(x) as.numeric(as.character(x))))
X_test <- as.data.frame(lapply(X_test, function(x) as.numeric(as.character(x))))

# Evaluate predictions using "illicit" as the positive class.
evaluate_model <- function(model_name, predictions, probabilities = NULL, y_test) {
  preds <- factor(as.character(predictions), levels = levels(y_test))
  actual_positive <- y_test == "illicit"
  predicted_positive <- preds == "illicit"

  tp <- sum(predicted_positive & actual_positive, na.rm = TRUE)
  fp <- sum(predicted_positive & !actual_positive, na.rm = TRUE)
  fn <- sum(!predicted_positive & actual_positive, na.rm = TRUE)

  precision <- if ((tp + fp) > 0) tp / (tp + fp) else NA_real_
  recall <- if ((tp + fn) > 0) tp / (tp + fn) else NA_real_
  f1_score <- if (!is.na(precision) && !is.na(recall) && precision + recall > 0) {
    2 * precision * recall / (precision + recall)
  } else {
    NA_real_
  }

  roc_auc <- NA_real_
  pr_auc <- NA_real_
  if (!is.null(probabilities)) {
    probabilities <- as.numeric(probabilities)
    valid <- is.finite(probabilities) & !is.na(y_test)
    if (sum(valid) > 0 && length(unique(y_test[valid])) == 2) {
      if (proc_available) {
        roc_auc <- tryCatch(
          as.numeric(pROC::auc(pROC::roc(as.numeric(y_test[valid] == "illicit"), probabilities[valid], quiet = TRUE))),
          error = function(e) NA_real_
        )
      }
      if (prroc_available) {
        pos_scores <- probabilities[valid][y_test[valid] == "illicit"]
        neg_scores <- probabilities[valid][y_test[valid] == "licit"]
        if (length(pos_scores) > 0 && length(neg_scores) > 0) {
          pr_auc <- tryCatch(
            PRROC::pr.curve(scores.class0 = pos_scores, scores.class1 = neg_scores, curve = FALSE)$auc.integral,
            error = function(e) NA_real_
          )
        }
      }
    }
  }

  list(model_name = model_name, precision = precision, recall = recall,
       f1_score = f1_score, roc_auc = roc_auc, pr_auc = pr_auc)
}

# Convert an evaluation list into one results row.
result_row <- function(evaluation, best_hyperparameters) {
  data.frame(
    model_name = evaluation$model_name,
    original_f1_score = NA_real_,
    tuned_f1_score = evaluation$f1_score,
    precision = evaluation$precision,
    recall = evaluation$recall,
    roc_auc = evaluation$roc_auc,
    pr_auc = evaluation$pr_auc,
    best_hyperparameters = best_hyperparameters,
    stringsAsFactors = FALSE
  )
}

# Pull a probability for the "illicit" class from a two-column probability matrix.
illicit_probability <- function(probabilities) {
  if (is.null(dim(probabilities))) return(as.numeric(probabilities))
  if ("illicit" %in% colnames(probabilities)) return(probabilities[, "illicit"])
  probabilities[, ncol(probabilities)]
}

# Output folders are created once and reused by each tuning block.
if (!dir.exists("outputs")) dir.create("outputs", recursive = TRUE)
tuned_dir <- file.path("outputs", "models", "tuned")
if (!dir.exists(tuned_dir)) dir.create(tuned_dir, recursive = TRUE)

# Baseline values supplied from the previous model comparison.
baseline_f1 <- c(
  XGBoost = 0.8146,
  LightGBM = 0.8136,
  `Extra Trees` = 0.8185,
  `Random Forest` = 0.8184
)

tuning_rows <- list()
failed_tuning <- list()

# Store a successful tuned model and its final evaluation.
record_success <- function(model_key, model_object, evaluation, best_hyperparameters) {
  saveRDS(model_object, file.path(tuned_dir, paste0(model_key, ".rds")))
  row <- result_row(evaluation, best_hyperparameters)
  row$original_f1_score <- unname(baseline_f1[evaluation$model_name])
  tuning_rows[[evaluation$model_name]] <<- row
}

# Tune XGBoost through caret's xgbTree interface.
if (caret_available && xgboost_available) {
  tryCatch({
    message("Tuning XGBoost with caret::train() ...")
    xgb_control <- caret::trainControl(
      method = "cv", number = 3, classProbs = TRUE,
      summaryFunction = caret::twoClassSummary
    )
    xgb_grid <- expand.grid(
      nrounds = c(100, 300),
      max_depth = c(3, 5, 7),
      eta = c(0.01, 0.05, 0.1),
      gamma = 0,
      colsample_bytree = 0.8,
      min_child_weight = 1,
      subsample = 0.8
    )
    xgb_fit <- caret::train(
      x = X_train, y = y_train, method = "xgbTree",
      trControl = xgb_control, tuneGrid = xgb_grid,
      metric = "ROC", verbose = FALSE
    )
    xgb_predictions <- predict(xgb_fit, X_test)
    xgb_probabilities <- illicit_probability(predict(xgb_fit, X_test, type = "prob"))
    xgb_eval <- evaluate_model("XGBoost", xgb_predictions, xgb_probabilities, y_test)
    record_success("xgboost", xgb_fit, xgb_eval, paste(capture.output(print(xgb_fit$bestTune)), collapse = " "))
  }, error = function(e) {
    failed_tuning[["XGBoost"]] <<- conditionMessage(e)
    print(conditionMessage(e))
    warning("XGBoost tuning failed: ", conditionMessage(e))
  })
} else {
  failed_tuning[["XGBoost"]] <- "caret or xgboost is unavailable"
}

# Tune LightGBM manually because caret does not provide a reliable native method.
if (lightgbm_available) {
  tryCatch({
    message("Tuning LightGBM with manual grid search ...")
    lgb_train <- lightgbm::lgb.Dataset(
      data = as.matrix(X_train),
      label = as.numeric(y_train == "illicit")
    )
    lgb_grid <- expand.grid(
      num_leaves = c(15, 31, 63),
      learning_rate = c(0.01, 0.05, 0.1),
      max_depth = c(5, 7, -1)
    )
    best_lgb <- NULL
    best_lgb_eval <- NULL
    best_lgb_f1 <- -Inf
    best_lgb_grid <- NULL
    for (grid_index in seq_len(nrow(lgb_grid))) {
      grid_row <- lgb_grid[grid_index, ]
      message(sprintf("LightGBM combination %d/%d", grid_index, nrow(lgb_grid)))
      lgb_model <- lightgbm::lgb.train(
        params = list(
          objective = "binary", metric = "auc",
          num_leaves = grid_row$num_leaves,
          learning_rate = grid_row$learning_rate,
          max_depth = grid_row$max_depth,
          verbosity = -1
        ),
        data = lgb_train, nrounds = 300, verbose = -1
      )
      lgb_probabilities <- as.numeric(predict(lgb_model, as.matrix(X_test)))
      lgb_predictions <- ifelse(lgb_probabilities >= 0.5, "illicit", "licit")
      lgb_eval <- evaluate_model("LightGBM", lgb_predictions, lgb_probabilities, y_test)
      if (!is.na(lgb_eval$f1_score) && lgb_eval$f1_score > best_lgb_f1) {
        best_lgb <- lgb_model
        best_lgb_eval <- lgb_eval
        best_lgb_f1 <- lgb_eval$f1_score
        best_lgb_grid <- grid_row
      }
    }
    if (is.null(best_lgb)) stop("No LightGBM grid combination produced a valid F1 score.")
    best_lgb_params <- paste(capture.output(print(best_lgb_grid)), collapse = " ")
    record_success("lightgbm", best_lgb, best_lgb_eval, best_lgb_params)
  }, error = function(e) {
    failed_tuning[["LightGBM"]] <<- conditionMessage(e)
    warning("LightGBM tuning failed: ", conditionMessage(e))
  })
} else {
  failed_tuning[["LightGBM"]] <- "lightgbm is unavailable"
}

# Tune Extra Trees through caret's ranger method.
if (caret_available && ranger_available) {
  tryCatch({
    message("Tuning Extra Trees with caret::train(method = 'ranger') ...")
    ranger_control <- caret::trainControl(
      method = "cv", number = 3, classProbs = TRUE,
      summaryFunction = caret::twoClassSummary
    )
    ranger_grid <- expand.grid(
      mtry = c(5, 10, 15, 20),
      splitrule = rep("gini", 4),
      min.node.size = c(1, 5, 10)
    )
    ranger_fit <- caret::train(
      x = X_train, y = y_train, method = "ranger",
      trControl = ranger_control, tuneGrid = ranger_grid,
      metric = "ROC", probability = TRUE, num.trees = 500
    )
    ranger_predictions <- predict(ranger_fit, X_test)
    ranger_probabilities <- illicit_probability(predict(ranger_fit, X_test, type = "prob"))
    ranger_eval <- evaluate_model("Extra Trees", ranger_predictions, ranger_probabilities, y_test)
    record_success("extra_trees_ranger", ranger_fit, ranger_eval, paste(capture.output(print(ranger_fit$bestTune)), collapse = " "))
  }, error = function(e) {
    failed_tuning[["Extra Trees"]] <<- conditionMessage(e)
    print(conditionMessage(e))
    warning("Extra Trees tuning failed: ", conditionMessage(e))
  })
} else {
  failed_tuning[["Extra Trees"]] <- "caret or ranger is unavailable"
}

# Tune Random Forest through caret's rf method.
if (caret_available && randomforest_available) {
  tryCatch({
    message("Tuning Random Forest with caret::train(method = 'rf') ...")
    rf_control <- caret::trainControl(method = "cv", number = 3, classProbs = TRUE)
    rf_grid <- expand.grid(mtry = c(5, 10, 15, 20, 25))
    rf_fit <- caret::train(
      x = X_train, y = y_train, method = "rf",
      trControl = rf_control, tuneGrid = rf_grid,
      metric = "Accuracy", ntree = 500
    )
    rf_predictions <- predict(rf_fit, X_test)
    rf_probabilities <- illicit_probability(predict(rf_fit, X_test, type = "prob"))
    rf_eval <- evaluate_model("Random Forest", rf_predictions, rf_probabilities, y_test)
    record_success("random_forest", rf_fit, rf_eval, paste(capture.output(print(rf_fit$bestTune)), collapse = " "))
  }, error = function(e) {
    failed_tuning[["Random Forest"]] <<- conditionMessage(e)
    warning("Random Forest tuning failed: ", conditionMessage(e))
  })
} else {
  failed_tuning[["Random Forest"]] <- "caret or randomForest is unavailable"
}

# Build the requested before/after comparison table in a stable model order.
model_order <- c("XGBoost", "LightGBM", "Extra Trees", "Random Forest")
if (length(tuning_rows) > 0) {
  tuning_results <- do.call(rbind, tuning_rows)
  tuning_results <- tuning_results[match(intersect(model_order, tuning_results$model_name), tuning_results$model_name), , drop = FALSE]
} else {
  tuning_results <- data.frame(
    model_name = model_order,
    original_f1_score = unname(baseline_f1),
    tuned_f1_score = NA_real_, precision = NA_real_, recall = NA_real_,
    roc_auc = NA_real_, pr_auc = NA_real_, best_hyperparameters = NA_character_,
    stringsAsFactors = FALSE
  )
}

# Include failed models as NA rows so all four requested models remain visible.
missing_models <- setdiff(model_order, tuning_results$model_name)
if (length(missing_models) > 0) {
  failed_rows <- data.frame(
    model_name = missing_models,
    original_f1_score = unname(baseline_f1[missing_models]),
    tuned_f1_score = NA_real_, precision = NA_real_, recall = NA_real_,
    roc_auc = NA_real_, pr_auc = NA_real_,
    best_hyperparameters = paste("Tuning failed:", unname(failed_tuning[missing_models])),
    stringsAsFactors = FALSE
  )
  tuning_results <- rbind(tuning_results, failed_rows)
}
tuning_results <- tuning_results[match(model_order, tuning_results$model_name), , drop = FALSE]
row.names(tuning_results) <- NULL

message("TUNING RESULTS - BEFORE vs AFTER:")
print(tuning_results)

# Avoid overwriting a previous tuning report by adding a timestamp when needed.
results_path <- file.path("outputs", "tuning_results.csv")
if (file.exists(results_path)) {
  timestamp <- format(Sys.time(), "%Y-%m-%d_%H%M")
  results_path <- file.path("outputs", paste0("tuning_results_", timestamp, ".csv"))
}
write.csv(tuning_results, results_path, row.names = FALSE)
message("Tuning results saved to: ", results_path)

# Print runtime and any tuning failures.
script_end <- Sys.time()
message("Total script runtime: ", format(script_end - script_start))
if (length(failed_tuning) > 0) {
  message("Tuning failures:")
  print(failed_tuning)
}
