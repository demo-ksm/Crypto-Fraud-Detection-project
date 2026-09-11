# scripts/04_train_models.R
# Purpose: Train multiple baseline classification models on the SMOTE-balanced
# `train_selected` dataset and evaluate them on `test_selected`. The goal is
# to produce a comparison table of model performance and persist trained
# model objects for later use. Highly robust: each model is attempted inside
# a tryCatch so failures don't stop the whole script. Class factor levels are
# set to c("licit", "illicit") so "illicit" is treated as the positive
# (minority) class consistently across evaluations.

# Load core libraries
library(DBI)
library(RSQLite)
# Many ML packages are optional; we check availability before use and skip
# models whose packages are missing. Note: catboost is optional and may be
# skipped if not installed on the system.

# Helper: safe load of namespaces
safe_require <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message(sprintf("Package '%s' not available — related models will be skipped.", pkg))
    return(FALSE)
  }
  return(TRUE)
}

# Attempt to load commonly used packages (quietly).
safe_require("caret")
safe_require("randomForest")
safe_require("e1071")
safe_require("class")
safe_require("naivebayes")
safe_require("xgboost")
safe_require("lightgbm")
safe_require("adabag")
safe_require("gbm")
safe_require("nnet")
# Optional: catboost
catboost_available <- safe_require("catboost")

# Additional metric packages
prroc_available <- safe_require("PRROC")
mlmetrics_available <- safe_require("MLmetrics")
pROC_available <- safe_require("pROC")

# Read data from DB and disconnect
db_path <- file.path("db", "fraud_detection.db")
if (!file.exists(db_path)) stop("Database not found at: ", db_path)
conn <- dbConnect(RSQLite::SQLite(), dbname = db_path)
train_df <- dbReadTable(conn, "train_selected")
test_df  <- dbReadTable(conn, "test_selected")
dbDisconnect(conn)

# Ensure class factors in consistent order so "illicit" is positive
train_df$class <- factor(train_df$class, levels = c("licit", "illicit"))
test_df$class  <- factor(test_df$class,  levels = c("licit", "illicit"))

# Define features
feature_cols <- setdiff(names(train_df), c("time_step", "class"))

# Create training and test matrices
X_train <- train_df[, feature_cols, drop = FALSE]
y_train <- train_df$class
X_test  <- test_df[, feature_cols, drop = FALSE]
y_test  <- test_df$class

# Utility: evaluate model predictions
evaluate_model <- function(model_name, predictions, probabilities = NULL, y_test) {
  # Ensure predictions are a factor with same levels as y_test
  preds <- as.character(predictions)
  preds <- factor(preds, levels = levels(y_test))

  # Compute confusion-based metrics
  tp <- sum(preds == "illicit" & y_test == "illicit")
  fp <- sum(preds == "illicit" & y_test == "licit")
  fn <- sum(preds == "licit" & y_test == "illicit")

  precision <- if ((tp + fp) > 0) tp / (tp + fp) else NA
  recall    <- if ((tp + fn) > 0) tp / (tp + fn) else NA
  f1_score  <- if (!is.na(precision) && !is.na(recall) && (precision + recall) > 0) 2 * precision * recall / (precision + recall) else NA

  # ROC AUC (requires probabilities for the positive class)
  roc_auc <- NA
  pr_auc  <- NA
  if (!is.null(probabilities) && !all(is.na(probabilities)) && pROC_available) {
    # probabilities should represent probability of 'illicit'
    roc_obj <- tryCatch(pROC::roc(response = as.numeric(y_test == "illicit"), predictor = as.numeric(probabilities), quiet = TRUE), error = function(e) NULL)
    if (!is.null(roc_obj)) roc_auc <- as.numeric(pROC::auc(roc_obj))
  }

  # PR AUC: try MLmetrics::PRAUC first, then PRROC
  if (!is.null(probabilities) && !all(is.na(probabilities))) {
    if (mlmetrics_available) {
      pr_auc <- tryCatch(MLmetrics::PRAUC(probabilities, as.numeric(y_test == "illicit")), error = function(e) NA)
    } else if (prroc_available) {
      # PRROC expects two score vectors: positives and negatives
      pos_scores <- probabilities[y_test == "illicit"]
      neg_scores <- probabilities[y_test == "licit"]
      if (length(pos_scores) > 0 && length(neg_scores) > 0) {
        pr <- tryCatch(PRROC::pr.curve(scores.class0 = pos_scores, scores.class1 = neg_scores, curve = FALSE), error = function(e) NULL)
        if (!is.null(pr)) pr_auc <- pr$auc.integral
      }
    }
  }

  return(list(model_name = model_name,
              precision = precision,
              recall = recall,
              f1_score = f1_score,
              roc_auc = roc_auc,
              pr_auc = pr_auc))
}

# Prepare storage for results and failed models
results_list <- list()
failed_models <- list()
models_dir <- file.path("outputs", "models")
if (!dir.exists("outputs")) dir.create("outputs")
if (!dir.exists(models_dir)) dir.create(models_dir)

# Helper to append result
append_result <- function(res) {
  results_list[[length(results_list) + 1]] <<- res
}

# Training helpers per model
attempt_model <- function(name, code_block) {
  message(sprintf("Training model: %s", name))
  tryCatch({
    out <- code_block()
    # out should return a list with elements: model, predictions, probabilities (nullable)
    eval_res <- evaluate_model(name, out$predictions, out$probabilities, y_test)
    append_result(eval_res)
    # Save model object if present
    if (!is.null(out$model)) {
      saveRDS(out$model, file = file.path(models_dir, paste0(gsub("[[:space:]]+", "_", tolower(name)), ".rds")))
    }
    TRUE
  }, error = function(e) {
    warning(sprintf("Model %s failed: %s", name, e$message))
    failed_models[[name]] <<- e$message
    FALSE
  })
}

# 1) Logistic Regression
if (safe_require("stats")) {
  attempt_model("Logistic Regression", function() {
    model <- glm(class ~ ., data = train_df[, c("class", feature_cols)], family = binomial)
    probs <- tryCatch(predict(model, newdata = X_test, type = "response"), error = function(e) rep(NA, nrow(X_test)))
    preds <- ifelse(!is.na(probs) & probs > 0.5, "illicit", "licit")
    list(model = model, predictions = preds, probabilities = probs)
  })
}

# 2) Decision Tree (rpart)
if (safe_require("rpart")) {
  attempt_model("Decision Tree", function() {
    library(rpart)
    model <- rpart::rpart(class ~ ., data = train_df[, c("class", feature_cols)], method = "class")
    preds_raw <- predict(model, newdata = X_test, type = "class")
    probs_mat <- predict(model, newdata = X_test, type = "prob")
    probs <- if ("illicit" %in% colnames(probs_mat)) probs_mat[, "illicit"] else probs_mat[,1]
    list(model = model, predictions = preds_raw, probabilities = probs)
  })
}

# 3) Random Forest
if (safe_require("randomForest")) {
  attempt_model("Random Forest", function() {
    model <- randomForest::randomForest(x = X_train, y = y_train)
    probs_mat <- predict(model, newdata = X_test, type = "prob")
    probs <- if ("illicit" %in% colnames(probs_mat)) probs_mat[, "illicit"] else probs_mat[,1]
    preds <- predict(model, newdata = X_test, type = "response")
    list(model = model, predictions = preds, probabilities = probs)
  })
}

# 4) Extra Trees / Extremely Randomized Trees (ranger preferred)
if (safe_require("ranger")) {
  attempt_model("Extra Trees (ranger)", function() {
    model <- ranger::ranger(dependent.variable.name = "class", data = train_df[, c("class", feature_cols)], probability = TRUE)
    probs_mat <- predict(model, data = X_test)$predictions
    probs <- if ("illicit" %in% colnames(probs_mat)) probs_mat[, "illicit"] else probs_mat[,1]
    preds <- factor(ifelse(probs > 0.5, "illicit", "licit"), levels = levels(y_test))
    list(model = model, predictions = preds, probabilities = probs)
  })
} else if (safe_require("randomForest")) {
  # Fallback: use randomForest with smaller mtry to increase randomness
  attempt_model("Extra Trees (randomForest-fallback)", function() {
    model <- randomForest::randomForest(x = X_train, y = y_train, mtry = max(1, floor(ncol(X_train)/3)))
    probs_mat <- predict(model, newdata = X_test, type = "prob")
    probs <- if ("illicit" %in% colnames(probs_mat)) probs_mat[, "illicit"] else probs_mat[,1]
    preds <- predict(model, newdata = X_test, type = "response")
    list(model = model, predictions = preds, probabilities = probs)
  })
}

# 5) AdaBoost (adabag)
if (safe_require("adabag")) {
  attempt_model("AdaBoost", function() {
    # adabag::boosting expects formula and data
    model <- adabag::boosting(class ~ ., data = train_df[, c("class", feature_cols)], boos = TRUE)
    pred_obj <- adabag::predict.boosting(model, newdata = X_test)
    preds <- pred_obj$class
    # pred_obj$prob is a matrix of class probabilities if available
    probs <- if (!is.null(pred_obj$prob) && "illicit" %in% colnames(pred_obj$prob)) pred_obj$prob[, "illicit"] else NULL
    list(model = model, predictions = preds, probabilities = probs)
  })
}

# 6) Gradient Boosting (gbm)
if (safe_require("gbm")) {
  attempt_model("GBM", function() {
    # gbm requires numeric response 0/1 for bernoulli
    gbm_train <- train_df[, c("class", feature_cols)]
    gbm_train$y_numeric <- as.numeric(gbm_train$class == "illicit")
    formula_str <- as.formula(paste("y_numeric ~", paste(feature_cols, collapse = " + ")))
    model <- gbm::gbm(formula = formula_str, data = gbm_train, distribution = "bernoulli", n.trees = 100, interaction.depth = 3, shrinkage = 0.05, verbose = FALSE)
    probs <- predict(model, newdata = X_test, n.trees = 100, type = "response")
    preds <- ifelse(probs > 0.5, "illicit", "licit")
    list(model = model, predictions = preds, probabilities = probs)
  })
}

# 7) SVM (e1071)
if (safe_require("e1071")) {
  attempt_model("SVM", function() {
    model <- e1071::svm(x = X_train, y = y_train, probability = TRUE)
    preds <- predict(model, newdata = X_test)
    probs_attr <- attr(predict(model, newdata = X_test, probability = TRUE), "probabilities")
    probs <- if (!is.null(probs_attr) && "illicit" %in% colnames(probs_attr)) probs_attr[, "illicit"] else NULL
    list(model = model, predictions = preds, probabilities = probs)
  })
}

# 8) KNN (class::knn, k=5) — note: knn doesn't produce a reusable model object
if (safe_require("class")) {
  attempt_model("KNN (k=5)", function() {
    # class::knn expects training and test matrices
    k <- 5
    knn_out <- class::knn(train = as.matrix(X_train), test = as.matrix(X_test), cl = y_train, k = k, prob = TRUE)
    preds <- knn_out
    prob_attr <- attr(knn_out, "prob")
    # prob_attr is the proportion of votes for the winning class; derive probability of 'illicit'
    probs <- rep(NA, length(prob_attr))
    probs[preds == "illicit"] <- prob_attr[preds == "illicit"]
    probs[preds == "licit"] <- 1 - prob_attr[preds == "licit"]
    # For saving, store the training set and k so the procedure can be rerun later
    model_obj <- list(method = "knn", train = X_train, cl = y_train, k = k)
    list(model = model_obj, predictions = preds, probabilities = probs)
  })
}

# 9) Naive Bayes (naivebayes)
if (safe_require("naivebayes")) {
  attempt_model("Naive Bayes", function() {
    model <- naivebayes::naive_bayes(x = X_train, y = y_train)
    probs_mat <- predict(model, newdata = X_test, type = "prob")
    probs <- if ("illicit" %in% colnames(probs_mat)) probs_mat[, "illicit"] else probs_mat[,1]
    preds <- predict(model, newdata = X_test)
    list(model = model, predictions = preds, probabilities = probs)
  })
}

# 10) XGBoost
if (safe_require("xgboost")) {
  attempt_model("XGBoost", function() {
    dtrain <- xgboost::xgb.DMatrix(data = as.matrix(X_train), label = as.numeric(y_train == "illicit"))
    dtest  <- xgboost::xgb.DMatrix(data = as.matrix(X_test))
    params <- list(objective = "binary:logistic", eval_metric = "auc")
    model <- xgboost::xgb.train(params = params, data = dtrain, nrounds = 100, verbose = 0)
    probs <- predict(model, dtest)
    preds <- ifelse(probs > 0.5, "illicit", "licit")
    list(model = model, predictions = preds, probabilities = probs)
  })
}

# 11) LightGBM
if (safe_require("lightgbm")) {
  attempt_model("LightGBM", function() {
    ltrain <- lightgbm::lgb.Dataset(data = as.matrix(X_train), label = as.numeric(y_train == "illicit"))
    params <- list(objective = "binary", metric = "auc")
    model <- lightgbm::lgb.train(params = params, data = ltrain, nrounds = 100, verbose = -1)
    probs <- predict(model, as.matrix(X_test))
    preds <- ifelse(probs > 0.5, "illicit", "licit")
    list(model = model, predictions = preds, probabilities = probs)
  })
} else {
  message("LightGBM not available; skipping.")
}

# 12) Neural Network (nnet)
if (safe_require("nnet")) {
  attempt_model("Neural Network (nnet)", function() {
    # nnet requires numeric predictors and a factor response; size=10 hidden units
    model <- nnet::nnet(x = as.matrix(X_train), y = class.ind(y_train), size = 10, softmax = FALSE, trace = FALSE, maxit = 200)
    # For binary we can predict raw outputs then convert
    probs_raw <- predict(model, as.matrix(X_test))
    # If model produced a matrix, assume second column corresponds to 'illicit'
    probs <- if (is.matrix(probs_raw)) {
      # If two columns, try to map
      if (ncol(probs_raw) == 2) probs_raw[,2] else as.numeric(probs_raw)
    } else {
      as.numeric(probs_raw)
    }
    preds <- ifelse(probs > 0.5, "illicit", "licit")
    list(model = model, predictions = preds, probabilities = probs)
  })
}

# Optional: CatBoost (skip gracefully if unavailable)
if (catboost_available) {
  attempt_model("CatBoost", function() {
    # CatBoost R interface usage requires catboost package; training here would
    # require specifying pool and params; skipping detailed tuning for brevity.
    pool <- catboost::catboost.load_pool(data = as.matrix(X_train), label = as.numeric(y_train == "illicit"))
    params <- list(loss_function = 'Logloss', iterations = 100, verbose = 0)
    model <- catboost::catboost.train(pool, NULL, params = params)
    probs <- catboost::catboost.predict(model, catboost::catboost.load_pool(data = as.matrix(X_test)), prediction_type = 'Probability')
    preds <- ifelse(probs > 0.5, "illicit", "licit")
    list(model = model, predictions = preds, probabilities = probs)
  })
} else {
  # Note in comments that catboost was skipped when package was unavailable
  message("CatBoost not available; skipped.")
}

# Build results dataframe
if (length(results_list) > 0) {
  results_df <- do.call(rbind, lapply(results_list, function(x) data.frame(model_name = x$model_name,
                                                                            precision = x$precision,
                                                                            recall = x$recall,
                                                                            f1_score = x$f1_score,
                                                                            roc_auc = x$roc_auc,
                                                                            pr_auc = x$pr_auc,
                                                                            stringsAsFactors = FALSE)))
} else {
  results_df <- data.frame(model_name = character(0), precision = numeric(0), recall = numeric(0), f1_score = numeric(0), roc_auc = numeric(0), pr_auc = numeric(0), stringsAsFactors = FALSE)
}

# Print final comparison sorted by F1
results_df_sorted <- results_df[order(-results_df$f1_score, results_df$model_name), , drop = FALSE]
message("FINAL MODEL COMPARISON:")
print(results_df_sorted)

# Save results and models
write.csv(results_df_sorted, file = file.path("outputs", "model_comparison_results.csv"), row.names = FALSE)

# Summary of success/failures
n_attempted <- length(results_list) + length(failed_models)
n_success <- length(results_list)
message(sprintf("Models trained successfully: %d out of %d attempted", n_success, n_attempted))
if (length(failed_models) > 0) {
  message("Failed models and error messages:")
  print(failed_models)
}

message("Script complete. Results saved to outputs/model_comparison_results.csv and models saved under outputs/models/.")
