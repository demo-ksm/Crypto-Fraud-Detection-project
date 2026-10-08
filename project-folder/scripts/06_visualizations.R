# scripts/06_visualizations.R
# Purpose: Produce the DA2 comparative visualisations (ROC curves, Precision-Recall
# curves, confusion-matrix grid, feature-importance grid) for the four strongest
# models from the baseline comparison. Tuned model objects are used where they
# exist (LightGBM, Random Forest); the original baseline objects are used
# otherwise (XGBoost, Extra Trees).
#
# Every plotting block is wrapped in tryCatch so that a single failure produces a
# warning and the remaining plots are still written. All figures are saved to
# outputs/plots/ at 300 DPI.

script_start <- Sys.time()

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
ggplot2_available      <- safe_require("ggplot2")
proc_available         <- safe_require("pROC")
prroc_available        <- safe_require("PRROC")
rcolorbrewer_available <- safe_require("RColorBrewer")
xgboost_available      <- safe_require("xgboost")
lightgbm_available     <- safe_require("lightgbm")
ranger_available       <- safe_require("ranger")
randomforest_available <- safe_require("randomForest")
caret_available        <- safe_require("caret")

if (!dbi_available || !rsqlite_available) {
  stop("DBI and RSQLite are required to read the test data.")
}
if (!ggplot2_available) {
  stop("ggplot2 is required to produce the visualisations.")
}

library(ggplot2)

# ---------------------------------------------------------------------------
# 2. Test data from SQLite
# ---------------------------------------------------------------------------

db_path <- file.path("db", "fraud_detection.db")
if (!file.exists(db_path)) {
  stop("Database not found at: ", db_path)
}

conn <- DBI::dbConnect(RSQLite::SQLite(), dbname = db_path)
test_df <- DBI::dbReadTable(conn, "test_selected")
DBI::dbDisconnect(conn)

# "illicit" is the positive (minority) class everywhere in this project.
test_df$class <- factor(test_df$class, levels = c("licit", "illicit"))

feature_cols <- setdiff(names(test_df), c("time_step", "class", "txId"))
X_test <- as.data.frame(lapply(test_df[, feature_cols, drop = FALSE],
                               function(x) as.numeric(as.character(x))))
names(X_test) <- feature_cols
y_test <- test_df$class

message(sprintf("Test set loaded: %d rows, %d predictors (%d illicit / %d licit).",
                nrow(X_test), ncol(X_test),
                sum(y_test == "illicit"), sum(y_test == "licit")))

plots_dir <- file.path("outputs", "plots")
if (!dir.exists(plots_dir)) dir.create(plots_dir, recursive = TRUE)

# ---------------------------------------------------------------------------
# 3. Model loading (tuned where available, original otherwise)
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

# Return the probability of the "illicit" class for any of the model object
# types persisted by scripts 04 and 05.
predict_illicit_probability <- function(model, newdata) {
  if (inherits(model, "xgb.Booster")) {
    cols <- model$feature_names
    if (is.null(cols)) cols <- names(newdata)
    missing_cols <- setdiff(cols, names(newdata))
    if (length(missing_cols) > 0) {
      stop("Features missing from test data: ", paste(missing_cols, collapse = ", "))
    }
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

# Load one model, preferring the tuned object but falling back to the original
# if the tuned file is absent or cannot produce predictions.
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
      list(model = model, probabilities = probabilities,
           version = attempt$version, path = attempt$path)
    }, error = function(e) {
      warning(sprintf("Could not use %s model for %s (%s): %s",
                      attempt$version, spec$label, attempt$path, conditionMessage(e)))
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
    warning(sprintf("No usable model object for %s; it is excluded from all plots.", spec$label))
    next
  }
  loaded$label <- spec$label
  loaded$predictions <- factor(ifelse(loaded$probabilities >= 0.5, "illicit", "licit"),
                               levels = levels(y_test))
  models[[spec$label]] <- loaded
  message(sprintf("%-14s -> %-8s version  (%s)", spec$label, loaded$version, loaded$path))
}

if (length(models) == 0) {
  stop("No models could be loaded; nothing to plot.")
}

model_labels <- names(models)

# ---------------------------------------------------------------------------
# 4. Shared colour scale (RColorBrewer "Set2", one colour per model everywhere)
# ---------------------------------------------------------------------------

base_palette <- if (rcolorbrewer_available) {
  RColorBrewer::brewer.pal(max(3, length(model_spec)), "Set2")
} else {
  c("#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3")
}
model_colors <- setNames(base_palette[seq_along(model_spec)],
                         vapply(model_spec, function(s) s$label, character(1)))
model_colors <- model_colors[model_labels]

plot_theme <- theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold", size = 14),
        legend.position = "bottom",
        panel.grid.minor = element_blank())

created_plots <- character(0)
failed_plots  <- character(0)

record_plot <- function(name, success) {
  if (success) {
    created_plots <<- c(created_plots, name)
  } else {
    failed_plots <<- c(failed_plots, name)
  }
}

# ---------------------------------------------------------------------------
# 5a. ROC curves (all models overlaid, AUC in the legend)
# ---------------------------------------------------------------------------

roc_success <- tryCatch({
  if (!proc_available) stop("pROC is not installed.")

  roc_frames <- list()
  roc_legend <- character(0)
  for (label in model_labels) {
    roc_obj <- pROC::roc(response = as.numeric(y_test == "illicit"),
                         predictor = models[[label]]$probabilities,
                         quiet = TRUE)
    auc_value <- as.numeric(pROC::auc(roc_obj))
    legend_label <- sprintf("%s (AUC = %.4f)", label, auc_value)
    roc_legend[label] <- legend_label
    roc_frames[[label]] <- data.frame(
      fpr = 1 - roc_obj$specificities,
      tpr = roc_obj$sensitivities,
      model = legend_label,
      stringsAsFactors = FALSE
    )
    message(sprintf("ROC-AUC  %-14s %.4f", label, auc_value))
  }

  roc_df <- do.call(rbind, roc_frames)
  roc_df$model <- factor(roc_df$model, levels = unname(roc_legend[model_labels]))
  roc_colors <- setNames(unname(model_colors[model_labels]),
                         unname(roc_legend[model_labels]))

  roc_plot <- ggplot(roc_df, aes(x = fpr, y = tpr, colour = model)) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = "grey55") +
    geom_line(linewidth = 0.9) +
    scale_colour_manual(values = roc_colors, name = NULL) +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
    coord_equal() +
    labs(title = "ROC Curves - Top 4 Models",
         subtitle = "Positive class: illicit transactions (Elliptic test set)",
         x = "False Positive Rate (1 - Specificity)",
         y = "True Positive Rate (Sensitivity)") +
    plot_theme +
    guides(colour = guide_legend(ncol = 2))

  ggsave(file.path(plots_dir, "roc_curves_comparison.png"), roc_plot,
         width = 8, height = 8, dpi = 300)
  message("Saved: outputs/plots/roc_curves_comparison.png")
  TRUE
}, error = function(e) {
  warning("ROC curve plot failed: ", conditionMessage(e))
  FALSE
})
record_plot("roc_curves_comparison.png", roc_success)

# ---------------------------------------------------------------------------
# 5b. Precision-Recall curves (all models overlaid, PR-AUC in the legend)
# ---------------------------------------------------------------------------

pr_success <- tryCatch({
  if (!prroc_available) stop("PRROC is not installed.")

  pr_frames <- list()
  pr_legend <- character(0)
  for (label in model_labels) {
    probabilities <- models[[label]]$probabilities
    pos_scores <- probabilities[y_test == "illicit"]
    neg_scores <- probabilities[y_test == "licit"]
    if (length(pos_scores) == 0 || length(neg_scores) == 0) {
      stop("Both classes must be present in the test set for ", label, ".")
    }
    pr_obj <- PRROC::pr.curve(scores.class0 = pos_scores,
                              scores.class1 = neg_scores,
                              curve = TRUE)
    legend_label <- sprintf("%s (PR-AUC = %.4f)", label, pr_obj$auc.integral)
    pr_legend[label] <- legend_label
    pr_frames[[label]] <- data.frame(
      recall = pr_obj$curve[, 1],
      precision = pr_obj$curve[, 2],
      model = legend_label,
      stringsAsFactors = FALSE
    )
    message(sprintf("PR-AUC   %-14s %.4f", label, pr_obj$auc.integral))
  }

  pr_df <- do.call(rbind, pr_frames)
  pr_df$model <- factor(pr_df$model, levels = unname(pr_legend[model_labels]))
  pr_colors <- setNames(unname(model_colors[model_labels]),
                        unname(pr_legend[model_labels]))

  baseline_rate <- mean(y_test == "illicit")

  pr_plot <- ggplot(pr_df, aes(x = recall, y = precision, colour = model)) +
    geom_hline(yintercept = baseline_rate, linetype = "dashed", colour = "grey55") +
    geom_line(linewidth = 0.9) +
    annotate("text", x = 0.02, y = baseline_rate + 0.03, hjust = 0, size = 3.2,
             colour = "grey35",
             label = sprintf("No-skill baseline = %.3f (illicit prevalence)", baseline_rate)) +
    scale_colour_manual(values = pr_colors, name = NULL) +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
    coord_equal() +
    labs(title = "Precision-Recall Curves - Top 4 Models",
         subtitle = "Positive class: illicit transactions (Elliptic test set)",
         x = "Recall", y = "Precision") +
    plot_theme +
    guides(colour = guide_legend(ncol = 2))

  ggsave(file.path(plots_dir, "pr_curves_comparison.png"), pr_plot,
         width = 8, height = 8, dpi = 300)
  message("Saved: outputs/plots/pr_curves_comparison.png")
  TRUE
}, error = function(e) {
  warning("Precision-Recall plot failed: ", conditionMessage(e))
  FALSE
})
record_plot("pr_curves_comparison.png", pr_success)

# ---------------------------------------------------------------------------
# 5c. Confusion matrix grid (2x2, counts and row percentages)
# ---------------------------------------------------------------------------

cm_success <- tryCatch({
  cm_frames <- list()
  for (label in model_labels) {
    predictions <- models[[label]]$predictions
    counts <- table(Actual = y_test, Predicted = predictions)
    cm <- as.data.frame(counts, stringsAsFactors = FALSE)
    names(cm) <- c("Actual", "Predicted", "Count")
    actual_totals <- tapply(cm$Count, cm$Actual, sum)
    cm$Percent <- 100 * cm$Count / as.numeric(actual_totals[cm$Actual])
    cm$model <- label
    cm_frames[[label]] <- cm
  }

  cm_df <- do.call(rbind, cm_frames)
  cm_df$model <- factor(cm_df$model, levels = model_labels)
  cm_df$Actual <- factor(cm_df$Actual, levels = rev(c("licit", "illicit")))
  cm_df$Predicted <- factor(cm_df$Predicted, levels = c("licit", "illicit"))
  cm_df$label_text <- sprintf("%d\n(%.1f%%)", cm_df$Count, cm_df$Percent)

  cm_plot <- ggplot(cm_df, aes(x = Predicted, y = Actual)) +
    geom_tile(aes(fill = model, alpha = Percent), colour = "white", linewidth = 1) +
    geom_text(aes(label = label_text), size = 3.6, colour = "grey15") +
    facet_wrap(~ model, nrow = 2) +
    scale_fill_manual(values = model_colors, guide = "none") +
    scale_alpha_continuous(range = c(0.12, 1), guide = "none") +
    labs(title = "Confusion Matrices - Top 4 Models",
         subtitle = "Cell labels: transaction count and percentage of the actual class (threshold = 0.5)",
         x = "Predicted class", y = "Actual class") +
    plot_theme +
    theme(panel.grid = element_blank(),
          strip.text = element_text(face = "bold"))

  ggsave(file.path(plots_dir, "confusion_matrices_grid.png"), cm_plot,
         width = 10, height = 9, dpi = 300)
  message("Saved: outputs/plots/confusion_matrices_grid.png")
  TRUE
}, error = function(e) {
  warning("Confusion matrix grid failed: ", conditionMessage(e))
  FALSE
})
record_plot("confusion_matrices_grid.png", cm_success)

# ---------------------------------------------------------------------------
# 5d. Feature importance grid (2x2, top 15 features per model)
# ---------------------------------------------------------------------------

# Return a data frame of feature / importance for a single model object, or raise
# an error when the object does not carry importance information.
extract_importance <- function(model) {
  if (inherits(model, "xgb.Booster")) {
    if (!xgboost_available) stop("xgboost is not installed.")
    imp <- as.data.frame(xgboost::xgb.importance(model = model))
    return(data.frame(feature = imp$Feature, importance = imp$Gain,
                      stringsAsFactors = FALSE))
  }
  if (inherits(model, "lgb.Booster")) {
    if (!lightgbm_available) stop("lightgbm is not installed.")
    imp <- as.data.frame(lightgbm::lgb.importance(model))
    return(data.frame(feature = imp$Feature, importance = imp$Gain,
                      stringsAsFactors = FALSE))
  }
  if (inherits(model, "ranger")) {
    imp <- model$variable.importance
    if (is.null(imp) || length(imp) == 0) {
      stop("the ranger model was trained without importance = 'impurity'")
    }
    return(data.frame(feature = names(imp), importance = as.numeric(imp),
                      stringsAsFactors = FALSE))
  }
  if (inherits(model, "train")) {
    if (!caret_available) stop("caret is not installed.")
    imp <- caret::varImp(model)$importance
    values <- if ("Overall" %in% names(imp)) imp$Overall else rowMeans(imp, na.rm = TRUE)
    return(data.frame(feature = rownames(imp), importance = as.numeric(values),
                      stringsAsFactors = FALSE))
  }
  if (inherits(model, "randomForest")) {
    if (!randomforest_available) stop("randomForest is not installed.")
    imp <- randomForest::importance(model)
    return(data.frame(feature = rownames(imp), importance = as.numeric(imp[, ncol(imp)]),
                      stringsAsFactors = FALSE))
  }
  stop("no importance extractor for class ", paste(class(model), collapse = "/"))
}

# The baseline ranger model (script 04) was fitted without importance = "impurity",
# so it stores no importance vector. Rather than dropping that panel, refit an
# equivalent ranger on train_selected purely to obtain impurity importances. This
# is an importance-only refit and is not used for any reported metric.
refit_ranger_importance <- function() {
  if (!ranger_available) stop("ranger is not installed.")
  message("Extra Trees: refitting ranger on train_selected to obtain impurity importances ...")
  conn <- DBI::dbConnect(RSQLite::SQLite(), dbname = db_path)
  train_df <- DBI::dbReadTable(conn, "train_selected")
  DBI::dbDisconnect(conn)
  train_df$class <- factor(train_df$class, levels = c("licit", "illicit"))
  fit <- ranger::ranger(dependent.variable.name = "class",
                        data = train_df[, c("class", feature_cols)],
                        probability = TRUE, importance = "impurity")
  imp <- fit$variable.importance
  data.frame(feature = names(imp), importance = as.numeric(imp), stringsAsFactors = FALSE)
}

fi_success <- tryCatch({
  fi_frames <- list()
  for (label in model_labels) {
    imp <- tryCatch(extract_importance(models[[label]]$model), error = function(e) {
      if (inherits(models[[label]]$model, "ranger")) {
        return(tryCatch(refit_ranger_importance(), error = function(e2) {
          warning(sprintf("Feature importance unavailable for %s: %s", label, conditionMessage(e2)))
          NULL
        }))
      }
      warning(sprintf("Feature importance unavailable for %s: %s", label, conditionMessage(e)))
      NULL
    })
    if (is.null(imp) || nrow(imp) == 0) next
    imp <- imp[is.finite(imp$importance), , drop = FALSE]
    if (nrow(imp) == 0) next
    # Scale to 0-100 within each model so the four panels are comparable in shape.
    max_importance <- max(imp$importance, na.rm = TRUE)
    if (max_importance > 0) imp$importance <- 100 * imp$importance / max_importance
    imp <- imp[order(-imp$importance), , drop = FALSE]
    imp <- head(imp, 15)
    imp$model <- label
    fi_frames[[label]] <- imp
  }

  if (length(fi_frames) == 0) stop("No model exposed usable feature importance values.")

  fi_df <- do.call(rbind, fi_frames)
  fi_df$model <- factor(fi_df$model, levels = model_labels[model_labels %in% fi_df$model])
  # Facet-local ordering without extra packages: make the axis key unique per
  # panel, order it globally, then strip the suffix when drawing the labels.
  fi_df$axis_key <- paste(fi_df$feature, fi_df$model, sep = "")
  fi_df <- fi_df[order(fi_df$model, fi_df$importance), , drop = FALSE]
  fi_df$axis_key <- factor(fi_df$axis_key, levels = fi_df$axis_key)

  fi_plot <- ggplot(fi_df, aes(x = axis_key, y = importance, fill = model)) +
    geom_col(width = 0.75) +
    coord_flip() +
    facet_wrap(~ model, nrow = 2, scales = "free") +
    scale_x_discrete(labels = function(x) sub(".*$", "", x)) +
    scale_fill_manual(values = model_colors, guide = "none") +
    labs(title = "Top 15 Feature Importances - Top 4 Models",
         subtitle = "Importance rescaled to 0-100 within each model",
         x = NULL, y = "Relative importance") +
    plot_theme +
    theme(strip.text = element_text(face = "bold"),
          axis.text.y = element_text(size = 8),
          panel.grid.major.y = element_blank())

  ggsave(file.path(plots_dir, "feature_importance_grid.png"), fi_plot,
         width = 11, height = 9, dpi = 300)
  message("Saved: outputs/plots/feature_importance_grid.png")
  TRUE
}, error = function(e) {
  warning("Feature importance grid failed: ", conditionMessage(e))
  FALSE
})
record_plot("feature_importance_grid.png", fi_success)

# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------

expected_plots <- c("roc_curves_comparison.png", "pr_curves_comparison.png",
                    "confusion_matrices_grid.png", "feature_importance_grid.png")

message("\n==================== VISUALISATION SUMMARY ====================")
message("Models used:")
for (label in model_labels) {
  message(sprintf("  %-14s %s version", label, models[[label]]$version))
}
message(sprintf("\nPlots created (%d of %d):", length(created_plots), length(expected_plots)))
if (length(created_plots) > 0) {
  for (p in created_plots) message("  [OK]     outputs/plots/", p)
}
if (length(failed_plots) > 0) {
  message("Plots NOT created:")
  for (p in failed_plots) message("  [FAILED] ", p)
}
message(sprintf("\nTotal runtime: %s", format(Sys.time() - script_start)))
message("==============================================================")
