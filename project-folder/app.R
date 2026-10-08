# app.R
# Interactive dashboard for the Elliptic cryptocurrency fraud detection project.
#
# HOW TO RUN
# ----------
# From R or RStudio, with the working directory set to project-folder/:
#
#     shiny::runApp()
#
# or explicitly:
#
#     shiny::runApp("C:/Users/91876/OneDrive/Desktop/PROJECTS/pds_project/project-folder")
#
# From a terminal:
#
#     Rscript -e "shiny::runApp('project-folder', launch.browser = TRUE)"
#
# Required packages: shiny, bslib, plotly, DT, DBI, RSQLite, pROC, PRROC,
# plus the model packages (xgboost, lightgbm, ranger, randomForest, caret).
# Install any that are missing with, for example:
#
#     install.packages(c("shiny", "bslib", "plotly", "DT"))
#
# The first launch scores all four models on the test set and caches the result
# to outputs/dashboard_data.rds (about 30 seconds). Later launches read the
# cache and start immediately. Delete that file to force a rebuild after
# retraining any model.

library(shiny)
library(bslib)
library(plotly)
library(DT)

# The model packages must have their namespaces loaded before any saved model is
# scored: predict() dispatches on S3 methods (predict.ranger, predict.lgb.Booster
# and so on) that are only registered once the namespace is loaded. Without this,
# loading a ranger object fails with "no applicable method for 'predict'".
for (pkg in c("DBI", "RSQLite", "pROC", "PRROC",
              "xgboost", "lightgbm", "ranger", "randomForest", "caret")) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    warning(sprintf("Package '%s' is not installed; models that depend on it will be skipped.", pkg))
  }
}

# ===========================================================================
# GLOBAL: data preparation (runs once when the app starts)
# ===========================================================================

APP_DIR <- getwd()
CACHE_PATH <- file.path("outputs", "dashboard_data.rds")

MODEL_SPEC <- list(
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

# One colour per model, shared by every chart in the dashboard (RColorBrewer Set2).
MODEL_COLORS <- c(
  "XGBoost"       = "#66C2A5",
  "LightGBM"      = "#FC8D62",
  "Extra Trees"   = "#8DA0CB",
  "Random Forest" = "#E78AC3"
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

extract_importance <- function(model, feature_cols, db_path) {
  if (inherits(model, "xgb.Booster")) {
    imp <- as.data.frame(xgboost::xgb.importance(model = model))
    return(data.frame(feature = imp$Feature, importance = imp$Gain, stringsAsFactors = FALSE))
  }
  if (inherits(model, "lgb.Booster")) {
    imp <- as.data.frame(lightgbm::lgb.importance(model))
    return(data.frame(feature = imp$Feature, importance = imp$Gain, stringsAsFactors = FALSE))
  }
  if (inherits(model, "ranger")) {
    imp <- model$variable.importance
    if (is.null(imp) || length(imp) == 0) {
      # The DA2 baseline ranger was fitted without importance = "impurity", so an
      # equivalent model is refitted once here purely to populate this chart.
      message("Refitting ranger for importance (one-off, cached afterwards) ...")
      conn <- DBI::dbConnect(RSQLite::SQLite(), dbname = db_path)
      train_df <- DBI::dbReadTable(conn, "train_selected")
      DBI::dbDisconnect(conn)
      train_df$class <- factor(train_df$class, levels = c("licit", "illicit"))
      fit <- ranger::ranger(dependent.variable.name = "class",
                            data = train_df[, c("class", feature_cols)],
                            probability = TRUE, importance = "impurity")
      imp <- fit$variable.importance
    }
    return(data.frame(feature = names(imp), importance = as.numeric(imp), stringsAsFactors = FALSE))
  }
  if (inherits(model, "train")) {
    imp <- caret::varImp(model)$importance
    values <- if ("Overall" %in% names(imp)) imp$Overall else rowMeans(imp, na.rm = TRUE)
    return(data.frame(feature = rownames(imp), importance = as.numeric(values), stringsAsFactors = FALSE))
  }
  if (inherits(model, "randomForest")) {
    imp <- randomForest::importance(model)
    return(data.frame(feature = rownames(imp), importance = as.numeric(imp[, ncol(imp)]),
                      stringsAsFactors = FALSE))
  }
  NULL
}

# Build everything the dashboard needs, or read it back from the cache.
build_dashboard_data <- function() {
  if (file.exists(CACHE_PATH)) {
    message("Loading cached dashboard data from ", CACHE_PATH)
    return(readRDS(CACHE_PATH))
  }

  message("Building dashboard data (first run) ...")
  db_path <- file.path("db", "fraud_detection.db")
  if (!file.exists(db_path)) {
    stop("Database not found at ", db_path,
         ". Run the app with the working directory set to project-folder/.")
  }

  conn <- DBI::dbConnect(RSQLite::SQLite(), dbname = db_path)
  test_df <- DBI::dbReadTable(conn, "test_selected")
  DBI::dbDisconnect(conn)

  test_df$class <- factor(test_df$class, levels = c("licit", "illicit"))
  feature_cols <- setdiff(names(test_df), c("time_step", "class", "txId"))
  X_test <- as.data.frame(lapply(test_df[, feature_cols, drop = FALSE],
                                 function(x) as.numeric(as.character(x))))
  names(X_test) <- feature_cols
  y_binary <- as.numeric(test_df$class == "illicit")

  models <- list()
  for (spec in MODEL_SPEC) {
    candidates <- list()
    if (file.exists(spec$tuned))    candidates[[length(candidates) + 1]] <- list(version = "tuned", path = spec$tuned)
    if (file.exists(spec$original)) candidates[[length(candidates) + 1]] <- list(version = "original", path = spec$original)

    for (candidate in candidates) {
      entry <- tryCatch({
        model <- readRDS(candidate$path)
        probabilities <- predict_illicit_probability(model, X_test)
        importance <- tryCatch(extract_importance(model, feature_cols, db_path),
                               error = function(e) NULL)
        list(label = spec$label, version = candidate$version,
             probabilities = probabilities, importance = importance)
      }, error = function(e) {
        warning(sprintf("Could not load %s (%s): %s", spec$label, candidate$version,
                        conditionMessage(e)))
        NULL
      })
      if (!is.null(entry)) {
        models[[spec$label]] <- entry
        message(sprintf("  %-14s -> %s version", spec$label, candidate$version))
        break
      }
    }
  }

  if (length(models) == 0) stop("No model objects could be loaded.")

  # Curves and confusion matrices, precomputed once per model.
  for (label in names(models)) {
    probabilities <- models[[label]]$probabilities
    predicted <- as.numeric(probabilities >= 0.5)

    roc_obj <- pROC::roc(y_binary, probabilities, quiet = TRUE,
                         direction = "<", levels = c(0, 1))
    models[[label]]$roc <- data.frame(fpr = 1 - roc_obj$specificities,
                                      tpr = roc_obj$sensitivities)
    models[[label]]$roc_auc <- as.numeric(pROC::auc(roc_obj))

    pr_obj <- PRROC::pr.curve(scores.class0 = probabilities[y_binary == 1],
                              scores.class1 = probabilities[y_binary == 0],
                              curve = TRUE)
    models[[label]]$pr <- data.frame(recall = pr_obj$curve[, 1],
                                     precision = pr_obj$curve[, 2])
    models[[label]]$pr_auc <- pr_obj$auc.integral

    tp <- sum(predicted == 1 & y_binary == 1)
    fp <- sum(predicted == 1 & y_binary == 0)
    fn <- sum(predicted == 0 & y_binary == 1)
    tn <- sum(predicted == 0 & y_binary == 0)
    models[[label]]$confusion <- data.frame(
      actual    = c("licit", "licit", "illicit", "illicit"),
      predicted = c("licit", "illicit", "licit", "illicit"),
      count     = c(tn, fp, fn, tp),
      stringsAsFactors = FALSE
    )
    models[[label]]$precision <- if ((tp + fp) > 0) tp / (tp + fp) else NA_real_
    models[[label]]$recall    <- if ((tp + fn) > 0) tp / (tp + fn) else NA_real_
    models[[label]]$f1 <- if (!is.na(models[[label]]$precision) && !is.na(models[[label]]$recall) &&
                              (models[[label]]$precision + models[[label]]$recall) > 0) {
      2 * models[[label]]$precision * models[[label]]$recall /
        (models[[label]]$precision + models[[label]]$recall)
    } else NA_real_
  }

  comparison_path <- file.path("outputs", "model_comparison_results.csv")
  comparison <- if (file.exists(comparison_path)) {
    read.csv(comparison_path, stringsAsFactors = FALSE)
  } else {
    warning("outputs/model_comparison_results.csv not found; the table tab will be empty.")
    data.frame()
  }

  stats_path <- file.path("outputs", "final_statistical_comparison.csv")
  statistics <- if (file.exists(stats_path)) read.csv(stats_path, stringsAsFactors = FALSE) else NULL

  out <- list(
    models = models,
    comparison = comparison,
    statistics = statistics,
    n_transactions = nrow(X_test),
    n_illicit = sum(y_binary),
    n_licit = sum(y_binary == 0),
    n_features = length(feature_cols),
    time_step_range = range(test_df$time_step, na.rm = TRUE)
  )

  if (!dir.exists("outputs")) dir.create("outputs", recursive = TRUE)
  saveRDS(out, CACHE_PATH)
  message("Cached dashboard data to ", CACHE_PATH)
  out
}

DASH <- build_dashboard_data()
MODEL_CHOICES <- names(DASH$models)

# Best model per metric, computed over the FINAL model set (tuned where DA2
# produced a tuned object). The 11-model comparison CSV is not used here because
# it holds the untuned baseline figures only, which would report a different
# winner from the rest of the dashboard.
best_by <- function(metric) {
  field <- switch(metric, f1 = "f1", roc = "roc_auc", pr = "pr_auc")
  values <- vapply(DASH$models, function(m) {
    v <- m[[field]]
    if (is.null(v) || !is.finite(v)) NA_real_ else v
  }, numeric(1))
  if (all(is.na(values))) return(list(name = "-", value = NA_real_))
  idx <- which.max(values)
  list(name = names(values)[idx], value = unname(values[idx]))
}

BEST_F1  <- best_by("f1")
BEST_ROC <- best_by("roc")
BEST_PR  <- best_by("pr")

# ===========================================================================
# UI
# ===========================================================================

ui <- page_sidebar(
  title = "Elliptic Cryptocurrency Fraud Detection - Model Dashboard",
  theme = bs_theme(version = 5, bootswatch = "flatly"),
  # fillable = FALSE lets the main column flow at its natural height and scroll
  # with the browser window. Left at the default, bslib squeezes every card into
  # one viewport-height fill container with its own inner scrollbars.
  fillable = FALSE,

  sidebar = sidebar(
    width = 320,
    selectInput("model", "Select model",
                choices = MODEL_CHOICES,
                selected = if ("LightGBM" %in% MODEL_CHOICES) "LightGBM" else MODEL_CHOICES[1]),
    uiOutput("model_version_note"),
    hr(),
    h6("Test set"),
    tags$ul(
      tags$li(sprintf("%s transactions", format(DASH$n_transactions, big.mark = ","))),
      tags$li(sprintf("%s illicit (%.1f%%)", format(DASH$n_illicit, big.mark = ","),
                      100 * DASH$n_illicit / DASH$n_transactions)),
      tags$li(sprintf("%s licit (%.1f%%)", format(DASH$n_licit, big.mark = ","),
                      100 * DASH$n_licit / DASH$n_transactions)),
      tags$li(sprintf("%d features", DASH$n_features)),
      tags$li(sprintf("time_step %d-%d", DASH$time_step_range[1], DASH$time_step_range[2]))
    ),
    hr(),
    p(class = "text-muted small",
      "Decision threshold fixed at 0.5. Tuned model objects are used where DA2 ",
      "produced one; otherwise the original baseline object is used.")
  ),

  layout_columns(
    fill = FALSE,
    value_box(title = "Transactions analysed",
              value = format(DASH$n_transactions, big.mark = ","),
              showcase = icon("database"), theme = "primary"),
    value_box(title = "Class distribution",
              value = sprintf("%.1f%% illicit", 100 * DASH$n_illicit / DASH$n_transactions),
              showcase = icon("scale-unbalanced"), theme = "secondary",
              p(sprintf("%s illicit / %s licit",
                        format(DASH$n_illicit, big.mark = ","),
                        format(DASH$n_licit, big.mark = ",")))),
    value_box(title = "Best by F1",
              value = BEST_F1$name, showcase = icon("trophy"), theme = "success",
              p(sprintf("F1 = %.4f", BEST_F1$value))),
    value_box(title = "Best by ROC-AUC",
              value = BEST_ROC$name, showcase = icon("chart-line"), theme = "info",
              p(sprintf("ROC-AUC = %.4f", BEST_ROC$value))),
    value_box(title = "Best by PR-AUC",
              value = BEST_PR$name, showcase = icon("bullseye"), theme = "warning",
              p(sprintf("PR-AUC = %.4f", BEST_PR$value)))
  ),

  p(class = "text-muted small mb-2",
    "\"Best\" is taken over the four final models shown in this dashboard, using the ",
    "tuned object where DA2 produced one. The 'All models' tab reports the untuned ",
    "baseline figures for all 11 algorithms."),

  navset_card_tab(
    id = "tabs",

    # Each card is given an explicit height: inside bslib's fill layout a card
    # whose body has no intrinsic height collapses to nothing and the plot is
    # never drawn.
    nav_panel(
      "Model performance",
      layout_columns(
        col_widths = c(6, 6),
        row_heights = list("420px"),
        card(height = "420px", full_screen = TRUE,
             card_header("ROC curve"),
             card_body(plotlyOutput("roc_plot", height = "100%"), padding = 8)),
        card(height = "420px", full_screen = TRUE,
             card_header("Precision-Recall curve"),
             card_body(plotlyOutput("pr_plot", height = "100%"), padding = 8))
      ),
      layout_columns(
        col_widths = c(7, 5),
        row_heights = list("400px"),
        card(height = "400px", full_screen = TRUE,
             card_header("Confusion matrix"),
             card_body(plotlyOutput("confusion_plot", height = "100%"), padding = 8)),
        card(height = "400px",
             card_header("Metrics for the selected model"),
             card_body(tableOutput("model_metrics")))
      )
    ),

    nav_panel(
      "Feature importance",
      card(
        height = "600px", full_screen = TRUE,
        card_header("Top 15 features"),
        card_body(plotlyOutput("importance_plot", height = "100%"), padding = 8),
        card_footer(class = "text-muted small",
                    "Importance is rescaled to 0-100 within the selected model. Gain is used ",
                    "for the boosting models and impurity decrease for the tree ensembles, so ",
                    "values are comparable in rank but not on an absolute scale across models.")
      )
    ),

    nav_panel(
      "All models",
      card(
        height = "620px",
        card_header("Comparison of all 11 trained models"),
        card_body(DTOutput("comparison_table")),
        card_footer(class = "text-muted small",
                    "Source: outputs/model_comparison_results.csv. Sort any column or use the ",
                    "search box to filter. AdaBoost has no ROC-AUC/PR-AUC because the adabag ",
                    "prediction object did not return class probabilities.")
      )
    ),

    nav_panel(
      "Statistical comparison",
      card(
        height = "620px",
        card_header("Bootstrap 95% confidence intervals and pairwise significance"),
        card_body(DTOutput("statistics_table")),
        card_footer(class = "text-muted small",
                    "Source: outputs/final_statistical_comparison.csv, produced by ",
                    "scripts/07_final_statistical_analysis.R. Run that script if this tab is empty.")
      )
    )
  )
)

# ===========================================================================
# SERVER
# ===========================================================================

server <- function(input, output, session) {

  selected <- reactive({
    req(input$model)
    DASH$models[[input$model]]
  })

  model_color <- reactive({
    unname(MODEL_COLORS[input$model])
  })

  output$model_version_note <- renderUI({
    m <- selected()
    tags$p(class = "small text-muted",
           sprintf("Using the %s model object.", m$version))
  })

  # --- ROC curve -----------------------------------------------------------
  output$roc_plot <- renderPlotly({
    m <- selected()
    plot_ly() |>
      add_lines(x = c(0, 1), y = c(0, 1), line = list(dash = "dash", color = "grey"),
                name = "No skill", hoverinfo = "skip") |>
      add_lines(data = m$roc, x = ~fpr, y = ~tpr,
                line = list(color = model_color(), width = 2.5),
                name = sprintf("%s (AUC = %.4f)", input$model, m$roc_auc),
                hovertemplate = "FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>") |>
      layout(xaxis = list(title = "False Positive Rate", range = c(0, 1)),
             yaxis = list(title = "True Positive Rate", range = c(0, 1)),
             legend = list(orientation = "h", x = 0.25, y = -0.2),
             margin = list(t = 30))
  })

  # --- Precision-Recall curve ---------------------------------------------
  output$pr_plot <- renderPlotly({
    m <- selected()
    baseline <- DASH$n_illicit / DASH$n_transactions
    plot_ly() |>
      add_lines(x = c(0, 1), y = c(baseline, baseline),
                line = list(dash = "dash", color = "grey"),
                name = sprintf("No skill (%.3f)", baseline), hoverinfo = "skip") |>
      add_lines(data = m$pr, x = ~recall, y = ~precision,
                line = list(color = model_color(), width = 2.5),
                name = sprintf("%s (PR-AUC = %.4f)", input$model, m$pr_auc),
                hovertemplate = "Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>") |>
      layout(xaxis = list(title = "Recall", range = c(0, 1)),
             yaxis = list(title = "Precision", range = c(0, 1)),
             legend = list(orientation = "h", x = 0.15, y = -0.2),
             margin = list(t = 30))
  })

  # --- Confusion matrix ----------------------------------------------------
  output$confusion_plot <- renderPlotly({
    m <- selected()
    cm <- m$confusion
    # Percentage of the actual class, so the two rows are each read as 100%.
    totals <- tapply(cm$count, cm$actual, sum)
    cm$percent <- 100 * cm$count / as.numeric(totals[cm$actual])
    cm$label <- sprintf("%s<br>(%.1f%%)", format(cm$count, big.mark = ","), cm$percent)

    x_levels <- c("licit", "illicit")
    y_levels <- c("illicit", "licit")
    z <- matrix(NA_real_, nrow = 2, ncol = 2)
    text_matrix <- matrix("", nrow = 2, ncol = 2)
    for (i in seq_len(nrow(cm))) {
      r <- match(cm$actual[i], y_levels)
      c_ <- match(cm$predicted[i], x_levels)
      z[r, c_] <- cm$percent[i]
      text_matrix[r, c_] <- cm$label[i]
    }

    plot_ly(x = x_levels, y = y_levels, z = z, type = "heatmap",
            colorscale = list(c(0, "#FFFFFF"), c(1, model_color())),
            showscale = FALSE,
            hovertemplate = "Actual: %{y}<br>Predicted: %{x}<br>%{z:.1f}% of actual class<extra></extra>") |>
      # x/y/text are matched element-wise. text_matrix is indexed [row = y_levels,
      # col = x_levels], and as.vector() unrolls it column-major, so the x and y
      # vectors must vary y fastest to stay aligned with it.
      add_annotations(
        x = rep(x_levels, each = length(y_levels)),
        y = rep(y_levels, times = length(x_levels)),
        text = as.vector(text_matrix),
        showarrow = FALSE, font = list(size = 15, color = "#222222")
      ) |>
      layout(xaxis = list(title = "Predicted class", side = "bottom"),
             yaxis = list(title = "Actual class"),
             margin = list(t = 30))
  })

  # --- Metrics for the selected model -------------------------------------
  output$model_metrics <- renderTable({
    m <- selected()
    data.frame(
      Metric = c("Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC", "Model object"),
      Value = c(sprintf("%.4f", m$precision), sprintf("%.4f", m$recall),
                sprintf("%.4f", m$f1), sprintf("%.4f", m$roc_auc),
                sprintf("%.4f", m$pr_auc), m$version),
      stringsAsFactors = FALSE
    )
  }, striped = TRUE, width = "100%")

  # --- Feature importance --------------------------------------------------
  output$importance_plot <- renderPlotly({
    m <- selected()
    validate(need(!is.null(m$importance) && nrow(m$importance) > 0,
                  "Feature importance is not available for this model object."))

    imp <- m$importance
    imp <- imp[is.finite(imp$importance), , drop = FALSE]
    max_importance <- max(imp$importance, na.rm = TRUE)
    if (max_importance > 0) imp$importance <- 100 * imp$importance / max_importance
    imp <- head(imp[order(-imp$importance), , drop = FALSE], 15)
    imp <- imp[order(imp$importance), , drop = FALSE]
    imp$feature <- factor(imp$feature, levels = imp$feature)

    plot_ly(imp, x = ~importance, y = ~feature, type = "bar", orientation = "h",
            marker = list(color = model_color()),
            hovertemplate = "%{y}: %{x:.1f}<extra></extra>") |>
      layout(xaxis = list(title = "Relative importance (0-100)"),
             yaxis = list(title = ""),
             margin = list(l = 90, t = 30))
  })

  # --- All-models comparison table ----------------------------------------
  output$comparison_table <- renderDT({
    validate(need(nrow(DASH$comparison) > 0, "outputs/model_comparison_results.csv was not found."))
    df <- DASH$comparison
    names(df) <- c("Model", "Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC")[seq_along(df)]
    datatable(df, rownames = FALSE, filter = "top",
              options = list(pageLength = 11, dom = "ftip", order = list(list(3, "desc")))) |>
      formatRound(columns = c("Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC"), digits = 4)
  })

  # --- Statistical comparison table ---------------------------------------
  output$statistics_table <- renderDT({
    validate(need(!is.null(DASH$statistics) && nrow(DASH$statistics) > 0,
                  "Run scripts/07_final_statistical_analysis.R to generate this table."))
    df <- DASH$statistics
    datatable(df, rownames = FALSE, filter = "top",
              options = list(pageLength = 12, dom = "ftip", scrollX = TRUE)) |>
      formatRound(columns = intersect(c("point_estimate", "ci_lower", "ci_upper", "bootstrap_se"),
                                      names(df)), digits = 4)
  })
}

shinyApp(ui, server)
