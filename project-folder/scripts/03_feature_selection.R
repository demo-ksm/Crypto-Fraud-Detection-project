# scripts/03_feature_selection.R
# Purpose: Remove highly correlated (redundant) features from the training
# dataset to reduce multicollinearity and dimensionality before model
# training. This script uses correlation analysis on the SMOTE-balanced
# `transactions_train` table and deterministically marks features to drop.

library(DBI)
library(RSQLite)

# Connect to the SQLite DB
db_path <- file.path("db", "fraud_detection.db")
if (!file.exists(db_path)) stop("Database not found at: ", db_path)
conn <- dbConnect(RSQLite::SQLite(), dbname = db_path)

# Read transactions_train into memory
train_df <- dbReadTable(conn, "transactions_train")

# Identify feature columns (all except time_step and class)
feature_cols <- setdiff(names(train_df), c("time_step", "class"))
message("Number of feature columns:")
print(length(feature_cols))  # expected 165

# Build numeric matrix for correlation (coerce to numeric safely)
# Use pairwise.complete.obs to handle any missing values.
feat_mat <- as.matrix(sapply(train_df[feature_cols], function(x) as.numeric(as.character(x))))
cor_mat <- cor(feat_mat, use = "pairwise.complete.obs")

# Find pairs with absolute correlation > 0.9 (exclude self-correlation)
thresh <- 0.9
pairs_idx <- which(abs(cor_mat) > thresh & abs(cor_mat) < 1, arr.ind = TRUE)

# Build pairs dataframe and normalize ordering so feature_1 < feature_2 alphabetically
if (nrow(pairs_idx) == 0) {
  pairs_df <- data.frame(feature_1 = character(0), feature_2 = character(0), correlation = numeric(0), stringsAsFactors = FALSE)
} else {
  pairs_raw <- data.frame(
    feature_1 = rownames(cor_mat)[pairs_idx[,1]],
    feature_2 = colnames(cor_mat)[pairs_idx[,2]],
    correlation = cor_mat[pairs_idx],
    stringsAsFactors = FALSE
  )

  # For each row, ensure feature_1 < feature_2 alphabetically; swap if necessary
  swap_idx <- which(pairs_raw$feature_1 > pairs_raw$feature_2)
  if (length(swap_idx) > 0) {
    tmp <- pairs_raw$feature_1[swap_idx]
    pairs_raw$feature_1[swap_idx] <- pairs_raw$feature_2[swap_idx]
    pairs_raw$feature_2[swap_idx] <- tmp
    pairs_raw$correlation[swap_idx] <- pairs_raw$correlation[swap_idx]  # correlation symmetric
  }

  # Remove duplicated mirrored pairs
  pairs_df <- unique(pairs_raw)

  # Remove self-pairs if any linger
  pairs_df <- pairs_df[pairs_df$feature_1 != pairs_df$feature_2, , drop = FALSE]
}

# Sort by absolute correlation descending and show top 20
pairs_df$abs_cor <- abs(pairs_df$correlation)
pairs_sorted <- pairs_df[order(-pairs_df$abs_cor, pairs_df$feature_1, pairs_df$feature_2), c("feature_1", "feature_2", "correlation")]
message("HIGHLY CORRELATED PAIRS (>0.9):")
print(head(pairs_sorted, 20))

# From each correlated pair, mark the SECOND feature (feature_2) to be dropped
# Using the deterministic ordering above ensures reproducible picks.
if (nrow(pairs_sorted) == 0) {
  to_drop <- character(0)
} else {
  to_drop <- unique(pairs_sorted$feature_2)
}

# Print counts before/after
message("Number of features to drop:")
print(length(to_drop))
message("Reduced feature count (original - dropped):")
print(length(feature_cols) - length(to_drop))

# Create reduced feature list
feature_cols_reduced <- setdiff(feature_cols, to_drop)

# Read transactions_test
test_df <- dbReadTable(conn, "transactions_test")

# Build final dataframes
train_final <- train_df[, c("time_step", "class", feature_cols_reduced), drop = FALSE]
# test_df has txId
test_final <- test_df[, c("txId", "time_step", "class", feature_cols_reduced), drop = FALSE]

# Write new tables to DB without modifying original tables
dbWriteTable(conn, "train_selected", train_final, overwrite = TRUE)
dbWriteTable(conn, "test_selected",  test_final,  overwrite = TRUE)

# Confirm tables
message("Tables in DB after writing selected sets:")
print(dbListTables(conn))

# Disconnect
dbDisconnect(conn)

message("Feature selection complete. train_selected and test_selected created in DB.")
