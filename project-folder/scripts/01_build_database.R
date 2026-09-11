# scripts/01_build_database.R
# Build a SQLite database from the provided Elliptic CSV files.
# This script is idempotent and can be run top-to-bottom without manual input.

# Load required libraries for DB handling
# - DBI: generic database interface
# - RSQLite: SQLite backend implementation for DBI
setwd("C:/Users/91876/OneDrive/Desktop/PROJECTS/pds_project/project-folder")
library(DBI)
library(RSQLite)

# Read the CSV files into data frames
# Use relative paths so the script runs from the project root.
# Do NOT modify the CSV files themselves; we only read them.
df_train <- read.csv("data/elliptic_train_balanced.csv", stringsAsFactors = FALSE)
df_test  <- read.csv("data/elliptic_test.csv", stringsAsFactors = FALSE)

# IMPORTANT: Ensure both dataframes use the same target column name `class`.
# The training file may have the target named `label` instead of `class`.
# We prefer `class` to match the test set and to simplify downstream code.
if ("label" %in% names(df_train)) {
  # Rename `label` to `class` in df_train
  names(df_train)[names(df_train) == "label"] <- "class"
  message("Renamed column 'label' to 'class' in df_train")
}

# Print column names so you can visually verify alignment between train/test
message("Column names in df_train:")
print(names(df_train))
message("Column names in df_test:")
print(names(df_test))

# Print dimensions to confirm expected sizes
message("Dimensions:")
print(dim(df_train))  # expected approx 39648 x 167
print(dim(df_test))   # expected approx 16670 x 169

# Print time_step ranges to confirm train/test split by time
if ("time_step" %in% names(df_train)) {
  message("Range of time_step in df_train:")
  print(range(df_train$time_step, na.rm = TRUE))
} else {
  message("time_step column missing in df_train")
}

if ("time_step" %in% names(df_test)) {
  message("Range of time_step in df_test:")
  print(range(df_test$time_step, na.rm = TRUE))
} else {
  message("time_step column missing in df_test")
}

# Print class distributions for simple sanity checks
if ("class" %in% names(df_train)) {
  message("Class distribution in df_train:")
  print(table(df_train$class))
} else {
  message("class column missing in df_train")
}

if ("class" %in% names(df_test)) {
  message("Class distribution in df_test:")
  print(table(df_test$class))
} else {
  message("class column missing in df_test")
}

# Prepare SQLite database path
# We remove any existing DB so the script is re-runnable and idempotent.
db_path <- file.path("db", "fraud_detection.db")
if (file.exists(db_path)) {
  message("Existing database detected at ", db_path, "; removing to ensure a fresh build.")
  file.remove(db_path)
}

# Connect to a new SQLite database (file will be created)
# Using a file-backed SQLite DB keeps outputs small and portable.
conn <- dbConnect(RSQLite::SQLite(), dbname = db_path)

# Write two separate tables: transactions_train and transactions_test
# Rationale: training (SMOTE-balanced) and test (real-world distribution)
# have different distributions and column sets; keeping them separate
# prevents accidental mixing and makes experiments reproducible.
dbWriteTable(conn, "transactions_train", df_train, overwrite = TRUE)
dbWriteTable(conn, "transactions_test",  df_test,  overwrite = TRUE)

# Verify database contents and basic metadata
message("Database tables:")
print(dbListTables(conn))

message("Row counts:")
print(dbGetQuery(conn, "SELECT COUNT(*) AS n FROM transactions_train"))
print(dbGetQuery(conn, "SELECT COUNT(*) AS n FROM transactions_test"))

message("Time step ranges in DB:")
print(dbGetQuery(conn, "SELECT MIN(time_step) AS min_ts, MAX(time_step) AS max_ts FROM transactions_train"))
print(dbGetQuery(conn, "SELECT MIN(time_step) AS min_ts, MAX(time_step) AS max_ts FROM transactions_test"))

# Disconnect to release the DB file handle
dbDisconnect(conn)

message("Database build complete. DB file path: ", db_path)
