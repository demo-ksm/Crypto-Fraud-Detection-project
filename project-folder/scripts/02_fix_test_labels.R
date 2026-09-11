# scripts/02_fix_test_labels.R
# Fix class labels in transactions_test table of db/fraud_detection.db
# Reason: The training and test tables must use the same class encoding
# (e.g., "illicit"/"licit") before training/evaluation. The test table
# currently uses numeric codes (1/2); we convert them to the string labels
# to match the training table and avoid mismatches during modeling.

# Load database libraries
library(DBI)
library(RSQLite)

# Path to the SQLite DB (relative to project root)
db_path <- file.path("db", "fraud_detection.db")
if (!file.exists(db_path)) {
  stop("Database not found at: ", db_path, " — ensure you run this from the project root and that the DB exists.")
}

# Connect to DB
conn <- dbConnect(RSQLite::SQLite(), dbname = db_path)

# Read transactions_test into memory
# Use dbReadTable so we preserve all columns as stored
df_test <- dbReadTable(conn, "transactions_test")

# Print class distribution BEFORE any change
message("BEFORE FIX:")
print(table(df_test$class, useNA = "ifany"))

# Recode class column safely and idempotently:
# - If class values are numeric (1/2), convert 1 -> "illicit", 2 -> "licit"
# - If already character with expected labels, leave as-is
# - Preserve any unexpected values unchanged (but report them)

# Ensure we operate on character representation for mapping
orig_class <- df_test$class
char_class <- as.character(orig_class)

# Determine if mapping is needed
needs_mapping <- all(char_class %in% c("1", "2")) || is.numeric(orig_class) || is.integer(orig_class)

if (needs_mapping) {
  df_test$class <- ifelse(as.integer(char_class) == 1, "illicit",
                          ifelse(as.integer(char_class) == 2, "licit", char_class))
  message("Recode applied: numeric 1/2 -> 'illicit'/'licit'")
} else {
  message("No numeric mapping needed; leaving `class` values as-is")
}

# Print class distribution AFTER change
message("AFTER FIX:")
print(table(df_test$class, useNA = "ifany"))

# Write the fixed table back to the DB (overwrite)
# This permanently replaces the transactions_test table with the adjusted labels.
dbWriteTable(conn, "transactions_test", df_test, overwrite = TRUE)

# Re-query the DB for final confirmation (from DB-side)
message("Final counts from DB — transactions_test:")
print(dbGetQuery(conn, "SELECT class, COUNT(*) AS n FROM transactions_test GROUP BY class ORDER BY class"))

message("Counts from DB — transactions_train:")
print(dbGetQuery(conn, "SELECT class, COUNT(*) AS n FROM transactions_train GROUP BY class ORDER BY class"))

# Disconnect
dbDisconnect(conn)

message("Fix script completed. The transactions_test table now uses 'illicit'/'licit' labels where applicable.")
