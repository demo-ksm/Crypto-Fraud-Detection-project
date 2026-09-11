Project-folder
===============

Cryptocurrency Fraud Detection (DA2 assignment)

This project uses the Elliptic dataset to build models for detecting fraudulent cryptocurrency transactions. The repository is organized to separate raw data, scripts to build a reproducible SQLite database, outputs for later model results, and documentation.

How to run
----------
1. Ensure the two CSV files are present in the `data/` directory:
   - `data/elliptic_train_balanced.csv`
   - `data/elliptic_test.csv`
2. Open `scripts/01_build_database.R` in RStudio or VS Code and run it top-to-bottom.
   - The script will create (or recreate) `db/fraud_detection.db` and populate two tables:
     `transactions_train` and `transactions_test`.

Note about committing the DB
---------------------------
The script generates `db/fraud_detection.db`. If the repository is large you may want to add this file to `.gitignore`. If the DB file is small and convenient to share, you can commit it instead. Decide based on your repository and GitHub storage policy.

Folder structure
----------------
project-folder/
├── data/
│   ├── elliptic_train_balanced.csv   (already exists — DO NOT modify)
│   └── elliptic_test.csv             (already exists — DO NOT modify)
├── db/
│   └── (empty for now — `fraud_detection.db` will be created by the script)
├── scripts/
│   └── 01_build_database.R
├── outputs/
│   └── (empty for now — model outputs, plots, results go here)
├── docs/
│   └── (empty for now — report and documentation files)
└── README.md
