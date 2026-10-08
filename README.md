# Crypto Fraud Tracker

### Cryptocurrency Transaction Fraud Detection: A Multi-Algorithmic Approach to Anti-Money Laundering on the Bitcoin Network

Crypto Fraud Tracker is a machine learning-based project designed to identify potentially illicit Bitcoin transactions by analyzing transaction-level and graph-derived features. The project uses the **Elliptic Bitcoin Dataset**, a real-world cryptocurrency transaction graph containing labeled licit and illicit transactions.

The system is designed as a multi-stage pipeline covering data preparation, exploratory analysis, machine learning-based classification, and an interactive fraud detection dashboard.

---

## 📌 Project Overview

Cryptocurrencies provide fast and decentralized financial transactions, but their pseudonymous nature can also make them attractive for activities such as:

- Money laundering
- Ransomware payments
- Dark-web transactions
- Other forms of financial crime

This project aims to distinguish between **licit** and **illicit** Bitcoin transactions using machine learning techniques.

A major challenge is the severe class imbalance in the dataset, where illicit transactions represent only a small fraction of labeled transactions. Additionally, the transactions are distributed across sequential time steps, requiring a time-aware approach to prevent data leakage.

---

## 🎯 Objectives

The main objectives of the project are:

- Build a clean and reliable dataset from the raw Elliptic Bitcoin transaction graph.
- Handle severe class imbalance using **SMOTE (Synthetic Minority Over-sampling Technique)**.
- Preserve the temporal structure of the dataset during model training and evaluation.
- Perform exploratory data analysis to identify patterns associated with illicit transactions.
- Compare multiple machine learning algorithms for fraud detection.
- Evaluate models using metrics appropriate for highly imbalanced classification problems.
- Develop an interactive dashboard for fraud prediction and model explanations.

---

## 📊 Dataset

The project uses the **Elliptic Bitcoin Dataset**, a publicly available real-world cryptocurrency transaction dataset.

### Dataset Statistics

| Property | Value |
|---|---:|
| Transactions | 203,769 |
| Directed transaction edges | 234,355 |
| Time steps | 49 |
| Anonymized features | 165 |
| Labeled licit transactions | ~21% |
| Labeled illicit transactions | ~2% |
| Unlabeled transactions | ~77% |

The dataset consists of three primary files:

```text
elliptic_txs_features.csv
elliptic_txs_classes.csv
elliptic_txs_edgelist.csv
