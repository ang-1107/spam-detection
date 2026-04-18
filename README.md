# SMS Spam Detection

An end-to-end **machine learning + NLP** pipeline that classifies SMS messages as **ham** (legitimate) or **spam** using classical text features and scikit-learn. The default model pairs **TF–IDF** vectorization with **Multinomial Naive Bayes**, with optional benchmarking across many other vectorizers and classifiers.

---

## Contents

- [About the project](#about-the-project)
- [Features](#features)
- [Dataset](#dataset)
- [Pipeline overview](#pipeline-overview)
- [Project layout](#project-layout)
- [Getting started (step by step)](#getting-started-step-by-step)
  - [1. Clone the repository](#1-clone-the-repository)
  - [2. Create and activate a virtual environment](#2-create-and-activate-a-virtual-environment)
  - [3. Install dependencies](#3-install-dependencies)
  - [4. Configure Kaggle API access](#4-configure-kaggle-api-access)
  - [5. Download the dataset (optional)](#5-download-the-dataset-optional)
  - [6. Train the default model](#6-train-the-default-model)
  - [7. Run the Streamlit web app](#7-run-the-streamlit-web-app)
  - [8. Run CLI prediction](#8-run-cli-prediction)
  - [9. Run the benchmarking script](#9-run-the-benchmarking-script)
- [Outputs and artifacts](#outputs-and-artifacts)
- [Author](#author)

---

## About the project

This project demonstrates a complete **text classification** workflow for SMS spam detection: ingesting raw labeled messages, normalizing text with **NLTK**, turning text into **numerical features** (TF–IDF), training a **probabilistic classifier** (Multinomial Naive Bayes), and serving predictions through a **Streamlit** UI or a small **command-line** tool.

A separate **benchmark** script automates preprocessing ablations, vectorizer choices (bag-of-words vs TF–IDF, n-grams, vocabulary size), and many scikit-learn models, so you can compare approaches on the same dataset splits.

---

## Features

- **NLTK preprocessing:** lowercasing, tokenization, alphanumeric filtering, English stopword removal, punctuation handling, **Porter stemming**.
- **TF–IDF features** via `sklearn.feature_extraction.text.TfidfVectorizer` (training script); benchmarking also explores `CountVectorizer`.
- **Default classifier:** `sklearn.naive_bayes.MultinomialNB` with saved artifacts for reproducible inference.
- **Automatic dataset fetch:** training and benchmarking can trigger a **Kaggle download** if `data/raw/spam.csv` is missing (credentials required).
- **Encoding handling:** converts the Kaggle CSV from **Windows-1252** to **UTF-8** before cleaning and modeling.
- **Streamlit app** for interactive classification.
- **CLI** for quick scripted predictions.
- **Benchmark runner** for large-scale experiments and optional hyperparameter grids (including XGBoost when available).

---

## Dataset

The SMS Spam Collection dataset is used (UCI / Kaggle mirror):

- **[SMS Spam Collection Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)**

After download, the pipeline expects **`spam.csv`** under **`data/raw/`** (created automatically by the included download script or by the training/benchmark entrypoints when the file is absent).

---

## Pipeline overview

The flow from raw data to a prediction is:

1. **Acquire data**  
   `scripts/download_dataset.py` (or automatic download from `train_model.py` / `benchmark_models.py`) pulls the dataset from Kaggle and unzips it so **`data/raw/spam.csv`** exists.

2. **Encoding**  
   `src/data.py` reads `spam.csv` as **Windows-1252** and writes **`data/raw/spam_utf8.csv`** in **UTF-8** for consistent downstream I/O.

3. **Cleaning & labels**  
   The same module loads the UTF-8 CSV, drops unused columns if present, renames message/label columns to **`text`** and **`target`**, **label-encodes** `ham` / `spam` to **`0` / `1`**, and removes duplicate rows.

4. **Text normalization (NLP)**  
   `src/preprocess.py` implements **`transform_text`**: tokenize with NLTK, keep alphanumeric tokens, remove stopwords and punctuation tokens, apply **Porter stemming**, and join tokens back into a single string.

5. **Feature extraction**  
   `src/features.py` applies `transform_text` to every row (column **`transformed_text`**) and fits a **`TfidfVectorizer`** on the training corpus (sparse matrices internally; same modeling behavior as dense for this linear generative model).

6. **Training & evaluation**  
   `src/train.py` splits data (with **stratification** when the split is valid for all classes), fits **MultinomialNB**, reports **accuracy**, **precision** (spam = positive class), and a **confusion matrix**, then saves:
   - **`vectorizer.pkl`**
   - **`model.pkl`**

7. **Inference**  
   `src/inference.py` loads both pickles, runs **`transform_text`** on new user text, **transforms** with the saved vectorizer, and returns **`0`** (ham) or **`1`** (spam). The Streamlit app (`app/app.py`) and CLI (`app/predict_cli.py`) call this layer.

8. **Benchmarking (optional)**  
   `scripts/benchmark_models.py` repeats variations of preprocessing, vectorizers, and models on the same dataset, logs metrics, and writes **`benchmark_results.csv`** (path configurable). **XGBoost** runs only if the `xgboost` package is installed.

---

## Project layout

| Path | Purpose |
|------|---------|
| `src/data.py` | Encoding conversion, load/clean DataFrame, label encoding |
| `src/preprocess.py` | NLTK downloads + `transform_text` |
| `src/features.py` | Build `transformed_text`, fit TF–IDF |
| `src/train.py` | Train MultinomialNB, metrics, save pickles |
| `src/inference.py` | Load artifacts, `predict_text` |
| `scripts/download_dataset.py` | Download & unzip Kaggle dataset → `data/raw/spam.csv` |
| `scripts/train_model.py` | CLI entry: ensure data → train → write `vectorizer.pkl` / `model.pkl` |
| `scripts/benchmark_models.py` | Grid/ablation experiments → CSV results |
| `app/app.py` | Streamlit UI |
| `app/predict_cli.py` | CLI for single-message prediction |
| `requirements.txt` | Python dependencies |

---

## Getting started (step by step)

Run these commands from a terminal. On Windows, **PowerShell** or **Command Prompt** is fine. On Linux or macOS, use **bash** (adjust paths only where noted).

### 1. Clone the repository

```bash
git clone https://github.com/ang-1107/spam-detection.git
cd spam-detection
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```cmd
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs **NumPy**, **pandas**, **scikit-learn**, **NLTK**, **Streamlit**, **kaggle**, and **xgboost**.

### 4. Configure Kaggle API access

The dataset is downloaded via the **Kaggle API**. Configure **one** of the following (required once per machine):

**Option A — `kaggle.json` (recommended)**

- Download your API credentials from your Kaggle account.
- Place the file at:
  - **Linux / macOS:** `~/.kaggle/kaggle.json`
  - **Windows:** `%USERPROFILE%\.kaggle\kaggle.json`

**Option B — environment variables**

- `KAGGLE_USERNAME`
- `KAGGLE_KEY`

See Kaggle’s documentation for creating API tokens. The dataset page is here: [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

### 5. Download the dataset (optional)

Training and benchmarking **automatically** download the data if `data/raw/spam.csv` is missing. To download **explicitly**:

```bash
python scripts/download_dataset.py
```

Useful options:

```bash
python scripts/download_dataset.py --out-dir data/raw
python scripts/download_dataset.py --force
```

`--force` re-downloads even if files already exist.

### 6. Train the default model

From the **repository root** (`spam-detection/`):

```bash
python scripts/train_model.py
```

Common options:

```bash
python scripts/train_model.py --test-size 0.2 --random-state 2
python scripts/train_model.py --vectorizer my_vectorizer.pkl --model my_model.pkl
python scripts/train_model.py --input data/raw/spam.csv --utf8 data/raw/spam_utf8.csv
```

This writes **`vectorizer.pkl`** and **`model.pkl`** in the current working directory unless you pass custom paths.

### 7. Run the Streamlit web app

With the virtual environment activated and still at the repo root:

```bash
streamlit run app/app.py
```

Open the URL Streamlit prints in the browser (usually `http://localhost:8501`), enter a message, and click **Classify**.

### 8. Run CLI prediction

From the repo root, after training has produced the default artifact names:

```bash
python app/predict_cli.py --text "Congratulations, you've won a prize!"
```

Custom artifact paths:

```bash
python app/predict_cli.py --text "Hello, are we still on for lunch?" --vectorizer vectorizer.pkl --model model.pkl
```

**Output:** prints **`1`** for spam and **`0`** for ham (consistent with label encoding in `src/data.py`).

### 9. Run the benchmarking script

From the repo root:

```bash
python scripts/benchmark_models.py
```

This uses **`data/raw/spam.csv`** and **`data/raw/spam_utf8.csv`** by default (and downloads the raw CSV via Kaggle if needed). To limit runtime during exploration:

```bash
python scripts/benchmark_models.py --max-runs 50 --out-csv benchmark_results.csv --metric precision
```

**Useful flags:**

| Flag | Description |
|------|-------------|
| `--input` | Path to raw `spam.csv` (default: `data/raw/spam.csv`) |
| `--utf8` | Path for converted UTF-8 CSV (default: `data/raw/spam_utf8.csv`) |
| `--test-size` | Holdout fraction (default: `0.2`) |
| `--random-state` | Random seed for splits (default: `2`) |
| `--max-runs` | Cap number of runs (`0` = no cap) |
| `--out-csv` | Results table path (default: `benchmark_results.csv`) |
| `--metric` | `precision`, `f1`, or `accuracy` for selecting the “best” run |

---

## Outputs and artifacts

| File | Description |
|------|-------------|
| `data/raw/spam.csv` | Raw Kaggle export (Windows-1252) |
| `data/raw/spam_utf8.csv` | UTF-8 copy used for pandas/sklearn |
| `vectorizer.pkl` | Fitted `TfidfVectorizer` |
| `model.pkl` | Fitted `MultinomialNB` classifier |
| `benchmark_results.csv` | Benchmark runs (when using `benchmark_models.py`) |

---

## Author

Angel Mandhwani | IIT Kharagpur

Email: [angelmandhwani@gmail.com]

