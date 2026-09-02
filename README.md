# Mushroom Classification

An end-to-end machine learning project that predicts whether a mushroom is **edible (`e`)** or **poisonous (`p`)** based on its physical characteristics.

## Project Overview

This project uses the classic [UCI Mushroom dataset](https://archive.ics.uci.edu/ml/datasets/mushroom) to classify mushrooms. The core deliverable is the Jupyter notebook `mushroom_classification.ipynb`, which walks through a complete ML pipeline:

1. **Data Cleaning & Preprocessing** — load and inspect the data, handle missing/placeholder values, and encode categorical variables.
2. **Exploratory Data Analysis (EDA)** — examine class balance and rank how strongly each feature separates edible vs. poisonous mushrooms.
3. **Feature Engineering & Selection** — one-hot encode features and use chi-squared `SelectKBest` to reduce dimensionality.
4. **Model Selection** — compare several classifier families with 5-fold stratified cross-validation.
5. **Model Training** — train the shortlisted models on the full training set.
6. **Model Evaluation & Tuning** — evaluate on a held-out test set and tune the best model with `GridSearchCV`.

## Dataset

- **File:** `data/mushrooms.csv`
- **Rows:** 8,124
- **Columns:** 22 categorical descriptive features + 1 target (`class`)
- **Target:** `class`, where `e` = edible and `p` = poisonous

All features are categorical (single-letter codes). The dataset is nearly balanced (~52% edible / ~48% poisonous).

Key characteristics discovered during analysis:
- `odor` alone almost perfectly separates edible from poisonous mushrooms.
- `spore-print-color`, `gill-color`, and `ring-type` are also highly discriminative.
- Missing `stalk-root` values are encoded with the placeholder `"?"` rather than `NaN`.
- The `veil-type` column has only a single unique value and is dropped as uninformative.

## Models

The notebook compares the following classifiers using 5-fold cross-validation:

- Logistic Regression (interpretable linear baseline)
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machine (RBF kernel)

**Best model:** Random Forest, tuned via `GridSearchCV`, achieves very high accuracy and recall on the held-out test set.

## Metric of Interest

Because misclassifying a **poisonous** mushroom as edible is the costly error, **recall on the poisonous class** is the primary metric of interest — even more important than overall accuracy.

## Getting Started

1. Open the notebook:

   ```bash
   jupyter notebook mushroom_classification.ipynb
   ```

2. The notebook requires the following Python libraries:
   - `numpy`, `pandas`
   - `matplotlib`, `seaborn`
   - `scikit-learn`

## Project Structure

```
mushroom_classification/
├── README.md
├── mushroom_classification.ipynb   # End-to-end analysis & modeling notebook
└── data/
    └── mushrooms.csv               # UCI Mushroom dataset
```

## Caveat

Because this dataset separates the classes almost perfectly on a handful of features, near-100% scores here reflect the dataset's structure, not a guarantee of real-world performance. A food-safety model like this must always be validated against fresh, independently collected data before being trusted for any real decision.
