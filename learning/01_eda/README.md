# Module 1: Exploratory Data Analysis (EDA)

## Overview
Exploratory Data Analysis (EDA) is the detective work of data science. Before applying any machine learning algorithms, you must understand the nature, quality, and relationships within your data.

This module is broken down into specific topics to guide you through the EDA process.

---

## 📚 Contents

### 1. [Understanding Data Types](./01_data_types.md)
*   Difference between Numerical (Continuous/Discrete) and Categorical (Nominal/Ordinal) data.
*   How to handle and encode them using Pandas.

### 2. [Handling Missing Data](./02_missing_data.md)
*   **Identification:** Detecting missing patterns (MCAR, MAR, MNAR).
*   **Imputation:** Techniques ranging from simple Mean/Mode filling to advance KNN/MICE imputation.

### 3. [Detecting & Handling Outliers](./03_outlier_detection.md)
*   **Detection:** Z-Score, IQR Method, and Isolation Forest.
*   **Handling:** When to drop, cap (Winsorize), or transform (Log) outliers.

### 4. [Data Visualization](./04_visualization.md)
*   **Univariate:** Histograms, Boxplots, Countplots.
*   **Bivariate:** Scatterplots, Correlation Heatmaps, Violin plots.
*   **Multivariate:** Adding dimensions with hue and size.

### 5. [EDA Workflow Checklist](./05_eda_workflow.md)
*   A step-by-step checklist to ensure a rigorous and consistent analysis process for every project.

### 6. [EDA Strategy: Regression vs Classification](./06_regression_vs_classification.md)
*   How to tailor your visual analysis based on your prediction goal (Continuous vs Categorical target).
*   Key plots: Scatter/Correlation (Regression) vs Boxplots/Class Balance (Classification).

### 7. [Handling Class Imbalance](./07_handling_class_imbalance.md)
*   **The Trap:** Why Accuracy fails for imbalanced data.
*   **The Fix:** SMOTE (Synthetic Over-sampling), Undersampling, and Class Weights.
*   **The Metrics:** Precision, Recall, F1-Score, ROC-AUC.

### 8. [Automated EDA](./08_automated_eda.md)
*   Generate comprehensive HTML reports in 1 line of code using **ydata-profiling**, **Sweetviz**, and **Klib**.

### 9. [Feature Engineering from EDA](./09_feature_engineering_from_eda.md)
*   Transforming insights into model improvements.
*   Binning, Log Transforms, Interactions, and Grouping rare labels.

### 10. [Advanced High-Dimensional EDA](./10_advanced_eda_techniques.md)
*   How to visualize datasets with 100+ columns.
*   **Dimensionality Reduction:** PCA and t-SNE visualizations.

### 11. [EDA Corner Cases & Gotchas](./11_corner_cases.md)
*   15+ edge cases that trip up data scientists.
*   **High Cardinality**, Data Leakage, Temporal Violations, Mixed Types, Silent Nulls.
*   Production-ready checklist.

---

## 🛠️ Key Libraries
- **Pandas:** Data manipulation.
- **Matplotlib / Seaborn:** Static visualization.
- **Missingno:** Missing data visualization.
- **Scikit-learn:** Imputation (`SimpleImputer`, `KNNImputer`) and anomaly detection (`IsolationForest`).

---

## 📋 Specialized EDA Techniques
Core EDA is complete. For domain-specific techniques (Time Series, Geospatial, NLP, Big Data, Dashboards), see the **[specialized/](./specialized/)** folder.

---

## 🚀 Next Steps
Start by understanding your data types in **[01_data_types.md](./01_data_types.md)**, or jump to the **[Workflow Checklist](./05_eda_workflow.md)** if you are starting a new analysis.
