# Project: Real Estate Valuation (Regression Pipeline)

## 🎯 1. Problem Statement
**"What is a home actually worth?"**
In real estate, underpricing leads to lost revenue, while overpricing leads to stagnant inventory. Simple averages don't work because location, size, and age interact in complex ways.
*   **The Conflict:** A Linear Regression model is interpretable but fails to capture complex non-linear price dynamics (e.g., location premiums).
*   **The Goal:** Build a robust, scalable prediction pipeline that outperforms simple baselines and identifies key price drivers.

## 💾 2. Dataset Description
We use the **California Housing Dataset** (Standard Scikit-Learn dataset).
*   **Size:** 20,640 Districts.
*   **Features:** MedInc (Median Income), HouseAge, AveRooms, Bedrooms, Population, Latitude, Longitude.
*   **Target:** Median House Value ($100k units).
*   **Data Characteristics:** Contains outliers and distinct geographic clusters, making it perfect for testing tree-based models vs. linear models.

## 💡 3. Key Learnings
By completing this project, we demonstrated:
1.  **Pipeline Architecture:** We used `sklearn.pipeline` to prevent data leakage, ensuring that scaling and imputation only happened within the cross-validation folds.
2.  **Algorithm Selection:** We benchmarked 3 models:
    *   **Linear Regression:** RMSE = 0.72 (Baseline).
    *   **Ridge Regression:** RMSE = 0.72 (No improvement, features aren't collinear enough).
    *   **XGBoost:** RMSE = 0.47 (**35% Error Reduction**).
3.  **Hyperparameter Tuning:** Using `GridSearchCV`, we optimized the tree depth and learning rate to balance bias and variance.
4.  **Residual Analysis:** We validated the final model by checking that residuals were mostly homoscedastic (random), confirming the model's reliability.

## 🚀 How to Run
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the regression pipeline:
    ```bash
    python notebooks/01_housing_regression.py
    ```
