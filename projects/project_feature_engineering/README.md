# Project: Advanced Feature Engineering

## 🎯 1. Problem Statement
**"Better Data > Better Algorithms."**
Real-world data is rarely ready for machine learning. In this project, we predict **Used Car Prices**, a task plagued by messy data types holding the most value.
*   **The Challenge:**
    *   **High Cardinality:** Variables like `Car_Model` (e.g., "Toyota Camry LE") and `Zip_Code` have hundreds of categories. One-Hot Encoding would create a sparse, massive dataset.
    *   **Hidden Interactions:** The depreciation of a car isn't just about Age, and isn't just about Mileage. It's about `Age * Mileage` (Usage Intensity).
*   **The Goal:** Beat the baseline model's error rate by **50%+** purely through clever feature engineering, without changing the algorithm.

## 💾 2. Dataset Description
We generated a **Synthetic Used Car Dataset** to mimic web-scraped listings (e.g., Craigslist/Autotrader).
*   **Size:** 50,000 Car Listings.
*   **Features:** Brand, Model (500+ types), Year, Mileage, ListingDate, ZipCode.
*   **Target:** Price ($).
*   **Why Synthetic?** Real scraped data is often proprietary or requires massive cleaning. This dataset captures the *mathematical relationships* (depreciation curves, seasonality) perfectly for demonstration.

## 💡 3. Key Learnings
By completing this project, we learned:
1.  **Target Encoding Wins:** For high-cardinality features like `Car_Model`, Target Encoding (Mean Encoding) provided a massive accuracy lift compared to Label Encoding or One-Hot Encoding.
2.  **Interaction Terms Matter:** Creating a custom feature `Miles_Per_Year` captured the "wear and tear" signal better than raw mileage.
3.  **Time is a Feature:** Extracting `Car_Age` and `Seasonality` (Month) from a date string turned a useless text column into a powerful predictor.
4.  **Engineering ROI:** We reduced the RMSE (Error) from ~$6,000 to ~$3,500 just by processing the data differently.

## 🚀 How to Run
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the pipeline:
    ```bash
    python notebooks/01_feature_engineering_pipeline.py
    ```
