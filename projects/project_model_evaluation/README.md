# Project: Advanced Model Evaluation & Business Impact Analysis

## 🎯 1. Problem Statement
**"Accuracy is not enough."**
In high-stakes industries like banking, a model with 95% accuracy can still bankrupt a company. This project simulates a **Credit Risk** scenario where a bank needs to approve or reject loan applications.

*   **The Conflict:** 
    *   Approving a bad loan costs **$50,000** (Default).
    *   Rejecting a good loan costs **$2,000** in lost profit.
    *   Standard metrics like Accuracy and F1-score treat these two errors equally. The business does not.
*   **The Goal:** Optimize the model threshold to maximize **Net Profit**, not just statistical metrics.

## 💾 2. Dataset Description
We generated a **Synthetic Credit Risk Dataset** to ensure a controlled experimental environment.
*   **Size:** 100,000 Loan Applications.
*   **Imbalance:** 95% Good Loans / 5% Defaults (Realistic imbalance).
*   **Features:** 20 numerical features (representing income, credit score, debt-to-income ratio, etc.).
*   **Why Synthetic?** It allows us to mathematically prove the "Ground Truth" and cost calculations without noise from dirty data.

## 💡 3. Key Learnings
By completing this project, we demonstrated:
1.  **The Accuracy Trap:** A naive model achieved 95% accuracy by simply rejecting everyone (or accepting everyone), but provided $0 value or massive loss.
2.  **Profit Curves > ROC Curves:** We plotted `Profit vs. Threshold` to find that the optimal decision threshold (e.g., 0.15) was very different from the default (0.5).
3.  **Model Calibration:** We used `CalibratedClassifierCV` to verify that when the model says "70% Risk", it actually means 70% of those loans default. This is critical for risk pricing.
4.  **Cost-Sensitive Learning:** We proved that minimizing False Negatives (Defaults) was 25x more important than False Positives.

## 🚀 How to Run
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the analysis script:
    ```bash
    python notebooks/01_advanced_evaluation_analysis.py
    ```

## 📊 Key Concepts Demonstrated
1.  **Profit Curve:** Plotting Potential Profit vs. Decision Threshold.
2.  **Calibration:** Using `CalibratedClassifierCV` to fix overconfident models.
3.  **Confusion Matrix:** Interpreted with dollar signs, not just counts.
