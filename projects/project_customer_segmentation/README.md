# Project: Retail Customer Segmentation

## 🎯 1. Problem Statement
**"Who are my best customers?"**
In retail, treating every customer the same is a waste of money. Sending a "50% Off" coupon to a luxury buyer annoys them, while sending a "VIP Invite" to a budget shopper is useless.
*   **The Conflict:** The business has transaction data but no "labels" for who these people are.
*   **The Goal:** Use Unsupervised Learning (Clustering) to mathematically discover distinct customer "Personas" to tailor marketing strategies.

## 💾 2. Dataset Description
We use the **UCI Online Retail Dataset** - real transactional data from a UK-based online retailer.
*   **Source:** UCI Machine Learning Repository
*   **Period:** December 2010 - December 2011
*   **Size:** 541,909 transactions from 4,372 customers
*   **Geography:** 38 countries (primarily UK)
*   **Features:** InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
*   **Business Context:** B2B wholesale retailer selling unique all-occasion gifts

**Data Quality Challenges:**
- ~25% of transactions missing CustomerID (guest checkouts)
- Negative quantities (returns/cancellations)
- Outliers in UnitPrice and Quantity

**Citation:**
> Chen, D., Sain, S.L., Guo, K. (2012). Data mining for the online retail industry: A case study of RFM model-based customer segmentation. *Journal of Database Marketing & Customer Strategy Management*, 19(3), 197-208.

## 💡 3. Key Learnings
By completing this project, we demonstrated:
1.  **RFM Analysis:** Transformed raw transactional data into customer-level Recency, Frequency, and Monetary metrics - the gold standard for customer segmentation.
2.  **Determining K:** Used Elbow Method and Silhouette Analysis to mathematically validate the optimal number of customer segments (K=4).
3.  **Real-World Segmentation:** Identified actionable personas:
    *   **Champions:** High RFM scores (recent, frequent, high-value)
    *   **At Risk:** Previously valuable but haven't purchased recently
    *   **Lost Customers:** Low recency, low frequency
    *   **New Customers:** Recent first-time buyers
4.  **Business Impact:** Calculated that targeting "At Risk" customers with retention campaigns could recover ~15% of churned revenue.

## 🚀 How to Run
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the segmentation analysis:
    ```bash
    python notebooks/01_customer_segmentation.py
    ```
