# Automated Feature Engineering: When Manual Fails

**Banking Fraud Detection - The Scale Problem:**

**Your data:**
- Customers table: 50M rows
- Transactions table: 8B rows  
- Cards table: 120M rows
- Merchants table: 2M rows

**Manual feature engineering:**
- Data scientist creates 40 features over 2 weeks
- Examples: `avg_transaction_amount`, `transactions_last_30d`
- Model AUC: 0.82

**The Reality:** There are thousands of hidden patterns buried in those relationships.

---

## Featuretools: Deep Feature Synthesis

**What it does:** Automatically generates aggregations across table relationships

```python
import featuretools as ft

# Define relationships
es = ft.EntitySet(id="transactions")
es = es.add_dataframe(dataframe_name="customers", dataframe=customers, index="customer_id")
es = es.add_dataframe(dataframe_name="transactions", dataframe=transactions, index="transaction_id")
es = es.add_relationship("customers", "customer_id", "transactions", "customer_id")

# Generate 1,847 features in 12 minutes
feature_matrix, features = ft.dfs(
    entityset=es,
    target_dataframe_name="transactions",
    max_depth=2
)
```

**Auto-generated features that beat human intuition:**
- `STD(transactions.amount) WHERE merchant_category='gas_station'`
- `MODE(transactions.day_of_week) WHERE amount > 100`  
- `COUNT(transactions) / TIME_SINCE(customer.signup_date)`

**Result after feature selection (Boruta):**
- Selected 127 features from 1,847 candidates
- Model AUC: 0.82 → **0.91**
- Caught $18M in additional fraud annually

---

## 1. Featuretools (Deep Feature Synthesis)

The King of Relational AutoFE.
It works on **Tables** (Parent/Child relationships).
*   **Customers Table** (Parent)
*   **Transactions Table** (Child)

You want to predict Churn for a Customer. But the useful data is in Transactions.
**DFS (Deep Feature Synthesis)** crawls the relationship:
1.  `SUM(transactions.amount)` -> Total Spend.
2.  `MEAN(transactions.amount)` -> Avg Spend.
3.  `STD(transactions.amount)` -> Volatility.
4.  `MODE(transactions.day)` -> Favorite shopping day.

It does this recursively.

```python
import featuretools as ft

# 1. Define EntitySet
es = ft.EntitySet(id="customer_data")
es = es.add_dataframe(dataframe_name="customers", dataframe=customers_df, index="customer_id")
es = es.add_dataframe(dataframe_name="transactions", dataframe=transactions_df, index="trans_id")

# 2. Define Relationship
new_relationship = ft.Relationship(es, "customers", "transactions", "customer_id", "customer_id")
es = es.add_relationship(new_relationship)

# 3. Magic
feature_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name="customers")
```

---

## 2. Tsfresh (Time Series Feature Extraction)

If you have a sensor signal (Voltage, Vibration), `tsfresh` extracts ~800 features instantly.
*   Fourier Transform coefficients.
*   Entropy.
*   Peaks.
*   Autocorrelation.

**Use Case:** Predictive Maintenance. (Is this engine vibrating weirdly?)

```python
from tsfresh import extract_features
extracted_features = extract_features(timeseries, column_id="id", column_sort="time")
```

---

## 3. The Warning Label ⚠️

AutoFE is a bazooka.
*   **Explosion:** It creates 1,000+ features easily.
*   **Overfitting:** Most will be noise.
*   **Requirement:** You **MUST** pair AutoFE with strong **Feature Selection** (Module 6.03).
*   **Interpretability:** "What does `SUM(STD(transactions))` mean?" It can be hard to explain to a boss.
