# Telco Customer Churn Dataset - Data Dictionary

## Dataset Overview

- **Source:** IBM Sample Data Sets (via Kaggle)
- **URL:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Size:** 7,043 rows × 21 columns
- **Target Variable:** Churn (Yes/No)

---

## Features

### Customer Demographics

| Feature | Type | Description | Values |
|---------|------|-------------|--------|
| `customerID` | String | Unique identifier | e.g., "7590-VHVEG" |
| `gender` | Categorical | Customer gender | Male, Female |
| `SeniorCitizen` | Binary | Whether customer is 65+ | 0 (No), 1 (Yes) |
| `Partner` | Categorical | Has partner/spouse | Yes, No |
| `Dependents` | Categorical | Has dependents | Yes, No |

### Service Information

| Feature | Type | Description | Values |
|---------|------|-------------|--------|
| `tenure` | Numerical | Months with company | 0-72 |
| `PhoneService` | Categorical | Has phone service | Yes, No |
| `MultipleLines` | Categorical | Multiple phone lines | Yes, No, No phone service |
| `InternetService` | Categorical | Internet service type | DSL, Fiber optic, No |
| `OnlineSecurity` | Categorical | Online security add-on | Yes, No, No internet service |
| `OnlineBackup` | Categorical | Online backup service | Yes, No, No internet service |
| `DeviceProtection` | Categorical | Device protection plan | Yes, No, No internet service |
| `TechSupport` | Categorical | Tech support service | Yes, No, No internet service |
| `StreamingTV` | Categorical | TV streaming service | Yes, No, No internet service |
| `StreamingMovies` | Categorical | Movie streaming service | Yes, No, No internet service |

### Account Information

| Feature | Type | Description | Values |
|---------|------|-------------|--------|
| `Contract` | Categorical | Contract term | Month-to-month, One year, Two year |
| `PaperlessBilling` | Categorical | Uses paperless billing | Yes, No |
| `PaymentMethod` | Categorical | Payment method | Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) |
| `MonthlyCharges` | Numerical | Current monthly charge | $18.25 - $118.75 |
| `TotalCharges` | Numerical | Total charges to date | $18.80 - $8,684.80 |

### Target Variable

| Feature | Type | Description | Values |
|---------|------|-------------|--------|
| `Churn` | Binary | Customer churned | Yes, No |

---

## Data Quality Notes

### Missing Values
- `TotalCharges`: 11 missing values (customers with 0 tenure)
- **Handling:** Impute with 0 or remove rows (minimal impact)

### Data Types
- `TotalCharges` stored as object (should be numeric)
- **Preprocessing needed:** Convert to float after handling missing values

### Business Logic
- `tenure = 0` indicates new customers (same month signup)
- `TotalCharges ≈ MonthlyCharges × tenure` (with some variation)

---

## Statistical Characteristics

### Numerical Features

| Feature | Mean | Std | Min | 25% | 50% | 75% | Max |
|---------|------|-----|-----|-----|-----|-----|-----|
| `tenure` | 32.4 | 24.6 | 0 | 9 | 29 | 55 | 72 |
| `MonthlyCharges` | 64.76 | 30.09 | 18.25 | 35.50 | 70.35 | 89.85 | 118.75 |
| `TotalCharges` | 2283.3 | 2266.8 | 18.8 | 401.4 | 1397.5 | 3794.7 | 8684.8 |

### Categorical Features

**Most Common Values:**
- `Contract`: Month-to-month (55.0%)
- `PaymentMethod`: Electronic check (33.6%)
- `InternetService`: Fiber optic (43.9%)

### Target Distribution

- **Churn = Yes:** 1,869 (26.5%)
- **Churn = No:** 5,174 (73.5%)
- **Class imbalance:** Moderate (worth addressing in modeling)

---

## Files

### Raw Data
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` - Original dataset from Kaggle

### Processed Data (Generated during analysis)
- `telco_churn_cleaned.csv` - After data cleaning
- `telco_churn_features.csv` - With engineered features
- `train_data.csv` - Training set (70%)
- `test_data.csv` - Test set (30%)

---

## Feature Engineering Ideas

1. **Tenure Bins:** Group tenure into categories (0-12, 13-24, 25-48, 49+ months)
2. **Charge per Month:** TotalCharges / tenure (average spending)
3. **Service Count:** Number of additional services subscribed
4. **Has Internet:** Binary flag for any internet service
5. **Contract Value:** Combine contract type with charges
6. **Payment Reliability:** Automatic vs manual payment methods

---

## Usage Notes

```python
import pandas as pd

# Load data
df = pd.read_csv('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Basic exploration
print(df.shape)  # (7043, 21)
print(df['Churn'].value_counts())
print(df.dtypes)
```

---

**Last Updated:** December 2024
