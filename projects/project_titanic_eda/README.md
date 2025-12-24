# Project: Titanic Survival Forensics (EDA)

## 🎯 1. Problem Statement
**"Why did some survive while others perished?"**
Following the sinking of the RMS Titanic, investigators need a forensic data analysis to understand the systematic failure points of the event. This is not just about prediction; it is about **Exploratory Data Analysis (EDA)** to uncover the hidden biases in survival protocols.
*   **The Conflict:** The "Women and Children First" protocol was stated, but did it hold up across all socio-economic classes?
*   **The Goal:** Use rigorous, production-grade EDA to audit the passenger manifest, identify data quality issues, and mathematically quantify survival factors.

## 💾 2. Dataset Description
We use the classic **Titanic Dataset** (via Seaborn/Kaggle).
*   **Size:** 891 Passengers (Training set).
*   **Features:** Pclass (Status), Sex, Age, SibSp (Siblings), Parch (Parents), Fare, Embarked, Deck.
*   **Data Quality Issues:**
    *   **Deck:** 77% Missing (MNAR - Missing Not At Random).
    *   **Age:** 20% Missing (Critical for protocol analysis).
    *   This forces us to use advanced imputation strategies (Median by Class+Sex) rather than simple drops.

## 💡 3. Key Learnings
By completing this project, we demonstrated:
1.  **Automated Data Auditing:** We built a reusable `generate_data_quality_report` function to instantly flag nulls and cardinality issues, a standard production step.
2.  **Protocol Verification:** Visualizations confirmed that "Women and Children First" was strictly followed (Female survival ~74% vs Male <20%), but...
3.  **Socio-Economic Bias:** The "Class Divide" was lethal. 1st Class passengers had a >60% survival rate, compared to <25% for 3rd Class, highlighting a failure in emergency egress design for lower decks.
4.  **Forensic Imputation:** We showed that dropping rows with missing Age would have introduced bias, so we imputed using group medians to preserve the signal.

## 🚀 How to Run
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the forensic analysis:
    ```bash
    python notebooks/01_comprehensive_eda.py
    ```
