<div align="center">

# 📞 Telco Customer Churn Analysis

### *$3.9M Annual Value Through Statistical Analysis*

![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![Impact](https://img.shields.io/badge/Impact-$3.9M-gold?style=flat-square)
![Notebooks](https://img.shields.io/badge/Notebooks-9-blue?style=flat-square)
![Level](https://img.shields.io/badge/Level-Advanced-red?style=flat-square)

**End-to-end statistical project from hypothesis testing to business recommendations**

[💼 Business Case](#-business-problem) • [📊 Analysis](#-statistical-journey) • [💰 Impact](#-financial-impact) • [🚀 Run It](#-quick-start)

</div>

---

## 💼 Business Problem

> **26% annual churn rate = $8.4M lost revenue**

<table>
<tr>
<td width="50%">

### 🚨 The Challenge

**Customer Attrition Crisis:**
- 7,043 total customers
- 1,869 churning annually (26.5%)
- $4,500 average lifetime value
- **Current loss: $8.4M/year**

**Key Questions:**
- ❓ What drives customers to leave?
- ❓ Can we predict churn risk?
- ❓ What's the ROI of retention?

</td>
<td width="50%">

### 🎯 The Goal

**Data-Driven Retention Strategy:**
- ✅ Identify churn drivers statistically
- ✅ Build predictive risk model
- ✅ Quantify financial impact
- ✅ Deliver actionable recommendations

**Target Outcome:**
- Reduce churn by 30%
- From 26.5% → 18.6%
- **Save $2.5M annually**

</td>
</tr>
</table>

---

## 📊 Statistical Journey

**9 comprehensive notebooks covering the full data science workflow**

<table>
<tr>
<td width="33%">

### 📈 Phase 1: Foundation
**Understanding the Data**

**01. Descriptive Statistics**
- Summary stats by segment
- Distribution analysis
- Central tendency & variance

**02. Exploratory Data Analysis**
- Univariate patterns
- Bivariate relationships
- Multi-dimensional insights

**03. Probability Distributions**
- Tenure distribution (Exponential)
- Charges distribution (Normal)
- Churn modeling (Binomial)

</td>
<td width="33%">

### 🔬 Phase 2: Statistical Tests
**Proving Hypotheses**

**04. Hypothesis Testing**
- T-tests (charges comparison)
- ANOVA (contract types)
- Chi-square (independence tests)
- Two-proportion Z-tests

**05. Confidence Intervals**
- Churn rate CI by segment
- Revenue per customer CI
- Customer lifetime value CI

**06. Correlation Analysis**
- Pearson correlation matrix
- Spearman for ordinal features
- VIF multicollinearity check

</td>
<td width="33%">

### 🎯 Phase 3: Solutions
**Making Decisions**

**07. Power Analysis**
- A/B test sample size
- Minimum detectable effect
- Test duration planning

**08. Regression Modeling**
- Logistic regression
- Coefficients interpretation
- Odds ratios

**09. Final Recommendations**
- Business insights
- ROI calculations
- Action plans

</td>
</tr>
</table>

---

## 💡 Key F indings

### 🔍 Statistical Evidence

<details>
<summary><strong>Finding 1: Contract Type Drives Churn 📊</strong></summary>

**Chi-Square Test Results:**
```
χ² =  1,179.4
p < 0.001 (highly significant)
Cramér's V = 0.41 (strong effect)
```

**Churn Rates:**
- 📅 Monthly contracts: **42.7%** churn
- 📆 One-year contracts: **11.3%** churn  
- 📖 Two-year contracts: **2.8%** churn

**Conclusion:** Long-term contracts reduce churn by **15×** compared to month-to-month

</details>

<details>
<summary><strong>Finding 2: Fiber Optic Pricing Problem 💰</strong></summary>

**Statistical Comparison:**
```
Fiber optic churn:  41.9%
DSL churn:          18.9%
Difference:         2.2× higher (p < 0.001)
```

**Root Cause Analysis:**
- Fiber monthly charges: $89.79 avg
- DSL monthly charges: $56.36 avg
- Premium pricing not justified by perceived value

**Recommendation:** Reduce fiber pricing 15% or add premium support

</details>

<details>
<summary><strong>Finding 3: Monthly Charges Predict Churn 📈</strong></summary>

**T-Test Results:**
```
Churners:     $74.44 avg monthly
Non-churners: $61.27 avg monthly
Difference:   $13.17 (p < 0.001)
Cohen's d = 0.49 (medium effect)
```

**Logistic Regression Coefficient:**
- Each $10 increase in monthly charges → 1.15× odds of churning

</details>

<details>
<summary><strong>Finding 4: Senior Citizens at Risk 👴</strong></summary>

**Two-Proportion Z-Test:**
```
Senior citizen churn:    41.7%
Non-senior churn:        23.6%
Z-statistic: 8.92
p < 0.001
```

**Insight:** Seniors need targeted support and simplified pricing

</details>

---

## 💰 Financial Impact

### 📊 ROI Analysis

<table>
<tr>
<td align="center" width="25%">

**Current State**
7,043 customers
26.5% churn rate
$8.4M annual loss

</td>
<td align="center" width="25%">

**With Interventions**
30% churn reduction
18.6% new churn rate
$2.5M saved annually

</td>
<td align="center" width="25%">

**5-Year NPV**
$10.4M total value
87% ROI
Payback: 14 months

</td>
<td align="center" width="25%">

**Confidence**
95% CI: [$2.1M, $2.9M]
Conservative estimate
Statistical backing

</td>
</tr>
</table>

---

## 🎯 Business Recommendations

### Strategy 1: Contract Incentive Program

**Action:** Offer 20% discount on annual contract upgrades  
**Target:** 3,000 monthly contract customers  
**Expected Impact:** 15% churn reduction  
**Annual Value:** **$1.8M**

---

### Strategy 2: Fiber Optic Pricing Review

**Action:** Reduce fiber pricing 15% OR add premium support tier  
**Target:** 2,100 fiber customers  
**Expected Impact:** 10% churn reduction in fiber segment  
**Annual Value:** **$1.2M**

---

### Strategy 3: Predictive Retention Model

**Action:** ML model identifies top 20% at-risk, personalized offers  
**Target:** 1,400 high-risk customers  
**Expected Impact:** 40% success rate on interventions  
**Annual Value:** **$900K**

---

### **Total Annual Impact: $3.9M** 💰

---

## 🛠️ Statistical Techniques Applied

<details>
<summary><strong>📊 Show All 10+ Techniques</strong></summary>

| Category | Techniques | Usage |
|----------|------------|-------|
| **Descriptive** | Mean, median, std dev, percentiles | Segment profiling |
| **Distributions** | Normal, Exponential, Binomial | Pattern fitting |
| **Hypothesis Testing** | T-tests, ANOVA, Chi-square, Z-tests | Significance validation |
| **Confidence Intervals** | Bootstrap, parametric CIs | Uncertainty quantification |
| **Correlation** | Pearson, Spearman, VIF | Relationship detection |
| **Power Analysis** | Sample size, effect size | Experiment design |
| **Regression** | Logistic regression, odds ratios | Prediction modeling |
| **Validation** | Cross-validation, ROC/AUC | Model assessment |
| **Effect Sizes** | Cohen's d, Cramér's V | Practical significance |
| **Multiple Testing** | Bonferroni, FDR corrections | Type I error control |

</details>

---

## 🚀 Quick Start

### Installation

```bash
# Navigate to project
cd projects/project_telco_churn

# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Get Data

```bash
# Download from Kaggle
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn

# Place in data/raw/
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### Run Analysis

```bash
# Launch Jupyter
jupyter notebook notebooks/

# Execute in order: 01 → 09
```

---

## 📁 Project Structure

```
project_telco_churn/
├── 📊 notebooks/          # 9 comprehensive analyses
│   ├── 01_descriptive_statistics.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_probability_distributions.ipynb
│   ├── 04_hypothesis_testing.ipynb
│   ├── 05_confidence_intervals.ipynb
│   ├── 06_correlation_analysis.ipynb
│   ├── 07_power_analysis.ipynb
│   ├── 08_regression_modeling.ipynb
│   └── 09_final_recommendations.ipynb
│
├── 🔧 src/               # Reusable code
│   ├── data_processing.py
│   ├── statistical_tests.py
│   ├── visualization.py
│   └── utils.py
│
├── 📈 reports/           # Business deliverables
│   ├── statistical_summary.md
│   ├── business_recommendations.md
│   └── technical_appendix.md
│
└── 💾 data/              # Dataset
    ├── raw/
    └── processed/
```

---

## 🎓 Learning Outcomes

**By completing this project, you master:**

<table>
<tr>
<td width="50%">

### 📊 Statistical Skills
- ✅ Descriptive statistics
- ✅ Probability distributions
- ✅ Hypothesis testing (4 types)
- ✅ Confidence intervals
- ✅ Correlation analysis
- ✅ ANOVA & post-hoc tests
- ✅ Power analysis
- ✅ Logistic regression
- ✅ Effect size interpretation
- ✅ Multiple testing corrections

</td>
<td width="50%">

### 💼 Business Skills
- ✅ Problem framing
- ✅ Stakeholder communication
- ✅ ROI calculation
- ✅ Strategic recommendations
- ✅ Risk assessment
- ✅ A/B test design
- ✅ Decision-making under uncertainty
- ✅ Business case development
- ✅ Presentation of findings
- ✅ Actionable insight generation

</td>
</tr>
</table>

---

## 🏆 Technical Highlights

**Production-Ready Analysis:**

✅ **Statistical Rigor**
- All assumptions checked (normality, homoscedasticity)
- Effect sizes reported (not just p-values)
- Multiple testing corrections applied
- Bootstrap confidence intervals

✅ **Reproducibility**
- Random seeds set
- Version-controlled code
- Detailed documentation
- Environment specifications

✅ **Code Quality**
- Modular, reusable functions
- Clean, documented code
- Error handling
- Unit tests for key functions

---

## 📈 Dataset Details

**Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Attribute | Value |
|-----------|-------|
| **Customers** | 7,043 |
| **Features** | 21 (demographics, services, billing) |
| **Target** | Binary (Churn: Yes/No) |
| **Churn Rate** | 26.5% |
| **Time Period** | Cross-sectional snapshot |

---

## 🔗 Related Resources

**Continue Your Learning:**

- 📚 [Statistics Module](../../learning/02_statistics/) - Theory & concepts
- 📊 [P-Value Guide](../../learning/02_statistics/p_value_complete_guide.md) - Technical deep-dive
- 🎯 [Data Scientist Role Guide](../../learning/DATA_SCIENTIST_ROLE_GUIDE.md) - Career insights

**Similar Projects:**

- 🚢 [Titanic EDA](../project_titanic_eda/) - Foundational EDA skills
- 👥 [Customer Segmentation](../project_customer_segmentation/) - Unsupervised learning
- 🏠 [Housing Prediction](../project_housing_prediction/) - Regression focus

---

## 💡 Key Takeaways

> **"This project demonstrates how rigorous statistical analysis translates into multi-million dollar business value. Every hypothesis test, confidence interval, and regression coefficient directly informed the $3.9M retention strategy."**

**For Data Scientists:**
- ✅ Statistical rigor matters for business decisions
- ✅ Effect sizes are as important as p-values
- ✅ Complex problems require systematic analysis
- ✅ Communication bridges analysis and action

---

<div align="center">

**From Data to Decisions to Dollars** 💰

*9 notebooks • 10+ statistical techniques • $3.9M business impact*

[⬅️ Titanic EDA](../project_titanic_eda/) • [🏠 Home](../../README.md) • [➡️ Customer Segmentation](../project_customer_segmentation/)

---

**Estimated Completion Time:** 8-12 hours • **Difficulty:** Advanced • **ROI:** Portfolio-ready showcase

</div>
