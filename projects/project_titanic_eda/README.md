<div align="center">

# 🚢 Titanic Survival Forensics

### *Investigating the Tragedy Through Data*

![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![Type](https://img.shields.io/badge/Type-EDA-blue?style=flat-square)
![Level](https://img.shields.io/badge/Level-Beginner-green?style=flat-square)

**Comprehensive exploratory analysis uncovering survival patterns and biases**

[🎯 Problem](#-problem-statement) • [📊 Dataset](#-dataset) • [💡 Findings](#-key-findings) • [🚀 Run It](#-how-to-run)

</div>

---

## 🎯 Problem Statement

> **"Why did some survive while others perished?"**

Following the sinking of the RMS Titanic on April 15, 1912, investigators need a **forensic data analysis** to understand systematic patterns in who survived and who didn't.

<table>
<tr>
<td width="50%">

### 🔍 The Investigation

**Not just prediction—UNDERSTANDING**

- ❓ Did "Women and Children First" hold true?
- ❓ Was survival equal across all classes?
- ❓ What factors most influenced survival?
- ❓ Were there hidden biases in the data?

</td>
<td width="50%">

### 🎯 The Goal

**Rigorous, production-grade EDA to:**

- ✅ Audit the passenger manifest
- ✅ Identify data quality issues
- ✅ Quantify survival factors
- ✅ Uncover class-based patterns

</td>
</tr>
</table>

---

## 💾 Dataset

### 📊 Titanic Passenger Manifest

**Source:** Seaborn/Kaggle Classic Dataset

<table>
<tr>
<td align="center" width="25%">

**👥 Passengers**
891 total

</td>
<td align="center" width="25%">

**📋 Features**
11 variables

</td>
<td align="center" width="25%">

**⚠️ Missing Data**
Age (20%), Deck (77%)

</td>
<td align="center" width="25%">

**✅ Survived**
342 (38%)

</td>
</tr>
</table>

### 🔢 Features Explained

| Feature | Type | Description | Example |
|---------|------|-------------|---------|
| **Pclass** | Categorical | Ticket class (1st, 2nd, 3rd) | Social status indicator |
| **Sex** | Categorical | Male/Female | Protocol compliance test |
| **Age** | Numerical | Age in years | 20% missing - critical! |
| **SibSp** | Numerical | Siblings/Spouses aboard | Family size |
| **Parch** | Numerical | Parents/Children aboard | Family dependency |
| **Fare** | Numerical | Ticket price | Economic indicator |
| **Embarked** | Categorical | Port (C, Q, S) | Boarding location |
| **Deck** | Categorical | Deck level | 77% missing (MNAR) |

---

## 🚨 Data Quality Challenges

<table>
<tr>
<td width="50%">

### ⚠️ Missing Age (20%)

**Problem:** 177 passengers missing age data  
**Type:** MAR (Missing At Random)  
**Impact:** Critical for "Children First" analysis

**Solution Applied:**
```
✅ Advanced imputation
✅ Median by Class + Gender
✅ Preserves signal
✅ Avoids row deletion bias
```

</td>
<td width="50%">

### ⚠️ Missing Deck (77%)

**Problem:** 687 passengers missing cabin/deck  
**Type:** MNAR (Missing Not At Random)  
**Impact:** Lower class passengers less likely to have cabins

**Solution Applied:**
```
✅ Acknowledge bias
✅ Create "Unknown" category
✅ Feature still informative
✅ Reflects reality
```

</td>
</tr>
</table>

---

## 💡 Key Findings

### 🏆 Major Discoveries

<details>
<summary><strong>1️⃣ "Women and Children First" Protocol - VERIFIED ✅</strong></summary>

**Finding:** Protocol was strictly followed

**Evidence:**
- 👩 Female survival: **74%**
- 👨 Male survival: **19%**
- 👶 Children (<16): **54%** survival

**Statistical Significance:** p < 0.001 (Chi-square test)

**Conclusion:** The protocol was honored, with women 3.9× more likely to survive than men.

</details>

<details>
<summary><strong>2️⃣ Socio-Economic Bias - THE CLASS DIVIDE 🚨</strong></summary>

**Finding:** Survival was NOT equal across classes

**Evidence:**
- 🥇 **1st Class:** 63% survival rate
- 🥈 **2nd Class:** 47% survival rate  
- 🥉 **3rd Class:** 24% survival rate

**Impact:** 1st class passengers were **2.6× more likely** to survive than 3rd class

**Root Cause Analysis:**
- Emergency egress design favored upper decks
- 3rd class passengers further from lifeboats
- Language barriers (many immigrants)
- Crew priorities

</details>

<details>
<summary><strong>3️⃣ Forensic Imputation - PRESERVING SIGNAL 📊</strong></summary>

**Challenge:** Simply dropping 20% missing Age data would introduce bias

**Our Approach:**
```
Group-based imputation:
├─ 1st Class Males: Median age 40
├─ 1st Class Females: Median age 36
├─ 2nd Class Males: Median age 30
├─ 2nd Class Females: Median age 28
└─ 3rd Class: Median age 24
```

**Validation:**
- Distribution shape preserved
- Class/gender patterns maintained
- No artificial peaks introduced

**Result:** More accurate analysis without losing 177 passengers

</details>

<details>
<summary><strong>4️⃣ Automated Data Auditing - PRODUCTION READY 🛠️</strong></summary>

**Innovation:** Built reusable `generate_data_quality_report()` function

**Capabilities:**
- ✅ Instantly flag missing values
- ✅ Identify high cardinality issues
- ✅ Detect data type mismatches
- ✅ Report correlation issues

**Business Value:** Standard first-step for all future EDA projects

</details>

---

## 📊 Visual Insights

### Survival by Class and Gender

```
1st Class Women:  ████████████████████████████████ 96% survived
1st Class Men:    ████████████ 37% survived
                  
2nd Class Women:  ████████████████████████ 92% survived  
2nd Class Men:    ███████ 16% survived
                  
3rd Class Women:  ████████████ 50% survived
3rd Class Men:    ███ 14% survived
```

**Key Insight:** Class mattered MORE for men than women. Female survival remained high across all classes, but male survival plummeted in lower classes.

---

## 🔬 EDA Techniques Demonstrated

<table>
<tr>
<td width="50%">

### 📊 Analysis Methods

- ✅ Missing data pattern detection
- ✅ Advanced imputation (group medians)
- ✅ Outlier detection & handling
- ✅ Correlation analysis
- ✅ Class imbalance assessment
- ✅ Hypothesis testing (Chi-square)

</td>
<td width="50%">

### 🎨 Visualizations

- ✅ Survival rate heatmaps
- ✅ Distribution plots (Age, Fare)
- ✅ Count plots (Class, Gender)
- ✅ Correlation matrices
- ✅ Missing data visualizations
- ✅ Box plots (Fare by Class)

</td>
</tr>
</table>

---

## 🚀 How to Run

### Prerequisites

```bash
Python 3.8+
Jupyter Notebook
```

### Installation

```bash
# Navigate to project
cd projects/project_titanic_eda

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### Run Analysis

**Option 1: Jupyter Notebook** (Recommended)
```bash
# Open and run
notebooks/01_comprehensive_eda.ipynb
```

**Option 2: Python Script**
```bash
# Run forensic analysis
python notebooks/01_comprehensive_eda.py
```

---

## 📚 What You'll Learn

<table>
<tr>
<td align="center" width="33%">

### 🔍 EDA Fundamentals

Data auditing
Missing data handling
Outlier detection
Distribution analysis

</td>
<td align="center" width="33%">

### 📊 Statistical Analysis

Hypothesis testing
Chi-square tests
Correlation analysis
Group comparisons

</td>
<td align="center" width="33%">

### 💼 Production Skills

Reusable functions
Data quality reports
Bias detection
Clean documentation

</td>
</tr>
</table>

---

## 🎯 Project Outcomes

**✅ Completed Deliverables:**
1. Comprehensive EDA notebook with 15+ visualizations
2. Reusable data quality audit function
3. Statistical validation of survival factors
4. Production-ready code with documentation
5. Bias detection and quantification

**📈 Skills Demonstrated:**
- Advanced missing data imputation
- Class imbalance analysis
- Hypothesis testing
- Data visualization mastery
- Production-quality code

---

## 💡 Key Takeaways

> **"Not all passengers were equal in the eyes of the Titanic disaster. While the 'Women and Children First' protocol was followed, socio-economic class created a 2.6× survival gap between 1st and 3rd class passengers."**

**For Data Scientists:**
- ✅ Always audit data quality first
- ✅ Missing data treatment can preserve or destroy signal
- ✅ Visualization reveals patterns statistics might miss
- ✅ Production code should be reusable

---

## 🔗 Related Projects

**Next Steps in Your Learning:**

- 📊 [Customer Segmentation](../project_customer_segmentation/) - Apply clustering
- 📞 [Telco Churn](../project_telco_churn/) - Statistical modeling ($3.9M impact)
- 🏠 [Housing Prediction](../project_housing_prediction/) - Regression techniques

---

<div align="center">

**Every Great Analysis Starts with Great EDA** 🎯

*This project demonstrates foundational EDA skills on a classic dataset*

[⬅️ Back to Projects](../) • [🏠 Home](../../README.md) • [➡️ Next: Housing Prediction](../project_housing_prediction/)

</div>
