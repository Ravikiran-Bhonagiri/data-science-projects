# Module 2: Statistical Foundations for Data Science

## Overview

Statistics is the mathematical foundation of Data Science. This module covers essential statistical concepts organized into focused topics.

---

## Contents

### 1. [Probability Distributions](./probability_distributions.md)
Learn about the key probability distributions used in data science:
- Normal (Gaussian) Distribution
- Binomial Distribution
- Poisson Distribution
- Exponential Distribution
- Central Limit Theorem

### 8. [End-to-End Project: Telco Customer Churn](./project_telco_churn/README.md)
A comprehensive practical application of all concepts:
- **Scenario:** Reducing customer churn by 30%
- **Analysis:** 9 Jupyter notebooks covering descriptive stats to ROI analysis
- **Outcome:** Strategic retention plan worth $3.9M/year
- **Techniques:** Hypothesis testing, power analysis, interaction effects, logistic regression


### 2. [Hypothesis Testing](./hypothesis_testing.md)
Framework for making data-driven decisions:
- Null vs Alternative Hypothesis
- p-values and Significance Levels
- T-Tests (One-Sample, Two-Sample, Paired)
- Type I and Type II Errors
- Statistical Power

### 3. [Confidence Intervals](./confidence_intervals.md)
Quantifying uncertainty in estimates:
- Constructing Confidence Intervals
- Interpretation and Common Mistakes
- Bootstrap Methods
- Margin of Error

### 4. [ANOVA (Analysis of Variance)](./anova.md)
Comparing means across multiple groups:
- One-Way ANOVA
- Post-hoc Tests (Tukey HSD)
- Assumptions and Diagnostics
- When to Use vs T-Tests

### 5. [Chi-Square Tests](./chi_square.md)
Testing relationships between categorical variables:
- Chi-Square Test of Independence
- Goodness of Fit Test
- Expected vs Observed Frequencies
- Cramér's V (Effect Size)

### 6. [Correlation Analysis](./correlation_analysis.md)
Understanding relationships between variables:
- Pearson Correlation
- Spearman's Rank Correlation
- Correlation vs Causation
- Partial Correlation
- Common Pitfalls

### 7. [Power Analysis & Sample Size](./power_analysis.md)
Designing experiments properly:
- Statistical Power Calculation
- Sample Size Determination
- Multiple Testing Corrections (Bonferroni, FDR)
- Effect Size Interpretation

---

## Learning Path

**Beginner:**
1. Start with Probability Distributions
2. Move to Hypothesis Testing
3. Learn Confidence Intervals

**Intermediate:**
4. ANOVA for multi-group comparisons
5. Chi-Square for categorical data
6. Correlation Analysis

**Advanced:**
7. Power Analysis for experiment design

**Project Phase:**
8. Complete the [Telco Customer Churn Project](./project_telco_churn/README.md) to build your portfolio.

---

## Why This Module Matters

Understanding statistics allows you to:
- **Design experiments** properly (A/B tests, clinical trials)
- **Validate assumptions** before applying ML models
- **Quantify uncertainty** in predictions
- **Make data-driven decisions** with confidence
- **Communicate results** effectively to stakeholders

---

## Prerequisites

- Basic Python knowledge
- Familiarity with NumPy, Pandas, Matplotlib
- Understanding of mean, median, variance

---

## Quick Reference

| Test | Use Case | Data Type | Example |
|------|----------|-----------|---------|
| **T-Test** | Compare 2 means | Numerical | Control vs Treatment group |
| **ANOVA** | Compare 3+ means | Numerical | Compare sales across 5 regions |
| **Chi-Square** | Test independence | Categorical | Gender vs Product Preference |
| **Pearson Correlation** | Linear relationship | Numerical | Age vs Income |
| **Spearman Correlation** | Monotonic relationship | Ordinal/Numerical | Education Level vs Salary |

---

## Practical Applications

- **A/B Testing:** Hypothesis testing, confidence intervals
- **Segmentation:** ANOVA to compare segments
- **Feature Engineering:** Correlation analysis for feature selection
- **Experimental Design:** Power analysis for sample size
- **Market Research:** Chi-square for survey analysis

---

## Next Steps

Choose a topic from the contents above based on your needs, or work through them sequentially for comprehensive understanding.
