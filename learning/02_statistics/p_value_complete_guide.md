# P-Values and Statistical Significance: A Complete Technical Guide

**Understanding the Mathematics, Interpretation, and Application of P-Values in Statistical Testing**

---

## Table of Contents

1. [Mathematical Foundation of P-Values](#foundation)
2. [Significance Thresholds (α Levels)](#thresholds)
3. [Type I and Type II Errors](#errors)
4. [P-Values Across Different Tests](#different-tests)
5. [Effect Size vs Statistical Significance](#effect-size)
6. [Multiple Testing Problem](#multiple-testing)
7. [Common Misconceptions](#misconceptions)
8. [Practical Decision Framework](#framework)

---

## 1. Mathematical Foundation of P-Values {#foundation}

### Formal Definition

**P-Value (p):** The probability of obtaining test results at least as extreme as the observed results, assuming the null hypothesis (H₀) is true.

**Mathematical Expression:**
```
p = P(|T| ≥ |t_observed| | H₀ is true)

Where:
- T = test statistic (random variable)
- t_observed = observed test statistic from your sample
- H₀ = null hypothesis
```

### What P-Value Actually Measures

**P-value answers this question:**
> "If there truly was NO effect/difference (H₀ true), what's the probability I would see data at least this extreme by random chance alone?"

**P-value DOES NOT answer:**
- ❌ Probability that H₀ is true
- ❌ Probability that results are due to chance
- ❌ Importance or magnitude of effect
- ❌ Probability of replication

---

### Technical Example: One-Sample T-Test

**Scenario:** Testing if average customer satisfaction score differs from 7.0

**Data:**
- Sample: n = 50 customers
- Sample mean (x̄) = 7.8
- Sample std dev (s) = 1.2
- Population mean under H₀ (μ₀) = 7.0

**Hypotheses:**
```
H₀: μ = 7.0  (population mean equals 7.0)
H₁: μ ≠ 7.0  (population mean differs from 7.0)
```

**Test Statistic Calculation:**
```
t = (x̄ - μ₀) / (s/√n)
t = (7.8 - 7.0) / (1.2/√50)
t = 0.8 / 0.1697
t = 4.714
```

**Degrees of Freedom:**
```
df = n - 1 = 49
```

**P-Value Calculation:**
```
p = 2 × P(T₄₉ > 4.714)
p = 2 × 0.000015
p ≈ 0.00003

Where T₄₉ follows t-distribution with 49 degrees of freedom
```

**Interpretation:**
If the true population mean were 7.0, the probability of observing a sample mean as extreme as 7.8 (or more) purely by chance is 0.003% (3 in 100,000).

---

## 2. Significance Thresholds (α Levels) {#thresholds}

### Standard Alpha Levels

**α (alpha):** Pre-determined threshold for rejecting H₀

| α Level | Interpretation | Common Use Cases |
|---------|---------------|------------------|
| **0.10** (10%) | Liberal | Exploratory research, screening studies |
| **0.05** (5%) | Standard | Most scientific research, A/B testing |
| **0.01** (1%) | Conservative | High-stakes decisions, medical trials |
| **0.001** (0.1%) | Very Conservative | Particle physics, genomics (after corrections) |

### Decision Rule

**Reject H₀ if:** p < α  
**Fail to reject H₀ if:** p ≥ α

**Important:** "Fail to reject" ≠ "Accept H₀"

---

### Why Different Thresholds?

#### Example 1: Medical Drug Approval (α = 0.01)

**Context:** Testing if new drug reduces blood pressure

**Why α = 0.01:**
- High cost of Type I error (approving ineffective/harmful drug)
- Patient safety paramount
- Need stronger evidence

**Interpretation of p = 0.02:**
```
p = 0.02 > α = 0.01
Decision: Fail to reject H₀
Conclusion: Insufficient evidence drug works
Action: Do not approve drug (yet)
```

#### Example 2: Website A/B Test (α = 0.05)

**Context:** Testing if new button color increases clicks

**Why α = 0.05:**
- Lower stakes (reversible decision)
- Need balance between Type I and Type II errors
- Industry standard

**Interpretation of p = 0.02:**
```
p = 0.02 < α = 0.05
Decision: Reject H₀
Conclusion: Significant difference detected
Action: Implement new button color
```

#### Example 3: Exploratory Data Analysis (α = 0.10)

**Context:** Screening 100 variables to find potential predictors

**Why α = 0.10:**
- Initial exploration phase
- Want to cast wider net
- Will validate findings later with stricter threshold

**Interpretation of p = 0.08:**
```
p = 0.08 < α = 0.10
Decision: Reject H₀
Conclusion: Variable shows promise
Action: Include in next phase of analysis
```

---

### Confidence Levels and Alpha

**Relationship:**
```
Confidence Level = 1 - α

α = 0.05 → 95% confidence level
α = 0.01 → 99% confidence level
α = 0.10 → 90% confidence level
```

**Example:**
If p = 0.03 and α = 0.05:
- We reject H₀ at 95% confidence level
- We are 95% confident the effect is real
- We accept 5% risk of Type I error

---

## 3. Type I and Type II Errors {#errors}

### Error Types Matrix

|  | **H₀ is True** | **H₀ is False** |
|---|---|---|
| **Reject H₀** | **Type I Error** (α) | ✅ Correct Decision (Power) |
| **Fail to Reject H₀** | ✅ Correct Decision | **Type II Error** (β) |

### Type I Error (False Positive)

**Definition:** Rejecting H₀ when it's actually true  
**Probability:** α (significance level)  
**Also called:** False alarm, False discovery

**Example: Medical Screening**
```
H₀: Patient does not have disease
Type I Error: Test says "positive" but patient is healthy

Consequences:
- Unnecessary treatment
- Patient anxiety
- Healthcare costs
- Further unnecessary testing
```

**Numerical Example:**
```
Scenario: Testing 1000 healthy people with α = 0.05

Expected Type I Errors:
1000 × 0.05 = 50 false positives

Interpretation: Even though everyone is healthy, 
we expect 50 people to test "positive" by chance alone.
```

### Type II Error (False Negative)

**Definition:** Failing to reject H₀ when it's actually false  
**Probability:** β (beta)  
**Power:** 1 - β (probability of correctly rejecting false H₀)

**Example: Medical Screening**
```
H₀: Patient does not have disease
Type II Error: Test says "negative" but patient is sick

Consequences:
- Missed diagnosis
- Delayed treatment
- Disease progression
- Patient harm
```

**Numerical Example:**
```
Scenario: Testing 100 sick patients with β = 0.20

Expected Type II Errors:
100 × 0.20 = 20 false negatives

Interpretation: We miss 20 out of 100 actual cases.
Power = 1 - 0.20 = 0.80 (we catch 80% of true cases)
```

---

### The Trade-off

**Decreasing α (stricter threshold):**
- ✅ Reduces Type I errors
- ❌ Increases Type II errors
- ❌ Requires larger sample size for same power

**Increasing α (looser threshold):**
- ❌ Increases Type I errors
- ✅ Reduces Type II errors
- ✅ Requires smaller sample size

**Example: Drug Testing Trade-off**

| α Level | Type I Error Risk | Type II Error Risk | Practical Impact |
|---------|------------------|-------------------|------------------|
| 0.10 | High (10%) | Lower | Might approve ineffective drugs |
| 0.05 | Medium (5%) | Medium | Balanced approach |
| 0.01 | Low (1%) | Higher | Might reject effective drugs |

---

### Calculating Required Sample Size

**For desired power (1 - β) = 0.80, α = 0.05:**

**Formula (two-sample t-test):**
```
n = 2 × (z_α/2 + z_β)² × (σ²/δ²)

Where:
- z_α/2 = 1.96 (for α = 0.05, two-tailed)
- z_β = 0.84 (for power = 0.80)
- σ = population standard deviation
- δ = minimum detectable effect size
```

**Example:**
```
Detect difference in means of 0.5 units
σ = 1.0
α = 0.05
Power = 0.80

n = 2 × (1.96 + 0.84)² × (1.0²/0.5²)
n = 2 × 7.84 × 4
n ≈ 63 per group
```

---

## 4. P-Values Across Different Tests {#different-tests}

### 4.1 One-Sample T-Test

**Use Case:** Compare sample mean to known value

**Example: Quality Control**
```
Manufacturing spec: Bolts should be 10.0 mm diameter
Sample: n = 30 bolts
Sample mean: x̄ = 10.15 mm
Sample SD: s = 0.25 mm

H₀: μ = 10.0
H₁: μ ≠ 10.0

Test Statistic:
t = (10.15 - 10.0) / (0.25/√30)
t = 0.15 / 0.0456
t = 3.29

df = 29
p-value = 0.0027

Decision at α = 0.05:
p < α → Reject H₀

Conclusion: Bolts are significantly different from spec
(too large on average)

Practical Significance:
Deviation of 0.15 mm may or may not matter depending
on tolerance requirements!
```

---

### 4.2 Two-Sample T-Test (Independent)

**Use Case:** Compare means of two independent groups

**Example: A/B Test**
```
Control Group (A): n₁ = 500, x̄₁ = 3.2% conversion, s₁ = 1.1%
Treatment Group (B): n₂ = 500, x̄₂ = 3.8% conversion, s₂ = 1.2%

H₀: μ₁ = μ₂ (no difference in conversion rates)
H₁: μ₁ ≠ μ₂ (conversion rates differ)

Pooled Standard Deviation:
s_p = √[((n₁-1)s₁² + (n₂-1)s₂²)/(n₁+n₂-2)]
s_p = √[(499×1.1² + 499×1.2²)/998]
s_p = 1.15%

Standard Error:
SE = s_p × √(1/n₁ + 1/n₂)
SE = 1.15 × √(1/500 + 1/500)
SE = 1.15 × 0.0632
SE = 0.0727%

Test Statistic:
t = (x̄₂ - x̄₁) / SE
t = (3.8 - 3.2) / 0.0727
t = 8.25

df = 998
p-value < 0.0001

Decision: Reject H₀
Conclusion: Treatment B has significantly higher conversion

Effect Size (Cohen's d):
d = (3.8 - 3.2) / 1.15 = 0.52 (medium effect)
```

---

### 4.3 Paired T-Test

**Use Case:** Compare two measurements on same subjects

**Example: Before-After Study**
```
Weight loss program: 25 participants

Before (kg): μ₁ = 85.2, s₁ = 12.3
After (kg):  μ₂ = 82.1, s₂ = 11.8
Mean difference (d̄): -3.1 kg
SD of differences (s_d): 2.5 kg

H₀: μ_d = 0 (no weight change)
H₁: μ_d ≠ 0 (weight changed)

Test Statistic:
t = d̄ / (s_d/√n)
t = -3.1 / (2.5/√25)
t = -3.1 / 0.5
t = -6.2

df = 24
p-value < 0.0001

Decision: Reject H₀
Conclusion: Significant weight loss occurred

Confidence Interval (95%):
CI = d̄ ± t₀.₀₂₅,₂₄ × (s_d/√n)
CI = -3.1 ± 2.064 × 0.5
CI = -3.1 ± 1.032
CI = [-4.13, -2.07] kg

Interpretation: We're 95% confident true weight loss
is between 2.07 and 4.13 kg.
```

---

### 4.4 ANOVA (Analysis of Variance)

**Use Case:** Compare means of 3+ groups

**Example: Marketing Campaign Performance**
```
Three email designs tested:
- Design A: n₁ = 100, x̄₁ = 5.2% CTR
- Design B: n₂ = 100, x̄₂ = 6.1% CTR  
- Design C: n₃ = 100, x̄₃ = 7.3% CTR

H₀: μ₁ = μ₂ = μ₃ (all designs equal)
H₁: At least one mean differs

ANOVA Table:
Source      | SS    | df  | MS   | F     | p-value
------------|-------|-----|------|-------|--------
Between     | 215.4 | 2   | 107.7| 12.31 | 0.0001
Within      | 2596.8| 297 | 8.74 |       |
Total       | 2812.2| 299 |      |       |

F-statistic = MS_between / MS_within = 107.7 / 8.74 = 12.31
p-value = 0.0001

Decision: Reject H₀
Conclusion: At least one design differs significantly

Post-hoc Tests (Tukey HSD):
A vs B: p = 0.089 (not significant at α=0.05)
A vs C: p = 0.0001 (significant)
B vs C: p = 0.012 (significant)

Final Interpretation:
Design C is significantly better than A and B.
Designs A and B are not significantly different.
```

---

### 4.5 Chi-Square Test of Independence

**Use Case:** Test association between categorical variables

**Example: Customer Segment and Product Preference**
```
Observed Frequencies:
              | Product X | Product Y | Product Z | Total
--------------|-----------|-----------|-----------|------
Segment A     |    45     |    30     |    25     | 100
Segment B     |    20     |    50     |    30     | 100
Segment C     |    15     |    20     |    65     | 100
Total         |    80     |   100     |   120     | 300

H₀: Segment and product preference are independent
H₁: Segment and product preference are associated

Expected Frequencies (under H₀):
E_ij = (Row Total × Column Total) / Grand Total

For Segment A, Product X:
E₁₁ = (100 × 80) / 300 = 26.67

Chi-Square Statistic:
χ² = Σ [(O_ij - E_ij)² / E_ij]

Calculations:
χ² = (45-26.67)²/26.67 + (30-33.33)²/33.33 + ... + (65-40)²/40
χ² = 12.59 + 0.33 + ... + 15.63
χ² = 58.92

df = (rows - 1) × (columns - 1) = 2 × 2 = 4
p-value < 0.0001

Decision: Reject H₀
Conclusion: Strong association between segment and preference

Effect Size (Cramér's V):
V = √[χ²/(n × min(rows-1, cols-1))]
V = √[58.92/(300 × 2)]
V = 0.31 (medium effect)

Interpretation:
- Segment A prefers Product X (45 vs 26.67 expected)
- Segment B prefers Product Y (50 vs 33.33 expected)
- Segment C prefers Product Z (65 vs 40 expected)
```

---

### 4.6 Correlation Test

**Use Case:** Test if correlation coefficient is significantly different from zero

**Example: Advertising Spend vs Sales**
```
n = 50 observations
Pearson correlation (r) = 0.42

H₀: ρ = 0 (no linear relationship)
H₁: ρ ≠ 0 (linear relationship exists)

Test Statistic:
t = r × √[(n-2)/(1-r²)]
t = 0.42 × √[(48)/(1-0.1764)]
t = 0.42 × √(48/0.8236)
t = 0.42 × 7.63
t = 3.20

df = n - 2 = 48
p-value = 0.0024

Decision: Reject H₀
Conclusion: Significant positive correlation

Confidence Interval for ρ (Fisher's Z-transform):
z = 0.5 × ln[(1+r)/(1-r)] = 0.448
SE_z = 1/√(n-3) = 0.146

95% CI for z: [0.162, 0.734]
Convert back to r: [0.16, 0.62]

Interpretation:
- Moderate positive correlation (r = 0.42)
- As advertising spend increases, sales tend to increase
- R² = 0.1764: Ad spend explains 17.64% of sales variance
- Other factors explain remaining 82.36%
```

---

## 5. Effect Size vs Statistical Significance {#effect-size}

### The Critical Distinction

**Statistical Significance (p-value):**
- Tells you IF an effect exists
- Affected by sample size
- Binary decision (sig/not sig)

**Effect Size:**
- Tells you HOW LARGE the effect is
- Independent of sample size
- Continuous measure

### Example: Large Sample, Small Effect

```
A/B Test: 1,000,000 users

Control: 10.50% conversion (n = 500,000)
Treatment: 10.55% conversion (n = 500,000)

Difference: 0.05 percentage points

Statistical Test:
z = (0.1055 - 0.1050) / SE
SE = √[p(1-p)(1/n₁ + 1/n₂)] = 0.00043
z = 0.0005 / 0.00043 = 1.16
p = 0.246 (not significant at α = 0.05)

Wait, increase sample to 10,000,000:
SE = 0.000136
z = 0.0005 / 0.000136 = 3.68
p = 0.0002 (highly significant!)

But effect size (absolute difference):
Still only 0.05 percentage points

Practical Significance:
Revenue increase: 10,000,000 × 0.0005 × $50 = $250,000/year
Despite small effect size, large practical value!
```

---

### Common Effect Size Measures

#### Cohen's d (for t-tests)

**Formula:**
```
d = (Mean₁ - Mean₂) / Pooled SD
```

**Interpretation:**
- d = 0.2: Small effect
- d = 0.5: Medium effect
- d = 0.8: Large effect

**Example:**
```
Training Program Impact:
Control: μ₁ = 72, s₁ = 10
Training: μ₂ = 78, s₂ = 12
s_pooled = 11

d = (78 - 72) / 11 = 0.55 (medium effect)

Even if p = 0.25 (not significant), the effect size suggests
meaningful difference that might be significant with larger sample.
```

#### Cramér's V (for Chi-Square)

**Formula:**
```
V = √[χ²/(n × min(rows-1, cols-1))]
```

**Interpretation:**
- V = 0.1: Small association
- V = 0.3: Medium association
- V = 0.5: Large association

#### R² (for Regression/ANOVA)

**Interpretation:**
- R² = 0.01: 1% of variance explained (small)
- R² = 0.09: 9% of variance explained (medium)
- R² = 0.25: 25% of variance explained (large)

---

## 6. Multiple Testing Problem {#multiple-testing}

### The Problem

**When conducting multiple tests, Type I error rate inflates:**

```
Single test at α = 0.05:
P(at least one Type I error) = 0.05

10 tests at α = 0.05:
P(at least one Type I error) = 1 - (1-0.05)¹⁰ = 0.40 (40%!)

20 tests at α = 0.05:
P(at least one Type I error) = 1 - (1-0.05)²⁰ = 0.64 (64%!)
```

### Example: Gene Expression Study

```
Testing 10,000 genes for differential expression
α = 0.05 for each test

Expected false positives:
10,000 × 0.05 = 500 genes

Even if NO genes are truly different, we expect to find
500 "significant" results by chance alone!
```

---

### Correction Methods

#### 1. Bonferroni Correction

**Most Conservative Approach**

**Formula:**
```
α_corrected = α / m

Where m = number of tests
```

**Example:**
```
10 tests, α = 0.05
α_corrected = 0.05 / 10 = 0.005

Only reject H₀ if p < 0.005
```

**Results:**
```
Test  | Original p | Reject at α=0.05? | Reject at α_Bonf=0.005?
------|------------|-------------------|------------------------
1     | 0.001      | Yes               | Yes
2     | 0.003      | Yes               | Yes
3     | 0.012      | Yes               | No
4     | 0.045      | Yes               | No
5     | 0.120      | No                | No

Original: 4/5 significant
Bonferroni: 2/5 significant
```

**Pros:** Simple, controls family-wise error rate (FWER)  
**Cons:** Very conservative, high Type II error when m is large

---

#### 2. Benjamini-Hochberg (FDR Control)

**Less Conservative, Controls False Discovery Rate**

**Procedure:**
1. Order p-values: p₁ ≤ p₂ ≤ ... ≤ p_m
2. Find largest k where: p_k ≤ (k/m) × α
3. Reject H₀ for tests 1 through k

**Example:**
```
m = 10 tests, α = 0.05

Test | p-value | Rank | (k/m)×α | Decision
-----|---------|------|---------|----------
A    | 0.001   | 1    | 0.005   | Reject (0.001 < 0.005)
B    | 0.003   | 2    | 0.010   | Reject (0.003 < 0.010)
C    | 0.008   | 3    | 0.015   | Reject (0.008 < 0.015)
D    | 0.012   | 4    | 0.020   | Reject (0.012 < 0.020)
E    | 0.019   | 5    | 0.025   | Reject (0.019 < 0.025)
F    | 0.027   | 6    | 0.030   | Reject (0.027 < 0.030)
G    | 0.035   | 7    | 0.035   | Reject (0.035 = 0.035)
H    | 0.042   | 8    | 0.040   | Don't Reject (0.042 > 0.040)
I    | 0.089   | 9    | 0.045   | Don't Reject
J    | 0.145   | 10   | 0.050   | Don't Reject

Reject H₀ for tests A through G (7 out of 10)
```

**Comparison:**
```
Method              | Significant Tests | False Discovery Rate
--------------------|-------------------|---------------------
None (α = 0.05)     | 8/10              | Unknown (high)
Bonferroni          | 2/10              | <5% (very low)
Benjamini-Hochberg  | 7/10              | <5% (controlled)
```

---

## 7. Common Misconceptions {#misconceptions}

### Misconception 1: "p = 0.05 means 95% chance result is real"

**WRONG!**

**What p = 0.05 actually means:**
"If H₀ were true, there's a 5% chance of observing data this extreme."

**It does NOT mean:**
- 95% probability the hypothesis is true
- 5% probability the result is due to chance

**Example:**
```
Drug trial: p = 0.04

CORRECT: If the drug had no effect, there's a 4% probability
of seeing results this good or better by chance.

INCORRECT: There's a 96% chance the drug works.
```

**Why the confusion?**
P(data | H₀) ≠ P(H₀ | data)

To get P(H₀ | data), you need Bayesian analysis with prior probabilities.

---

### Misconception 2: "Non-significant means no effect"

**WRONG!**

**p > α means:**
- Insufficient evidence to reject H₀
- NOT proof that H₀ is true

**Example:**
```
Weight loss study:
Mean difference: 2.5 kg
p = 0.08 (not significant at α = 0.05)
95% CI: [-0.3 kg, 5.3 kg]

WRONG Interpretation:
"The drug doesn't cause weight loss."

CORRECT Interpretation:
"We cannot conclude the drug causes weight loss with 95% confidence.
The effect could range from slight weight gain (-0.3 kg) to 
substantial weight loss (5.3 kg). Need larger study to know."
```

---

### Misconception 3: "p = 0.001 means larger effect than p = 0.04"

**WRONG!**

P-value indicates strength of EVIDENCE, not magnitude of EFFECT.

**Example:**
```
Study A: n = 1000, effect = 0.2 units, p = 0.001
Study B: n = 50, effect = 1.5 units, p = 0.04

Study B has:
- Larger effect (1.5 vs 0.2)
- Less statistical evidence (p = 0.04 vs 0.001)

Both are significant at α = 0.05
Study B's effect is 7.5× larger!
But Study A has stronger statistical evidence due to larger n.
```

---

### Misconception 4: "Significant results are always important"

**WRONG!**

**Example: Blood Pressure Study**
```
n = 10,000
Drug reduces BP by 0.5 mmHg
p < 0.0001 (highly significant)

Statistical Significance: Yes
Clinical Significance: No (0.5 mmHg is trivial)

Cardiologist's view:
"A 10 mmHg reduction matters clinically.
0.5 mmHg is meaningless despite being 'significant'."
```

---

### Misconception 5: "p-hacking isn't a problem if I find p < 0.05"

**WRONG!**

**P-Hacking Examples:**
1. Testing multiple outcomes, reporting only significant ones
2. Adding participants until p < 0.05
3. Trying different analyses until one is significant
4. Removing "outliers" to achieve significance

**Example:**
```
Researcher tests 20 different outcomes
1 shows p = 0.03
Reports only that one as "significant finding"

Problem: Expected false positives = 20 × 0.05 = 1
This "significant" result is likely a false positive!

Solution: Pre-register analysis plan
```

---

## 8. Practical Decision Framework {#framework}

### Step-by-Step Guide

#### Step 1: Set Hypotheses and α BEFORE analyzing data

```
✅ Good Practice:
1. State hypotheses
2. Choose α (usually 0.05)
3. Determine required sample size
4. Collect data
5. Analyze
6. Report all results

❌ Bad Practice:
1. Collect data
2. Try different tests
3. Report what's significant
```

#### Step 2: Check Assumptions

**For t-tests:**
- ✓ Normality (or large n via CLT)
- ✓ Independence
- ✓ Equal variances (for pooled t-test)

**For Chi-square:**
- ✓ Expected frequencies ≥ 5
- ✓ Independence

**For ANOVA:**
- ✓ Normality
- ✓ Independence
- ✓ Homogeneity of variance

#### Step 3: Calculate Test Statistic and P-Value

Use appropriate test based on:
- Data type (continuous, categorical)
- Number of groups
- Paired vs independent
- Sample size

#### Step 4: Report Results Completely

**Minimum reporting:**
```
For t-test:
"Mean difference = 2.5 kg (95% CI: 1.2 to 3.8 kg),
t(48) = 3.76, p = 0.0004, d = 0.54"

For Chi-square:
"χ²(4) = 58.92, p < 0.0001, Cramér's V = 0.31"

For ANOVA:
"F(2, 297) = 12.31, p = 0.0001, η² = 0.08"
```

#### Step 5: Interpret in Context

**Three Questions:**
1. Is it statistically significant? (p < α)
2. Is the effect size meaningful?
3. What's the practical significance?

**Example Decision Matrix:**

| p-value | Effect Size | Decision |
|---------|-------------|----------|
| <0.05 | Large | ✅ Act on finding |
| <0.05 | Small | ⚠️ Consider context |
| >0.05 | Large | ⚠️ Need more data? |
| >0.05 | Small | ❌ Likely not important |

---

### Real-World Example: Complete Analysis

**Scenario:** E-commerce company comparing two checkout flows

**Given:**
```
Checkout Flow A (current): n = 5,000, conversions = 350 (7.0%)
Checkout Flow B (new): n = 5,000, conversions = 400 (8.0%)
```

**Step 1: Hypotheses**
```
H₀: p_A = p_B (conversion rates are equal)
H₁: p_A ≠ p_B (conversion rates differ)
α = 0.05
```

**Step 2: Test Choice**
Two-proportion z-test (large samples, binary outcome)

**Step 3: Calculate**
```
p̂_A = 350/5000 = 0.070
p̂_B = 400/5000 = 0.080
p̂_pooled = 750/10000 = 0.075

SE = √[p̂_pooled(1 - p̂_pooled)(1/n_A + 1/n_B)]
SE = √[0.075 × 0.925 × 0.0004]
SE = 0.00527

z = (p̂_B - p̂_A) / SE
z = (0.080 - 0.070) / 0.00527
z = 1.90

p-value = 0.057 (two-tailed)
```

**Step 4: Effect Size**
```
Absolute difference: 1.0 percentage point
Relative lift: 14.3% increase
Odds ratio: 1.16
```

**Step 5: Confidence Interval**
```
95% CI for difference:
0.010 ± 1.96 × 0.00527
0.010 ± 0.0103
[-0.0003, 0.0203]
or [-0.03%, 2.03%]
```

**Step 6: Interpretation**

**Statistical:**
- p = 0.057 > 0.05: Not statistically significant at α = 0.05
- However, p = 0.057 is close to significance threshold
- 95% CI includes zero (barely: -0.03%)

**Practical:**
```
Expected annual impact if B is truly better:
Annual visitors: 2,000,000
Incremental conversions: 2,000,000 × 0.01 = 20,000
Value per conversion: $50
Annual value: 20,000 × $50 = $1,000,000
```

**Decision Framework:**

```
Option 1: Reject due to p > 0.05
Reasoning: Statistical evidence insufficient
Risk: Might miss $1M opportunity

Option 2: Continue testing
Reasoning: Close to significance, large potential value
Action: Run test for 2 more weeks (double sample size)
Risk: Small additional cost for more certainty

Option 3: Implement B anyway
Reasoning: Low implementation cost, high potential upside
Action: Monitor closely, A/B test other markets
Risk: Might not realize expected gains

Recommended: Option 2 (continue testing)
Rationale: Potential $1M value justifies gathering more evidence
```

---

## Summary

### Key Takeaways

1. **P-value measures evidence strength**, not effect size or importance

2. **Significance threshold (α) is a decision point**, not a truth detector

3. **Type I and Type II errors are inevitable trade-offs**
   - Lower α → fewer false positives, more false negatives
   - Higher α → more false positives, fewer false negatives

4. **Multiple testing inflates Type I error** - use corrections

5. **Always report**:
   - Exact p-value
   - Effect size
   - Confidence intervals
   - Context

6. **Statistical significance ≠ Practical significance**
   - Large samples can make tiny effects "significant"
   - Small samples can make large effects "non-significant"

7. **Context matters most**:
   - Cost of errors (Type I vs Type II)
   - Practical implications
   - Prior knowledge
   - Replication possibility

---

### Final Formula Reference

**Common Tests:**

```
One-sample t-test:
t = (x̄ - μ₀) / (s/√n)

Two-sample t-test:
t = (x̄₁ - x̄₂) / √(s₁²/n₁ + s₂²/n₂)

Paired t-test:
t = d̄ / (s_d/√n)

ANOVA F-test:
F = MS_between / MS_within

Chi-square:
χ² = Σ[(O - E)² / E]

Correlation:
t = r√[(n-2)/(1-r²)]

Z-test for proportions:
z = (p̂₁ - p̂₂) / SE_pooled
```

---

*Use this guide as a reference when conducting statistical tests. Remember: statistical methods are tools for decision-making under uncertainty, not absolute truth detectors.*
