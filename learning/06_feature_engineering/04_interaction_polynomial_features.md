# Interaction Features: When Addition Fails

**Insurance Pricing Model - The Problem:**

**Linear model assumption:**
```
Risk = 0.3×Age + 0.5×Smoking_Status + 0.2×BMI
```

**Reality:**
- 25-year-old smoker: Low risk (young body handles it)
- 65-year-old smoker: **Catastrophic risk** (heart disease spike)

The linear model treats smoking the same at all ages. **It's systematically wrong.**

**After adding interaction:**
```python
df['Age_x_Smoking'] = df['Age'] * df['IsSmoker']
```

**Result:**
- For non-smokers: `Age_x_Smoking = 0` (no effect)
- For smokers: Feature captures the multiplicative risk increase with age
- **Business impact:** Pricing accuracy improved 23%, reduced underwriting losses by $4.2M/year

---

## Real Example: Digital Advertising

**Problem:** Predict ad click-through rate
- `Hour_of_Day` alone: weak signal
- `Device_Type` alone: weak signal

**Interaction insight:**
- Mobile users click more at **8AM** (commute) and **10PM** (bed)
- Desktop users click more at **2PM** (work break)

```python
df['Device_Hour'] = df['Device'] + '_' + df['Hour'].astype(str)
# Creates: 'Mobile_8AM', 'Desktop_2PM', etc.
```

**A/B test result:** +18% CTR improvement

---

## 1. Polynomial Features (Curvature)

If the data is curved (like a parabola), a straight line fails.
You add `x^2` (Squared) and `x^3` (Cubed) columns.
Now the linear model can fit a curve.

```python
from sklearn.preprocessing import PolynomialFeatures

# degree=2 creates: [a, b, a^2, ab, b^2]
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X)
```

**Warning:** This explodes feature count. (10 features $\rightarrow$ ~65 features).

---

## 2. Interaction Features (A $\times$ B)

The "AND" logic.
*   **Scenario:** Ads work on "Young People" AND "Gaming Websites".
*   Feature: `Age_x_Gaming = Age * Is_Gaming_Site`.
    *   If Old (60) * Is_Gaming (1) = 60.
    *   If Young (20) * Is_Gaming (1) = 20.
    *   If Young (20) * Not_Gaming (0) = 0.

This helps the model isolate specific sub-groups.

---

## 3. Domain-Specific Ratios (The Gold Mine)

Automatic features (Poly/Interaction) are okay.
**Human Intuition features are God Tier.**

**Examples:**
1.  **Credit Scoring:**
    *   Raw: `Debt`, `Income`.
    *   Engineered: `Debt_to_Income_Ratio = Debt / Income`. (This is how banks actually decide).

2.  **E-Commerce:**
    *   Raw: `Total_Items`, `Total_Spent`.
    *   Engineered: `Avg_Price_Per_Item = Spent / Items`. (Luxury shopper vs Bulk buyer).

3.  **Real Estate:**
    *   Raw: `SqFt`, `Price`.
    *   Engineered: `Price_per_SqFt`.

**Strategy:** Ask yourself "How would a human make this decision?" and create that number using Math (+, -, /, *).
