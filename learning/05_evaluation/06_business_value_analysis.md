# Business Value Analysis: The ROI Calculation

**The Board Meeting:**

**You:** "Our churn prediction model has 0.84 AUC and 0.76 F1-score!"

**CFO:** "What does that mean in dollars?"

**You:** "Well... um... it's statistically significant..."

**CFO:** "You're wasting our time. Next agenda item."

---

## Real Case: Telecom Churn Prevention

**The Setup:**
- Customer base: 2 million subscribers
- Annual churn rate: 25% (500,000 leave per year)
- Average customer lifetime value: $2,400
- Retention offer cost: $150/customer

**Model Performance:**
- Predicts churn with 80% recall, 40% precision
- We action the top 100,000 highest-risk customers

**The Financial Matrix:**

| Outcome | Count | What Happens | Cost/Revenue |
|---------|-------|--------------|--------------|
| **True Positive** | 32,000 | We save a churner with offer | Revenue: +$2,400<br>Cost: -$150<br>**Net: +$2,250** |
| **False Positive** | 68,000 | We give offer to loyal customer | Cost: -$150<br>**Net: -$150** |
| **True Negative** | 1,400,000 | Loyal customer, no action | **Net: $0** |
| **False Negative** | 500,000 | They churn, we did nothing | Loss: -$2,400<br>**Net: -$2,400** |

**Total Business Impact:**
```
Revenue = (32,000 × $2,250) - (68,000 × $150) - (500,000 × $2,400)
        = $72M - $10.2M - $1.2B
        = -$1.138 Billion

# Wait, that's terrible!
```

**What went wrong?** We only reached 100,000 customers. We missed 468,000 churners.

---

## Optimizing for Profit, Not F1-Score

**Better strategy:** Reach more customers (action top 300,000)

**New numbers:**
- True Positives: 180,000 (saved churners)
- False Positives: 120,000 (wasted offers)

```
Revenue = (180,000 × $2,250) - (120,000 × $150)
        = $405M - $18M  
        = +$387M annual value
```

**Key insight:** The optimal threshold for F1-score (0.5) is NOT the optimal threshold for profit (0.23).

---

## The Profit Curve vs PR Curve

---

## 2. The Cumulative Gains Curve (Lift Chart)

Imagine you have 100,000 customers. You can only call 10,000 of them.
Who do you call?

*   **Random:** You call 10% of customers, you get 10% of the total churners. (Diagonal Line).
*   **Model:** You sort customers by "Probability of Churn". You call the top 10%.
*   **Result:** You might catch **40% of the total churners** in that top 10%.

**That "40% vs 10%" is the LIFT.**
(Lift = 4.0).
This proves to the CEO: "My model creates 4x more efficiency than random guessing."

---

## 3. The Profit Curve (The "Killer" Graph)

The most robust way to pick a threshold is not F1-score. It is **Max Profit**.

1.  Sweep thresholds from 0.0 to 1.0.
2.  At each threshold, calculate the Confusion Matrix (TP, FP, etc.).
3.  Multiply by the Dollar Values defined in Section 1.
4.  Plot **Total Profit** vs **Threshold**.

*   It will look like a parabola.
*   **Peak Profit** is your operating point.
*   *Note: This might be totally different from the Peak F1-score!*

```python
# Calculating Profit Curve
costs = {'TP': 450, 'FP': -50, 'FN': -500, 'TN': 0}
thresholds = np.linspace(0, 1, 100)
profits = []

for t in thresholds:
    y_pred = (y_proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    profit = (tp * costs['TP']) + (fp * costs['FP']) + (fn * costs['FN']) + (tn * costs['TN'])
    profits.append(profit)

plt.plot(thresholds, profits)
plt.xlabel("Threshold")
plt.ylabel("Total Profit ($)")
plt.title("Profit Curve: Optimizing for Money")
plt.show()
```

---

## 4. A/B Testing (The Final Proof)

Metrics are simulations. **A/B Tests are reality.**
Before full rollout:
1.  **Control Group (50%):** Business as usual (Random calls or Old Model).
2.  **Treatment Group (50%):** Your New Model actions.

Run for 1 month. Compare **Revenue Per User** (RPU).
If `Treatment_RPU > Control_RPU` (statistically significant t-test), you win.
