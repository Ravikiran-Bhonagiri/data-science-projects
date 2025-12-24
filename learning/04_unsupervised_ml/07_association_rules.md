# Association Rules: The "Mind Reader"

Have you ever walked into a grocery store to buy **milk**, and somehow left with **Oreos**?
That wasn't an accident. That was an algorithm.

**Association Rule Mining** doesn't care about "Why" you do things.
It just knows that **People who do X, often do Y.**
It is the engine behind "Customers who bought this also bought..."

### 🧠 When should you actually use this?

**1. The Walmart "Beer & Diapers" Legend:**
A famous data mining story. Walmart analyzed millions of baskets on Friday nights.
*   **The Discovery:** Young men bought **Diapers** (for the baby) and **Beer** (for themselves, since they couldn't go to the bar).
*   **The Action:** Walmart put the beer next to the diapers.
*   **The Result:** Sales exploded. The algorithm found a connection no human would ever guess.

**2. The Netflix Binge-Watcher:**
You just finished "Stranger Things."
*   **The Rule:** $\{Stranger Things, Dark\} \rightarrow Black Mirror$.
*   **The Recommendation:** "98% Match for you!"
*   **Why Rules?** It keeps you subscribed. It predicts your addiction before you even know it.

---

## 1. The Core Metrics: Support, Confidence, Lift

**1. Support (Popularity):**
How often does this itemset appear?
$$Support(A) = \frac{\text{Transactions with A}}{\text{Total Transactions}}$$

**2. Confidence (Trustworthiness):**
If I have A, how likely is B?
$$Confidence(A \rightarrow B) = \frac{Support(A \cap B)}{Support(A)}$$
*   "70% of people who bought bread also bought butter."

**3. Lift (The "Real" Signal):**
Is this relationship stronger than random chance?
$$Lift(A \rightarrow B) = \frac{Confidence(A \rightarrow B)}{Support(B)}$$
*   **Lift = 1:** Independent. (Buying A implies nothing about B).
*   **Lift > 1:** Positive correlation. (A boosts B). **This is what we want.**
*   **Lift < 1:** Negative correlation. (Buying A means they likely won't buy B).

---

## 2. The Algorithms

### A. Apriori
The classic. uses a "bottom-up" approach.
1.  Find all frequent single items.
2.  Combine them into pairs. Check frequency.
3.  Combine frequent pairs into triplets.
4.  **Pruning:** If $\{Beer\}$ is rare, then $\{Beer, Diapers\}$ must also be rare, so don't even check it.
*   **Con:** Slow on massive datasets.

### B. FP-Growth (Frequent Pattern)
Faster. Builds a tree structure (FP-Tree) in memory.
*   **Pro:** Scans database only twice. Much faster for Big Data.

---

## 3. Implementation (Mlxtend)

Sklearn does NOT support Association Rules natively. We use the standard `mlxtend` library.

```python
# pip install mlxtend
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# 1. The Transaction Data (One-Hot Encoded)
dataset = [
    ['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
    ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
    ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
    ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
    ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']
]

from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# 2. Run Apriori (Find Frequent Itemsets)
# min_support=0.6 means "Item must appear in 60% of transactions"
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)

# 3. Generate Rules
# metric="lift", min_threshold=1.0 (Only show positive correlations)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# 4. View Top Rules
result = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
print(result.sort_values('lift', ascending=False))

# Example Output Interpretation:
# Antecedents: (Onion) -> Consequents: (Eggs)
# Lift: 1.2 (They are 20% more likely to be bought together than by chance)
```

---

## 4. Visualizing Rules (Network Graph)

Association rules are best viewed as a graph where Node A connects to Node B with edge thickness = Lift.

```python
import networkx as nx
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
G = nx.DiGraph()

for i, row in rules.iterrows():
    # Only plot strong rules
    if row['lift'] > 1.1:
        # Frozenset to string
        ant = list(row['antecedents'])[0]
        con = list(row['consequents'])[0]
        G.add_edge(ant, con, weight=row['lift'])

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='skyblue', 
        node_size=2000, edge_color='gray', arrowsize=20)
plt.title("Product Association Network")
plt.show()
```
