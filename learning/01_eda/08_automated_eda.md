# 8. Automated EDA Tools

While manual EDA is crucial for deep understanding, **Automated EDA** tools are better for speed and breadth. They generate comprehensive reports in a single line of code, catching things you might miss (like "Column X has 98% zeros").

---

## 1. ydata-profiling (formerly pandas-profiling)
The gold standard for automated reports.

**What it gives you:**
*   **Overview:** duplicates, missing values, memory usage.
*   **Variables:** Detailed stats (mean, min, max, unique) + histograms for EVERY column.
*   **Interactions:** Correlation heatmaps (Pearson, Spearman).
*   **Alerts:** "Column X is highly correlated with Y (Rejected)".

```python
# pip install ydata-profiling

import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv('data.csv')

# Generate Report
profile = ProfileReport(df, title="Pandas Profiling Report")

# View in Notebook
profile.to_widgets()

# Save entire analysis as an interactive HTML file
profile.to_file("analysis_report.html")
```

---

## 2. Sweetviz
Focuses on **Target Analysis** and **Comparison** (e.g., Train vs Test).

**Why use it?**
It shows high-density plots that compare two datasets side-by-side. Perfect for detecting **Training-Serving Skew** or how features differ between "Churned" and "Retained" users.

```python
# pip install sweetviz

import sweetviz as sv

# Compare Train vs Test
my_report = sv.compare([train_df, "Train"], [test_df, "Test"], target_feat="Churn")

my_report.show_html("comparison_report.html")
```

---

## 3. Klib
A lightweight library for cleaning and visualizing. Less "report" generation, more "utility".

**Features:**
*   `klib.missingval_plot(df)`: Beautiful missing data map.
*   `klib.corr_plot(df)`: Clean correlation matrix (removes upper triangle automatically).
*   `klib.data_cleaning(df)`: Automatically drops empty cols, fixes types, optimizes memory.

```python
# pip install klib
import klib

# Visualize missing data
klib.missingval_plot(df)

# Clean dataset (Drop duplicates & empty cols) in one lin!
df_clean = klib.data_cleaning(df)
```

---

## ⚔️ Manual vs Automated

| Feature | Manual EDA (`seaborn`/`pandas`) | Automated (`profiling`/`sweetviz`) |
| :--- | :--- | :--- |
| **Speed** | Slow (write code for every plot) | Instant (1 line, 1 minute) |
| **Depth** | **High** (Custom questions) | **Low** (Generic stats) |
| **Control** | Full control (colors, bins) | No control (fixed template) |
| **Best For** | Investigating specific insights | **Initial Dataset Audit** |

**Recommendation:** Always run **Automated EDA first** to get the "Bird's Eye View", then use **Manual EDA** to dive deep into the specific weirdness you found.
