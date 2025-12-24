# Module 6: Advanced Feature Engineering

**"Data is fuel, but Feature Engineering is the Refinery."**

You can dump crude oil (Raw Data) into a Ferrari (XGBoost), and it will destroy the engine.
You must refine it first.
Feature Engineering is the art of **creating new information** from existing data to make the model's job easier.

**Golden Rule:** A simple model with amazing features > A complex model with garbage features.

---

## 📚 The Refinery

### Phase 1: The Foundation (Preprocessing)
*Moving digits around.*
1.  **[Encoding Strategies](./01_encoding_strategies.md):** The "Translator". Beyond One-Hot: Target Encoding, CatBoost, WoE.
2.  **[Scaling & Normalization](./02_scaling_normalization.md):** The "Equalizer". RobustScaler vs PowerTransformer.

### Phase 2: The Selection (Dieting)
*Removing the noise.*
3.  **[Feature Selection Methods](./03_feature_selection_methods.md):** The "Diet Plan". RFE, Boruta, and SHAP-based selection.

### Phase 3: The Creation (Synthesis)
*Creating gold from lead.*
4.  **[Interaction & Polynomials](./04_interaction_polynomial_features.md):** The "Alchemist". Capturing non-linear logic ($A \times B$).
5.  **[Handling High Cardinality](./05_handling_high_cardinality.md):** "Crowd Control". Compressing 1 million cities into vectors (Hashing/Embeddings).

### Phase 4: Automation (The Future)
6.  **[Automated Feature Engineering](./06_automated_feature_engineering.md):** "The Robot". Deep Feature Synthesis with Featuretools.
7.  **[Time-Based Features](./07_time_based_features.md):** Cyclical encoding, lags, rolling windows. The most valuable features in production.

---

## 🧠 Quick Reference: The Engineering Checklist

| Problem | Technique | Why? |
|---|---|---|
| **Categories have Order (S/M/L)** | Label Encoding | Preserves rank. |
| **Categories have Risk info (City)** | Target Encoding | Captures the signal directly. |
| **Outliers in Data** | RobustScaler | Median-based, ignores billonaires. |
| **Curved Data (Parabola)** | Polynomial Features | Allows linear models to fit curves. |
| **Too many columns** | Recursive Elimination (RFE) | Finds the strongest subset. |
| **Tables (SQL-like)** | Featuretools (DFS) | Crawls relationships automatically. |
