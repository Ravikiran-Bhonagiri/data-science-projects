# Specialized EDA Techniques

This folder contains advanced, domain-specific EDA techniques. These are **optional** and should only be used when working in these specific domains.

---

## 📚 Contents

### 1. [Time Series EDA](./01_time_series_eda.md)
**When:** Stock prices, IoT sensors, sales forecasting

**Covers:**
- Stationarity testing (ADF, KPSS)
- ACF/PACF autocorrelation analysis
- Seasonal decomposition
- Rolling statistics
- Change point detection

---

### 2. [Geospatial EDA](./02_geospatial_eda.md)
**When:** Delivery optimization, real estate, location-based services

**Covers:**
- Mapping (Plotly, Folium)
- Geocoding (address ↔ coordinates)
- Distance calculations (Haversine)
- Choropleth maps
- Spatial clustering (DBSCAN)
- Heatmaps

---

### 3. [Text/NLP EDA](./03_text_nlp_eda.md)
**When:** Sentiment analysis, chatbots, document classification

**Covers:**
- Word frequency & stopword removal
- Word clouds
- N-gram analysis (bigrams/trigrams)
- Sentiment distribution
- Language detection
- Topic modeling (LDA)
- Named entity recognition

---

### 4. [Big Data EDA](./04_big_data_eda.md)
**When:** Datasets > 10GB that don't fit in RAM

**Covers:**
- Sampling strategies
- Dask (pandas-like for big data)
- PySpark (distributed computing)
- Vaex (memory-mapping)
- Incremental statistics
- Approximate algorithms (HyperLogLog)
- Parquet format optimization

---

### 5. [Interactive Dashboards](./05_interactive_dashboards.md)
**When:** Sharing analysis with non-technical stakeholders

**Covers:**
- Streamlit (fastest to build)
- Plotly Dash (production-grade)
- Panel/HoloViz (Jupyter-based)
- Deployment (Streamlit Cloud, Heroku, Docker)
- Best practices for dashboards

---

## 🚦 When to Use Each

| Domain | Priority | Complexity | Common Industries |
|--------|----------|------------|-------------------|
| Time Series | 🟢 Medium | Medium | Finance, IoT, E-commerce |
| Geospatial | 🟡 Low | High | Logistics, Real Estate, GIS |
| Text/NLP | 🟢 Medium | Low-Medium | Social Media, Customer Support |
| Big Data | 🔴 Low | Very High | Tech Giants, Data Warehouses |
| Dashboards | 🟡 Low | Medium | Any (for stakeholder communication) |

---

## 📝 Note
These guides assume you've mastered the core EDA concepts in the main module. They build upon that foundation with domain-specific tools and techniques.
