# EDA Module - Future Enhancements

This module is **complete** for general-purpose EDA. The following topics are **optional** and should only be added if you work in these specialized domains.

---

## 📋 Potential Additions (Domain-Specific)

### 🕒 Time Series EDA
**When to add:** If doing forecasting, stock analysis, or IoT sensor data.

**Topics to cover:**
- [ ] Stationarity testing (ADF, KPSS tests)
- [ ] Autocorrelation plots (ACF, PACF)
- [ ] Seasonal decomposition (trend, seasonality, residuals)
- [ ] Rolling statistics (moving averages, std dev)
- [ ] Detecting change points
- [ ] Handling irregular time intervals

**Libraries:** `statsmodels`, `prophet`, `pmdarima`

---

### 🗺️ Geospatial EDA
**When to add:** If doing location-based analysis, GIS, or delivery optimization.

**Topics to cover:**
- [ ] Coordinate systems and projections
- [ ] Geocoding (address → lat/long)
- [ ] Distance calculations (Haversine)
- [ ] Choropleth maps (color-coded regions)
- [ ] Spatial clustering (DBSCAN with distance metrics)
- [ ] Heatmaps (density visualization)

**Libraries:** `geopandas`, `folium`, `plotly`, `contextily`

---

### 📝 Text/NLP EDA
**When to add:** If doing sentiment analysis, chatbots, or document classification.

**Topics to cover:**
- [ ] Word frequency distributions
- [ ] Word clouds
- [ ] N-gram analysis (bigrams, trigrams)
- [ ] Vocabulary size and diversity
- [ ] Sentence/document length distributions
- [ ] Language detection
- [ ] Topic modeling previews

**Libraries:** `nltk`, `spacy`, `wordcloud`, `textblob`

---

### 📊 Big Data EDA
**When to add:** If working with datasets > 10GB that don't fit in memory.

**Topics to cover:**
- [ ] Sampling strategies (random, stratified)
- [ ] Distributed computing (Spark, Dask)
- [ ] Incremental statistics
- [ ] Approximate algorithms (HyperLogLog for cardinality)
- [ ] Memory-efficient visualization

**Libraries:** `dask`, `pyspark`, `vaex`

---

### 🎛️ Interactive EDA Dashboards
**When to add:** If building internal tools or stakeholder demos.

**Topics to cover:**
- [ ] Streamlit apps
- [ ] Plotly Dash
- [ ] Panel/HoloViz
- [ ] Embedding plots in web apps

**Libraries:** `streamlit`, `dash`, `panel`

---

## 🚦 Priority Guide

| Addition | Priority | Effort | Use Cases |
|----------|----------|--------|-----------|
| Time Series EDA | 🟢 Medium | Medium | Finance, IoT, Forecasting |
| Geospatial EDA | 🟡 Low | High | Logistics, Real Estate |
| Text/NLP EDA | 🟢 Medium | Low | Chatbots, Sentiment Analysis |
| Big Data EDA | 🔴 Low | Very High | Only if dataset > 10GB |
| Dashboards | 🟡 Low | Medium | Stakeholder presentations |

---

## 📝 Notes
- **Don't add these preemptively.** Only add when you have a real project in that domain.
- Each would be a new file: `12_time_series_eda.md`, `13_geospatial_eda.md`, etc.
- Keep the module focused on general EDA. Specialized topics can bloat it.
