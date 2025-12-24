# Geospatial EDA

Geospatial data has **location** as a key dimension. Standard plots can't show "where" things happen. This guide covers map-based visualization and spatial analysis.

---

## When to Use This
- Delivery route optimization
- Real estate price analysis
- Disease outbreak tracking
- Customer location clustering
- Store placement planning

---

## 1. Understanding Coordinate Systems

**Lat/Long (Geographic):** Degrees on Earth's surface
- Latitude: -90° (South Pole) to +90° (North Pole)
- Longitude: -180° (West) to +180° (East)

```python
import pandas as pd
import geopandas as gpd

# Sample data
df = pd.DataFrame({
    'city': ['New York', 'Los Angeles', 'Chicago'],
    'latitude': [40.7128, 34.0522, 41.8781],
    'longitude': [-74.0060, -118.2437, -87.6298],
    'population': [8.3, 3.9, 2.7]
})

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(
    df, 
    geometry=gpd.points_from_xy(df.longitude, df.latitude),
    crs="EPSG:4326"  # WGS84 coordinate system
)
```

---

## 2. Projections & CRS (Coordinate Reference Systems)

**EPSG:4326 (WGS84):** Use for Latitude/Longitude (Degrees).
**EPSG:3857 (Web Mercator):** Used by Google Maps/OpenStreetMap (Meters).

```python
# Reprojecting for distance calculations (Meters is better than Degrees)
gdf = gdf.to_crs(epsg=3857)
print(gdf.geometry.iloc[0].xy) # Coordinates now in meters
```

---

## 3. Spatial Autocorrelation (Moran's I)

**Does location matter?** If high values are near other high values, we have **Spatial Autocorrelation**.

```python
import libpysal
from esda.moran import Moran

# Create weights matrix based on distance (or contiguity)
w = libpysal.weights.DistanceBand.from_dataframe(gdf, threshold=500000)
w.transform = 'R' # Row-standardize

# Calculate Moran's I
moran = Moran(gdf['population'], w)
print(f"Moran's I: {moran.I:.4f}")
print(f"p-value: {moran.p_sim:.4f}")

# I > 0: Positive Correlation (Clustered)
# I < 0: Negative Correlation (Dispersed)
# I ~ 0: Random
```

---

## 4. Working with GeoJSON & TopoJSON

```python
import geopandas as gpd

# Load administrative boundaries
# World: 'naturalearth_lowres' (built-in to geopandas)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Filter for a specific continent
europe = world[world.continent == 'Europe']
europe.plot()
```

---

## 5. Spatial Joins & Overlays

**Problem:** You have points (customers) and polygons (city zip codes). You want to know which zip code each customer is in.

```python
# Assume customers_gdf and zips_gdf are loaded
joined_df = gpd.sjoin(customers_gdf, zips_gdf, how="left", predicate="within")

# Spatial Aggregation: Count customers per zip code
count_per_zip = joined_df.groupby('zip_code').size().reset_index(name='count')
```

---

## 6. Advanced Mapping with Folium

### Choropleth with Legend
```python
import folium

m = folium.Map(location=[37, -102], zoom_start=4)

folium.Choropleth(
    geo_data='us-states.json',
    name='choropleth',
    data=state_data,
    columns=['state', 'sales'],
    key_on='feature.id',
    fill_color='YlGn',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Total Sales'
).add_to(m)

folium.LayerControl().add_to(m)
m.save('advanced_map.html')
```

---

## 5. Choropleth Maps (Region Coloring)

**Use case:** Color US states by sales, color countries by GDP.

```python
import plotly.express as px

# Sample state-level data
state_data = pd.DataFrame({
    'state': ['CA', 'TX', 'NY', 'FL'],
    'sales': [500000, 400000, 350000, 300000]
})

fig = px.choropleth(
    state_data,
    locations='state',
    locationmode="USA-states",
    color='sales',
    scope="usa",
    color_continuous_scale="Reds",
    labels={'sales': 'Total Sales ($)'}
)
fig.show()
```

---

## 6. Spatial Clustering (DBSCAN)

Find dense clusters of points in geographical space.

```python
from sklearn.cluster import DBSCAN
import numpy as np

# Sample: Customer locations
customers = pd.DataFrame({
    'lat': np.random.uniform(40.5, 41.0, 1000),
    'lon': np.random.uniform(-74.5, -73.5, 1000)
})

# Convert to radians for DBSCAN (uses haversine metric)
coords_rad = np.radians(customers[['lat', 'lon']].values)

# DBSCAN clustering
# eps = 0.5/6371 means ~0.5 km radius clusters
db = DBSCAN(eps=0.5/6371, min_samples=10, metric='haversine')
customers['cluster'] = db.fit_predict(coords_rad)

# Visualize
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
scatter = plt.scatter(customers['lon'], customers['lat'], 
                     c=customers['cluster'], cmap='viridis', s=10)
plt.colorbar(scatter, label='Cluster')
plt.title('Customer Location Clusters')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
```

---

## 7. Heatmaps (Density Visualization)

```python
import folium
from folium.plugins import HeatMap

# Create map
m = folium.Map(location=[40.7128, -74.0060], zoom_start=11)

# Add heatmap layer
heat_data = [[row['lat'], row['lon']] for idx, row in customers.iterrows()]
HeatMap(heat_data, radius=15).add_to(m)

m.save('heatmap.html')
```

---

## 8. Spatial Joins

**Use case:** Find which neighborhood each customer belongs to.

```python
import geopandas as gpd

# Load neighborhood boundaries (shapefile)
neighborhoods = gpd.read_file('neighborhoods.shp')

# Convert customer points to GeoDataFrame
customer_gdf = gpd.GeoDataFrame(
    customers,
    geometry=gpd.points_from_xy(customers.lon, customers.lat),
    crs="EPSG:4326"
)

# Spatial join: Find which neighborhood each customer is in
result = gpd.sjoin(customer_gdf, neighborhoods, how='left', predicate='within')
print(result[['lat', 'lon', 'neighborhood_name']].head())
```

---

## Checklist for Geospatial EDA

- [ ] Verify coordinate system (lat/long ranges make sense)
- [ ] Remove invalid coordinates (null, out of bounds)
- [ ] Plot basic map to see distribution
- [ ] Calculate distance matrix (if needed for optimization)
- [ ] Check for spatial clusters (DBSCAN)
- [ ] Create heatmap to see density patterns
- [ ] Geocode addresses if needed
- [ ] Perform spatial joins if using regions/boundaries
