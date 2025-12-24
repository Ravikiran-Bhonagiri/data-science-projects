# Big Data EDA (Spark & Dask)

When datasets exceed RAM (>10GB), standard pandas operations fail with `MemoryError`. This guide covers techniques for analyzing data that doesn't fit in memory.

---

## When to Use This
- Datasets > 10GB
- Logs from production servers (TBs)
- Historical transaction data (billions of rows)
- Click-stream analytics
- Satellite/sensor data

---

## 1. Sampling Strategies (First Defense)

**Before going distributed:** Can you sample?

### Random Sampling
```python
import pandas as pd

# Read only 10% of rows
df_sample = pd.read_csv('huge_file.csv', skiprows=lambda i: i>0 and np.random.rand() > 0.1)

# Or use chunksize
chunks = []
for chunk in pd.read_csv('huge_file.csv', chunksize=100000):
    sample = chunk.sample(frac=0.1)
    chunks.append(sample)
df_sample = pd.concat(chunks, ignore_index=True)
```

### Stratified Sampling
```python
# Ensure all categories represented
df_sample = df.groupby('category', group_keys=False).apply(lambda x: x.sample(frac=0.1))
```

---

## 2. Dask (Pandas-like API for Big Data)

**When to use:** 10GB - 1TB datasets, single machine with limited RAM

```python
import dask.dataframe as dd

# Read large CSV (lazy - doesn't load yet!)
ddf = dd.read_csv('huge_file.csv')

# Operations are lazy until compute()
print(ddf.head())  # Only loads first rows

# Compute statistics (triggers execution)
result = ddf.groupby('category')['value'].mean().compute()
print(result)
```

### Key Dask Patterns

**1. Filtering**
```python
# Filter first, then compute
ddf_filtered = ddf[ddf['age'] > 30]
result = ddf_filtered.compute()  # Only loads filtered rows
```

**2. Aggregations**
```python
# Group-by operations
result = ddf.groupby('state').agg({
    'sales': 'sum',
    'customers': 'count'
}).compute()
```

**3. Persist to avoid recomputation**
```python
# Expensive operation - persist result in memory
ddf_clean = ddf.dropna().persist()

# Now multiple operations on ddf_clean are fast
print(ddf_clean.mean().compute())
print(ddf_clean.std().compute())
```

---

## 3. PySpark (Cluster Computing)

**When to use:** > 1TB datasets, distributed cluster (multiple machines)

```python
from pyspark.sql import SparkSession

# Initialize Spark
spark = SparkSession.builder \
    .appName("EDA") \
    .getOrCreate()

# Read data
df = spark.read.csv('huge_file.csv', header=True, inferSchema=True)

# Show schema
df.printSchema()

# Basic stats
df.describe().show()

# Group-by
df.groupBy('category').count().show()

# SQL queries
df.createOrReplaceTempView("data")
result = spark.sql("SELECT category, AVG(value) FROM data GROUP BY category")
result.show()
```

### Spark EDA Operations

**1. Missing Data**
```python
from pyspark.sql.functions import col, count, when

# Count nulls per column
df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()
```

**2. Histograms**
```python
# Approximate histogram (exact is too slow)
df.select('age').rdd.flatMap(lambda x: x).histogram(10)
```

**3. Sampling**
```python
# Random sample
df_sample = df.sample(fraction=0.01, seed=42)
df_sample.toPandas()  # Convert to pandas for visualization
```

---

## 4. Vaex (Lazy Evaluation, Memory-Mapping)

**When to use:** Billion-row datasets on a single machine

```python
import vaex

# Memory-map file (doesn't load into RAM!)
df = vaex.open('huge_file.hdf5')  # Or .csv, .parquet

# Nearly instant operations
print(df.mean('value'))
print(df.correlation('age', 'income'))

# Filter (lazy)
df_filtered = df[df.age > 30]

# Plot (uses sampling internally for speed)
df.plot('age', 'income', shape=128)
```

---

## 5. Incremental Statistics

Calculate mean/std without loading all data at once.

```python
import pandas as pd

# Welford's online algorithm for mean & variance
def incremental_stats(filename, chunksize=100000):
    n = 0
    mean = 0
    M2 = 0
    
    for chunk in pd.read_csv(filename, chunksize=chunksize):
        for value in chunk['value']:
            n += 1
            delta = value - mean
            mean += delta / n
            delta2 = value - mean
            M2 += delta * delta2
    
    variance = M2 / (n - 1) if n > 1 else 0
    return mean, variance**0.5

mean, std = incremental_stats('huge_file.csv')
print(f"Mean: {mean:.2f}, Std: {std:.2f}")
```

---

## 6. Approximate Algorithms

For massive data, exactness isn't always necessary.

### HyperLogLog (Cardinality Estimation)
```python
from hyperloglog import HyperLogLog

hll = HyperLogLog(0.01)  # 1% error rate

# Add millions of values
for chunk in pd.read_csv('huge_file.csv', chunksize=100000):
    for value in chunk['user_id']:
        hll.add(str(value))

print(f"Approximate unique users: {len(hll)}")
```

### Count-Min Sketch (Frequency Estimation)
```python
from countminsketch import CountMinSketch

cms = CountMinSketch(width=1000, depth=7)

# Count word frequencies
for chunk in pd.read_csv('reviews.csv', chunksize=100000):
    for text in chunk['review']:
        for word in text.split():
            cms.add(word)

print(f"Approximate count of 'good': {cms.query('good')}")
```

---

## 5. Storage Formats: Parquet & Partitioning

**CSV is a row-format.** Reading 1 column requires reading the entire file.
**Parquet is a columnar-format.** Reading 1 column is near-instant.

### Partitioning Strategy
Partition by high-level categories (e.g., Year, Month, State) to skip data during reads.

```python
# Partition data on disk
df.to_parquet('data/', partition_cols=['year', 'month'])

# Structured disk:
# /data/year=2023/month=01/part-001.parquet
# /data/year=2023/month=02/part-001.parquet

# Fast read: (only reads month 01)
ddf = dd.read_parquet('data/year=2023/month=01/')
```

---

## 6. Spark Optimization: Shuffling & Partitioning

**Shuffling:** Moving data between machines. It is the slowest part of Big Data.

```python
# Check partitions
print(df.rdd.getNumPartitions())

# Repartition if data is too skewed (imbalanced across machines)
df_repartitioned = df.repartition(100, "user_id")

# Optimization: Broadcast Join (Send small DF to all machines instead of shuffling)
from pyspark.sql.functions import broadcast
large_df.join(broadcast(small_df), "key").show()
```

---

## 7. Cloud Integration (S3 / GCS)

```python
# AWS S3 with Boto3 and Dask
import dask.dataframe as dd

# Requires s3fs installed
ddf = dd.read_csv('s3://my-bucket/data/*.csv', 
                  storage_options={'key': '...', 'secret': '...'})
```

---

## 8. Data Lakes vs Data Warehouses in EDA

- **Data Lake (S3, HDFS):** Raw data (JSON, CSV, Parquet). Great for exploratory discovery.
- **Data Warehouse (Snowflake, BigQuery):** Structured data. Perfect for final dashboards and reporting.

**EDA Flow:**
1. Discover patterns in **Data Lake** using Spark/Dask.
2. Store refined, aggregate tables in **Data Warehouse**.
3. Point **Dashboards** at the Warehouse.

---

## Checklist for Big Data EDA

- [ ] Don't load entire dataset into pandas (will crash)
- [ ] Try sampling first (10% random sample)
- [ ] Convert CSV to Parquet for faster reads
- [ ] Use Dask for pandas-like API
- [ ] Use Spark for cluster computing
- [ ] Use incremental algorithms (online mean/variance)
- [ ] Use approximate algorithms (HyperLogLog for cardinality)
- [ ] Profile memory usage (monitor with `memory_profiler`)
