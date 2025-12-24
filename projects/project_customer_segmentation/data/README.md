# Online Retail Dataset

## Source
UCI Machine Learning Repository

## Download Instructions

### Option 1: Direct Download
```bash
# Download the dataset
curl -o data/raw/Online_Retail.xlsx "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
```

### Option 2: Manual Download
1. Visit: https://archive.ics.uci.edu/ml/datasets/Online+Retail
2. Download `Online Retail.xlsx`
3. Place in `data/raw/` folder

## Dataset Description
- **Transactions:** 541,909 (Dec 2010 - Dec 2011)
- **Customers:** ~4,300 unique
- **Countries:** 38
- **Features:**
  - InvoiceNo: Invoice number
  - StockCode: Product code
  - Description: Product name
  - Quantity: Quantity per transaction
  - InvoiceDate: Invoice date/time
  - UnitPrice: Product price per unit
  - CustomerID: Unique customer identifier
  - Country: Customer country

## Citation
```
Daqing Chen, Sai Liang Sain, and Kun Guo. Data mining for the online retail industry: 
A case study of RFM model-based customer segmentation using data mining. 
Journal of Database Marketing and Customer Strategy Management, 19(3):197-208, 2012.
```

## License
This dataset is publicly available for research purposes.
