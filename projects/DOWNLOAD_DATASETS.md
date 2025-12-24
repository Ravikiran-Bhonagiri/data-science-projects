# Dataset Download Instructions

## Automated Downloads

### 1. Online Retail Dataset (Customer Segmentation)
```powershell
Invoke-WebRequest -Uri "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx" `
    -OutFile "data/raw/Online_Retail.xlsx"
```

### 2. Credit Card Default Dataset (Model Evaluation)
```powershell
Invoke-WebRequest -Uri "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls" `
    -OutFile "data/raw/credit_card_default.xls"
```

### 3. Adult Census Income Dataset (Feature Engineering)
```powershell
Invoke-WebRequest -Uri "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data" `
    -OutFile "data/raw/adult.data"
```
**Note:** Dataset has no headers. We explicitly define them in the notebook:
`['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']`

## Manual Download Links

1. **Online Retail:** https://archive.ics.uci.edu/ml/datasets/Online+Retail
2. **Credit Card Default:** https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
3. **Used Cars:** https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data

## Dataset Sizes
- Online Retail: ~20 MB
- Credit Card Default: ~5 MB
- Used Cars: ~1.4 GB (compressed)
