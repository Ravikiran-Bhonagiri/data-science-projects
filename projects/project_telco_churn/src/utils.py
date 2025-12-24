"""
Utility functions for the Telco Churn Analysis project
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath: str = 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv') -> pd.DataFrame:
    """
    Load the Telco Customer Churn dataset
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for missing values in the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Summary of missing values
    """
    missing = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum().values,
        'Missing_Percentage': (df.isnull().sum().values / len(df) * 100).round(2)
    })
    missing = missing[missing['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    
    if len(missing) == 0:
        print("✓ No missing values found")
    else:
        print(f"⚠ Found missing values in {len(missing)} columns:")
        print(missing.to_string(index=False))
    
    return missing


def get_feature_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Categorize features by type
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    dict
        Dictionary with categorized features
    """
    numerical = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove ID column from analysis
    if 'customerID' in categorical:
        categorical.remove('customerID')
    
    feature_types = {
        'numerical': numerical,
        'categorical': categorical,
        'binary': [col for col in categorical if df[col].nunique() == 2],
        'multi_class': [col for col in categorical if df[col].nunique() > 2]
    }
    
    print(f"\nFeature Types:")
    print(f"  Numerical: {len(feature_types['numerical'])} features")
    print(f"  Categorical: {len(feature_types['categorical'])} features")
    print(f"    - Binary: {len(feature_types['binary'])}")
    print(f"    - Multi-class: {len(feature_types['multi_class'])}")
    
    return feature_types


def calculate_summary_stats(df: pd.DataFrame, feature: str) -> pd.Series:
    """
    Calculate comprehensive summary statistics for a feature
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    feature : str
        Feature name
        
    Returns:
    --------
    pd.Series
        Summary statistics
    """
    if df[feature].dtype in ['int64', 'float64']:
        stats = pd.Series({
            'count': df[feature].count(),
            'mean': df[feature].mean(),
            'std': df[feature].std(),
            'min': df[feature].min(),
            '25%': df[feature].quantile(0.25),
            '50%': df[feature].median(),
            '75%': df[feature].quantile(0.75),
            'max': df[feature].max(),
            'skewness': df[feature].skew(),
            'kurtosis': df[feature].kurtosis()
        })
    else:
        value_counts = df[feature].value_counts()
        stats = pd.Series({
            'count': df[feature].count(),
            'unique': df[feature].nunique(),
            'top': value_counts.index[0],
            'top_freq': value_counts.values[0],
            'top_pct': (value_counts.values[0] / len(df) * 100).round(2)
        })
    
    return stats


def print_separator(title: str = "", char: str = "=", width: int = 80):
    """
    Print a formatted separator line
    
    Parameters:
    -----------
    title : str
        Optional title to display
    char : str
        Character to use for separator
    width : int
        Width of the separator
    """
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"\n{char * padding} {title} {char * padding}")
    else:
        print(f"\n{char * width}")


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format a decimal as percentage
    
    Parameters:
    -----------
    value : float
        Value to format (0-1 scale)
    decimals : int
        Number of decimal places
        
    Returns:
    --------
    str
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, decimals: int = 2) -> str:
    """
    Format a value as currency
    
    Parameters:
    -----------
    value : float
        Value to format
    decimals : int
        Number of decimal places
        
    Returns:
    --------
    str
        Formatted currency string
    """
    return f"${value:,.{decimals}f}"


def calculate_churn_rate(df: pd.DataFrame, group_by: str = None) -> pd.DataFrame:
    """
    Calculate churn rate overall or by a grouping variable
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with 'Churn' column
    group_by : str, optional
        Column to group by
        
    Returns:
    --------
    pd.DataFrame
        Churn rate summary
    """
    if group_by is None:
        churn_count = (df['Churn'] == 'Yes').sum()
        total = len(df)
        churn_rate = churn_count / total
        
        result = pd.DataFrame({
            'Segment': ['Overall'],
            'Total_Customers': [total],
            'Churned': [churn_count],
            'Retained': [total - churn_count],
            'Churn_Rate': [churn_rate],
            'Retention_Rate': [1 - churn_rate]
        })
    else:
        grouped = df.groupby(group_by)['Churn'].agg([
            ('Total_Customers', 'count'),
            ('Churned', lambda x: (x == 'Yes').sum()),
            ('Retained', lambda x: (x == 'No').sum())
        ]).reset_index()
        
        grouped['Churn_Rate'] = grouped['Churned'] / grouped['Total_Customers']
        grouped['Retention_Rate'] = grouped['Retained'] / grouped['Total_Customers']
        grouped.rename(columns={group_by: 'Segment'}, inplace=True)
        result = grouped
    
    return result


# Statistical test helper functions (for later notebooks)

def interpret_p_value(p_value: float, alpha: float = 0.05) -> str:
    """
    Interpret p-value result
    
    Parameters:
    -----------
    p_value : float
        P-value from statistical test
    alpha : float
        Significance level
        
    Returns:
    --------
    str
        Interpretation message
    """
    if p_value < alpha:
        return f"✓ SIGNIFICANT (p={p_value:.6f} < {alpha})"
    else:
        return f"✗ NOT SIGNIFICANT (p={p_value:.6f} ≥ {alpha})"


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size
    
    Parameters:
    -----------
    group1, group2 : np.ndarray
        Two groups to compare
        
    Returns:
    --------
    float
        Cohen's d value
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d effect size
    
    Parameters:
    -----------
    d : float
        Cohen's d value
        
    Returns:
    --------
    str
        Interpretation
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "Negligible effect"
    elif abs_d < 0.5:
        return "Small effect"
    elif abs_d < 0.8:
        return "Medium effect"
    else:
        return "Large effect"
