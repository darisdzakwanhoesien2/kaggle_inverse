import numpy as np
import pandas as pd
from scipy.stats import skew

def analyze_target(df, target):
    y = df[target]
    return {
        "skewness": skew(y.dropna()),
        "mean": y.mean(),
        "std": y.std(),
        "min": y.min(),
        "max": y.max()
    }

def missing_report(df):
    return df.isnull().mean().sort_values(ascending=False)

def detect_feature_types(df):
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    categorical = df.select_dtypes(exclude=np.number).columns.tolist()
    return numeric, categorical