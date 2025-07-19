import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore

df=pd.read_excel("ml_lab1\Lab Session Data.xlsx",sheet_name="thyroid0387_UCI")


numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

numeric_with_outliers = []
numeric_no_outliers = []

for col in numeric_cols:
    if df[col].isnull().sum() == len(df):
        continue
    col_data = df[col].dropna()
    z_scores = zscore(col_data)
    has_outliers = np.any(np.abs(z_scores) > 3)
    
    if has_outliers:
        numeric_with_outliers.append(col)
    else:
        numeric_no_outliers.append(col)


#filling with mean for columns with no outliners
for col in numeric_no_outliers:
    mean_val = df[col].mean()
    df[col].fillna(mean_val, inplace=True)

#filling with median for outliner columns
for col in numeric_with_outliers:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)

#category wise filling 
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)


print("Missing values after imputation:")
print(df.isnull().sum())
