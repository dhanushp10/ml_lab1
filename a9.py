import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df=pd.read_excel("ml_lab1\Lab Session Data.xlsx",sheet_name="thyroid0387_UCI")

df = df.replace({'t': 1, 'f': 0})

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

#  Z-score Standardization
standard_scaler = StandardScaler()
df_zscore_scaled = df.copy()
df_zscore_scaled[numeric_cols] = standard_scaler.fit_transform(df[numeric_cols])


print(df_zscore_scaled[numeric_cols].head())
