import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

df=pd.read_excel("ml_lab1\Lab Session Data.xlsx",sheet_name="thyroid0387_UCI")

# true or false values
binary_cols = []
for col in df.columns:
    unique_vals = df[col].dropna().unique()
    if set(unique_vals).issubset({'t', 'f'}):
        binary_cols.append(col)

# convert to binary
df_binary = df[binary_cols].replace({'t': 1, 'f': 0}).astype(int)

vec1 = df_binary.iloc[0]
vec2 = df_binary.iloc[1]

f11 = ((vec1 == 1) & (vec2 == 1)).sum()
f00 = ((vec1 == 0) & (vec2 == 0)).sum()
f10 = ((vec1 == 1) & (vec2 == 0)).sum()
f01 = ((vec1 == 0) & (vec2 == 1)).sum()


jc = f11 / (f11 + f10 + f01)
smc = (f11 + f00) / (f11 + f10 + f01 + f00)

print("jaccard coefficient is ",jc)
print("simple matching coefficient is", smc)
