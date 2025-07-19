import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

excel_file_path = "ml_lab1\Lab Session Data.xlsx"

df = pd.read_excel(excel_file_path,sheet_name='thyroid0387_UCI')

print(df.dtypes)

#tsh,t3 measured ,t3,tt4,t4u,fit,tbg,referral source,condition 



# For ordinal encoding


# For one-hot encoding
nominal_cols = ["age","sex"]
df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)


numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

print(df[numeric_cols].describe())

#missing values
print(df.isnull().sum())

for col in numeric_cols:
    sns.boxplot(x=df[col])
    plt.title(f"Outlier check for {col}")
    plt.show()

print("Means:\n", df[numeric_cols].mean())
print("\nStandard Deviations:\n", df[numeric_cols].std())


