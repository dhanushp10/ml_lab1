import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

df=pd.read_excel("ml_lab1\Lab Session Data.xlsx",sheet_name="thyroid0387_UCI")

df_20 = df.iloc[:20].copy()
df_bin = df_20.replace({'t': 1, 'f': 0})

binary_cols = [col for col in df_bin.columns if set(df_bin[col].dropna().unique()).issubset({0, 1})]
df_binary = df_bin[binary_cols].astype(int)

n = df_binary.shape[0]
jc_matrix = np.zeros((n, n))
smc_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        v1, v2 = df_binary.iloc[i], df_binary.iloc[j]
        f11 = ((v1 == 1) & (v2 == 1)).sum()
        f00 = ((v1 == 0) & (v2 == 0)).sum()
        f10 = ((v1 == 1) & (v2 == 0)).sum()
        f01 = ((v1 == 0) & (v2 == 1)).sum()
        
        jc = f11 / (f11 + f10 + f01) if (f11 + f10 + f01) != 0 else 0
        smc = (f11 + f00) / (f11 + f10 + f01 + f00)
        
        jc_matrix[i][j] = jc
        smc_matrix[i][j] = smc

# Encode remaining categorical attributes
df_encoded = pd.get_dummies(df_bin, drop_first=True)

# Standardize
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_encoded), columns=df_encoded.columns)

cos_matrix = cosine_similarity(df_scaled)


#to plot 
def plot_heatmap(matrix, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=False, cmap='coolwarm', xticklabels=False, yticklabels=False)
    plt.title(title)
    plt.show()

plot_heatmap(jc_matrix, "Jaccard Coefficient Heatmap (First 20 Vectors)")
plot_heatmap(smc_matrix, "Simple Matching Coefficient Heatmap (First 20 Vectors)")
plot_heatmap(cos_matrix, "Cosine Similarity Heatmap (First 20 Vectors)")
