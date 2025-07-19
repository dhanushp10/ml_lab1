import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

df=pd.read_excel("ml_lab1\Lab Session Data.xlsx",sheet_name="thyroid0387_UCI")

df_replaced = df.replace({'t': 1, 'f': 0})

non_numeric_cols = df_replaced.select_dtypes(include=['object', 'category']).columns

df_encoded = pd.get_dummies(df_replaced, columns=non_numeric_cols, drop_first=True)

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_encoded), columns=df_encoded.columns)

vec1 = df_scaled.iloc[0].values.reshape(1, -1)
vec2 = df_scaled.iloc[1].values.reshape(1, -1)

similarity = cosine_similarity(vec1, vec2)
print("cosine b/w to vectors", similarity[0][0])
