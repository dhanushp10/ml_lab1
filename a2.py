# to classify the model as rich or poor based the payments

import pandas as pd
import numpy as np

excel_file_path = "ml_lab1\Lab Session Data.xlsx"



df = pd.read_excel(excel_file_path,sheet_name='Purchase data')
print(df.head())

df['is_high'] = df['Payment (Rs)'] > 200


for i in range(len(df)):
    if df.loc[i,"Payment (Rs)"]>200:
        df.loc[i,"is_high"]="Rich"
    else:
        df.loc[i,"is_high"]="Poor"    

print(df)