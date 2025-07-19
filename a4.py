import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as plt

excel_file_path = "ml_lab1\Lab Session Data.xlsx"

df = pd.read_excel(excel_file_path,sheet_name='thyroid0387_UCI')

print(df.head())
