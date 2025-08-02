import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

def evaluate_price_prediction(actual, predicted): # Function to calculate metrics
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return mse, rmse, mape, r2


df = pd.read_excel("ml_lab1\Lab Session Data.xlsx", sheet_name="Purchase data")

A = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values # Selecting only numeric columns
C = df['Payment (Rs)'].values

A_pinv = np.linalg.pinv(A) # Pseudo-inverse 
X = A_pinv @ C
predicted_C = A @ X # price prediction

mse, rmse, mape, r2 = evaluate_price_prediction(C, predicted_C) # Evaluating the metrics

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Percentage Error (MAPE):", mape)
print("R-squared (R²) Score:", r2)