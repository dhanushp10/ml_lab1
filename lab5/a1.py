


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import random

col = [
    "duration",
    "total_fiat",
    "total_biat",
    "min_fiat",
    "min_biat",
    "max_fiat",
    "max_biat",
    "mean_fiat",
    "mean_biat",
    "flowPktsPerSecond",
    "flowBytesPerSecond",
    "min_flowiat",
    "max_flowiat",
    "mean_flowiat",
    "std_flowiat",
    "min_active",
    "mean_active",
    "max_active",
    "std_active",
    "min_idle",
    "mean_idle",
    "max_idle",
    "std_idle",
    "class1"
]
df=pd.read_csv("ml_lab1/lab3/lab3_vpn_nonvpndataset.txt",names=col)

#shuffling of data using frac
# res = df.sample(frac=1).reset_index(drop=True)

X,y=df.loc[:,["min_idle"]],df["std_idle"]


xtrain,xtest,ytrain,ytest=train_test_split (X, y, test_size=0.33, random_state=42,shuffle=True)


reg =  LinearRegression().fit(xtrain, ytrain)
y_train_pred = reg.predict(xtest)
print(len(ytest))
print(len(y_train_pred))


#a2
def evaluate_price_prediction(actual, predicted): # Function to calculate metrics
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return mse, rmse, mape, r2

mse,rmse,mape,r2=evaluate_price_prediction(ytest,y_train_pred)

print(mse,rmse,r2,mape)

#a3
A,b=df.loc[:,["min_idle","mean_idle","max_idle"]],df["std_idle"]
xtrain,xtest,ytrain,ytest=train_test_split (A, b, test_size=0.33, random_state=42,shuffle=True)


reg =  LinearRegression().fit(xtrain, ytrain)
y_train_pred = reg.predict(xtest)
print(len(ytest))
print(len(y_train_pred))


def evaluate_price_prediction(actual, predicted): # Function to calculate metrics
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return mse, rmse, mape, r2

mse,rmse,mape,r2=evaluate_price_prediction(ytest,y_train_pred)

print(mse,rmse,r2,mape)

#a4
from sklearn.cluster import KMeans

attr=df.loc[:,"duration":"std_idle" ]
print(attr)

kmeans = KMeans(n_clusters=2, random_state=0,n_init="auto").fit(attr)
print(kmeans.labels_)
print(kmeans.cluster_centers_)


