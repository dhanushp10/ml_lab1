import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


col = [
    "duration","total_fiat","total_biat","min_fiat","min_biat","max_fiat","max_biat",
    "mean_fiat","mean_biat","flowPktsPerSecond","flowBytesPerSecond","min_flowiat","max_flowiat",
    "mean_flowiat","std_flowiat","min_active","mean_active","max_active","std_active",
    "min_idle","mean_idle","max_idle","std_idle","class1"
]

df = pd.read_csv("lab3\lab3_vpn_nonvpndataset.txt", names=col)
df = df.dropna()
#converting vpn to 1 and nonvpn to 0(binary target)
df["class1"]=(df["class1"]=="VPN").astype(int)

X = df.loc[:, "duration":"std_idle"]
y = df["class1"]

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)


scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

#a2 
#Hyperparameter tuning using RandomizedSearchCV

from sklearn.svm import SVC
param_dist = {"C": [0.1, 1, 10], "gamma": [0.01, 0.1, 1], "kernel": ["rbf", "linear"]}
svm = SVC()
svm_cv = RandomizedSearchCV(svm, param_distributions=param_dist, cv=3, n_iter=5, random_state=42)
svm_cv.fit(xtrain, ytrain)

#a3)
#  multiple classifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

models = {
    "SVM": svm_cv.best_estimator_,
    "DecisionTree": DecisionTreeClassifier(max_depth=5),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "AdaBoost": AdaBoostClassifier(n_estimators=100),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    "CatBoost": CatBoostClassifier(verbose=0),
    "NaiveBayes": GaussianNB(),
    "MLP": MLPClassifier(max_iter=300)
}

results = []
for name, model in models.items():
    model.fit(xtrain, ytrain)
    ypred_train = model.predict(xtrain)
    ypred_test = model.predict(xtest)
    results.append([name,accuracy_score(ytrain, ypred_train),accuracy_score(ytest, ypred_test)])

results_df = pd.DataFrame(results, columns=["model", "train accuracy", "test accuracy"])
print("classification results:")
print(results_df)


