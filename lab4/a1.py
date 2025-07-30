import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler


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
df=pd.read_csv("ml_lab1\lab3\lab3_vpn_nonvpndataset.txt",names=col)

df.head()

#converting vpn to 1 and nonvpn to 0(binary target)
df["class1"]=(df["class1"]=="VPN").astype(int)
vpn=df[df["class1"]==1]
nonvpn=df[df["class1"]==0]




X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#a5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



# a5
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# a6
accuracy_k3 = knn.score(X_test, y_test)
print("accuracy for k=3", accuracy_k3)

# a7
y_pred_k3 = knn.predict(X_test)
print(" prediction for test", y_pred_k3)



conf_mat = confusion_matrix(y_test, y_pred_k3)
print("confusion matrix -", conf_mat)
print("classification ", classification_report(y_test, y_pred_k3))


