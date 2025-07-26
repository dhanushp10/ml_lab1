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
print(df.head())
print(df["class1"].unique())
print(df.shape)

#converting vpn to 1 and nonvpn to 0(binary target)
df["class1"]=(df["class1"]=="VPN").astype(int)
df.head()
vpn=df[df["class1"]==1]
nonvpn=df[df["class1"]==0]
print(vpn.shape)

#a2
# for label in col[:-1]:
#   plt.hist(df[df["class1"]==1][label] , color="blue",label="gamma",alpha=0.7,density=True)
#   plt.hist(df[df["class1"]==0][label] , color="red",label="gamma",alpha=0.7,density=True)
#   plt.title(label)
#   plt.xlabel(label)
#   plt.ylabel("prob")
#   plt.legend()
#   plt.show()


centroid1=np.mean(df[df["class1"]==1])
centroid2=np.mean(df[df["class1"]==0])
#mean
print("the mean for the vpn class is " ,np.mean(df[df["class1"]==1]))
print("the mean for the non vpn class is " ,np.mean(df[df["class1"]==0]))

#std
print("the std for the vpn class is " ,np.std(df[df["class1"]==1]))
print("the std for the non vpn class is " ,np.std(df[df["class1"]==0]))

np.linalg.norm(centroid1 - centroid2) 

#plot
v=np.histogram(vpn['duration'])
nv=np.histogram(nonvpn['duration'])
plt.hist(v)
plt.hist(nv)

#a3
X = df.iloc[:, :-1].values

x_vec1 = X[0]
x_vec2 = X[1]

r_values = range(1, 11)
distances = []
for r in r_values:
    dist = np.sum(np.abs(x_vec1 - x_vec2)**r)**(1/r)
    distances.append(dist)

print("feature vector1:", x_vec1)
print("feature vector2:", x_vec2)
print("dist for r=1 to 10", distances)

plt.plot(r_values, distances, marker='o')
plt.xlabel("r")
plt.ylabel("distance")
plt.grid(True)
plt.show()


#a4
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

# Predict for one sample
sample_vector = X_test[0].reshape(1, -1)
print("Sample Prediction:", knn.predict(sample_vector))

# A8: Vary k from 1 to 11 and make accuracy plot
k_values = range(1, 12)
accuracies = []

for k in k_values:
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)
    acc = neigh.score(X_test, y_test)
    accuracies.append(acc)

plt.plot(k_values, accuracies, marker='o')
plt.title('Accuracy vs k')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

# a9
conf_mat = confusion_matrix(y_test, y_pred_k3)
print("confusion matrix -", conf_mat)
print("classification ", classification_report(y_test, y_pred_k3))
