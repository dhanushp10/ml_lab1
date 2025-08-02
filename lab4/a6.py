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
res = df.sample(frac=1).reset_index(drop=True)

#2 features extracted to plot the scatter plot
data=res.loc[:,["total_fiat","total_biat","class1"]]

data["color"]=(data["class1"]).apply(lambda x: "blue" if x=="VPN" else "red")


plt.scatter(data["total_fiat"],data["total_biat"],c=data["color"])
plt.title("Scatter plot for datapoints")
plt.xlabel("total_fiat")
plt.ylabel("total_biat")
plt.show()

xtrain=data.loc[:,["total_fiat","total_biat"]]
ytrain=data["class1"]
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(xtrain,ytrain)#trained data fitting

test_data=pd.read_csv("ml_lab1/2016_dataset/combine.txt",names=col)

test_data = test_data[test_data["class1"].isin(['MAIL' ,'STREAMING', 'VPN-MAIL','VPN-STREAMING'])]

print(test_data.shape)
xtest=test_data.loc[:,["total_fiat","total_biat"]]

pred=knn.predict(xtest)

xtest["class1"]=pred
xtest["color"]=xtest["class1"].apply(lambda x : "blue" if x=="VPN" else "red")


plt.scatter(xtest["total_fiat"],xtest["total_biat"],c=xtest["color"])
plt.title("Scatter with test data sets")
plt.xlabel("total_fiat")
plt.ylabel("total_biat")
plt.show()


#for different values of k
test=pd.DataFrame()
for i in range(4,11):
  knn = KNeighborsClassifier(n_neighbors=i)
  knn.fit(xtrain, ytrain)

  pred=knn.predict(xtest.loc[:,["total_fiat","total_biat"]])

  test["class1"]=pred
  test["color"]=test["class1"].apply(lambda x : "blue" if x=="VPN" else "red")


  plt.scatter(xtest["total_fiat"],xtest["total_biat"],c=xtest["color"])
  plt.title("Scatter with test data sets")
  plt.xlabel("total_fiat")
  plt.ylabel("total_biat")
  plt.show()


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


param_grid = {'n_neighbors': np.arange(1, 21)}  # k from 1 to 20


knn = KNeighborsClassifier()

grid_search = GridSearchCV(knn, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(xtrain, ytrain)  # Use your training data

print("Optimal k value:", grid_search.best_params_['n_neighbors']) 
print("Best cross-validated accuracy:", grid_search.best_score_)


  


 
  



