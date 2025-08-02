import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification


x=list()
y=list()
for i in range(20):
  x.append(random.randint(1,10))
  y.append(random.randint(1,10))

d={"x":x ,"y":y}
df=pd.DataFrame(data=d)

# classifying in basis of sum of x and y is greater 10 then blue else red for less than 10
df["class"]=(df["x"]+df["y"]).apply(lambda x: "blue" if x>10 else "red")


Xtrain,Ytrain=make_classification(n_samples=20,n_features=2,n_informative=2,n_redundant=0,n_clusters_per_class=1,random_state=62)

plt.scatter(df["x"],df["y"],c=df["class"],label="scatterplot")
plt.xlabel("X")
plt.ylabel("Y")
plt.title(" scatter plot for 2 different classes")
plt.show()

train=pd.DataFrame()
train["y"]=Ytrain
train["color"]=train["y"].apply(lambda x : "blue" if x==1 else "red")
plt.scatter(Xtrain[:,0],Xtrain[:,1],c=train["color"],label="scatterplot")
plt.xlabel("X")
plt.ylabel("Y")
plt.title(" scatter plot for 2 different classes")
plt.show()

  