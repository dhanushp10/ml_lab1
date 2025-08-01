import random 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

x=list()
y=list()
for i in range(20):
  x.append(random.randint(1,10))
  y.append(random.randint(1,10))

d={"x":x ,"y":y}
df=pd.DataFrame(data=d)

# classifying in basis of sum of x and y is greater 10 then blue else red for less than 10
df["class"]=(df["x"]+df["y"]).apply(lambda x: "blue" if x>10 else "red")

xdata=list()
ydata=list()
v1=0
while(v1<=10):
  v2=0
  while(v2<=10):
    xdata.append(v1)
    ydata.append(v2)
    v2=round(v2+0.1,1)
  v1=round(v1+0.1,1)

# random.shuffle(xdata)
# random.shuffle(ydata)

xtrain=list(zip(x,y))
ytrain=df["class"]

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(xtrain, ytrain)



x_test=list(zip(xdata,ydata))
print(x_test)

pre=knn.predict(x_test)
print(pre)

plt.scatter(xdata,ydata,c=pre,label="scatterplot")
plt.xlabel("X")
plt.ylabel("Y")
plt.title(" scatter plot for 2 different classes")
plt.show()




# a5
for i in range(4,11):
  knn = KNeighborsClassifier(n_neighbors=i)
  knn.fit(xtrain, ytrain)



  x_test=list(zip(xdata,ydata))
  print(x_test)

  pre=knn.predict(x_test)
  print(pre)

  plt.scatter(xdata,ydata,c=pre,label="scatterplot")
  plt.xlabel("X")
  plt.ylabel("Y")
  plt.title(" scatter plot for 2 different classes")
  plt.show()