import random
import pandas as pd
import matplotlib.pyplot as plt

x=list()
y=list()
for i in range(20):
  x.append(random.randint(1,10))
  y.append(random.randint(1,10))

d={"x":x ,"y":y}
df=pd.DataFrame(data=d)

# classifying in basis of sum of x and y is greater 10 then blue else red for less than 10
df["class"]=(df["x"]+df["y"]).apply(lambda x: "blue" if x>10 else "red")


plt.scatter(df["x"],df["y"],c=df["class"],label="scatterplot")
plt.xlabel("X")
plt.ylabel("Y")
plt.title(" scatter plot for 2 different classes")
plt.show()


  