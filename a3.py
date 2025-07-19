import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as plt

excel_file_path = "ml_lab1\Lab Session Data.xlsx"

df = pd.read_excel(excel_file_path,sheet_name='IRCTC Stock Price')

print(df.head())

#a)
#mean of the data
m=st.mean(df["Price"])
print("the mean of the data is " , m)
v=st.variance(df["Price"])
print("the variance of the data is ",v)


#b)
wed=df[df["Day"]=="Wed"]
mean_wed = st.mean(wed["Price"])
v_wed=st.variance(wed["Price"])
print("the mean on wed is " ,mean_wed,"the variance on wed is",v_wed)

#c)
apr=df[df["Month"]=="Apr"]
mean_apr = st.mean(apr["Price"])
v_apr=st.variance(apr["Price"])
print("the mean on apr is " ,mean_apr,"the variance on apr is",v_apr)

#d)
loss_count=len(df[df["Chg%"]<0])
profit_count=len(df[df["Chg%"]>=0])
print("the probability of loss",loss_count/(len(df["Chg%"])))

#e)
wprofit_count=len(wed[wed["Chg%"]>=0])
print("the probability of profit is  ", wprofit_count/len(wed["Chg%"]))

#f)
# conditional profit given that it is a wednesday
ans=(wprofit_count/len(wed["Chg%"]))/(profit_count/(len(df["Chg%"])))

#g)
data=df.loc[:,["Day","Chg%"]]
data.plot.scatter(x="Day",y="Chg%")
plt.show()