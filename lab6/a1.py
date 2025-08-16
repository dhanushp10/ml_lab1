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
import math

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


df=pd.read_csv(r"C:\Users\dhanu\OneDrive\Desktop\jup\ml_lab1\2016_dataset\combine.txt",names=col)
# df=pd.read_csv("/content/lab3_vpn_nonvpndataset.txt",names=col)
df=df.dropna()
class_columns=df["class1"].unique()
print(class_columns)
h=0
# print(len(df[df["class1"]==class_columns[]]))
X=df.loc[:,"duration":"std_idle"]
y=df["class1"]

pi=len(df[df["class1"]==class_columns[1]])/len(df)
print(pi)
p_values=[]
for i in range(0,len(class_columns)):
       pi=len(df[df["class1"]==class_columns[i]])/len(df)
       p_values.append(pi)

from scipy.stats import entropy

result=entropy(p_values,base=2)
print("the entropy of the dataset is ",result)


#a2

df = df.dropna()
target = "class1"


for col in df.columns:
    if col != target:
        df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna()

# ---------------- Entropy ----------------
def entropy(column):
    values, counts = np.unique(column, return_counts=True)
    total = len(column)
    ent = 0
    for c in counts:
        p = c / total
        ent += -p * math.log2(p)
    return ent

# for total - weighted entropy
def information_gain(data, split_attr, target_attr):
    total_entropy = entropy(data[target_attr])
    values, counts = np.unique(data[split_attr], return_counts=True)
    weighted_entropy = 0
    for i in range(len(values)):
        subset = data[data[split_attr] == values[i]][target_attr]
        weighted_entropy += (counts[i] / sum(counts)) * entropy(subset)
    return total_entropy - weighted_entropy

#bin using qcut
def bin_feature(series, bins=4):
    return pd.qcut(series, q=bins, labels=False, duplicates="drop")


gains = {}
features_to_check = df.columns[:-1]  

for col in features_to_check:
    binned_col = bin_feature(df[col], bins=4)
    gains[col] = information_gain(df.assign(temp=binned_col), "temp", target)

print("information gain for each feature =")
for k, v in gains.items():
    print(f"{k}: {v}")
print()
root = max(gains, key=gains.get)
print("the root  node is:", root)

#a4
def binning(series, bins=4, method="width"):
    if method == "width":
        return pd.cut(series, bins=bins, labels=False)
    elif method == "frequency":
        return pd.qcut(series, q=bins, labels=False, duplicates="drop")
    else:
        raise ValueError("Method must be 'width' or 'frequency'")


df["duration_width"] = binning(df["duration"], bins=4, method="width")
df["duration_freq"] = binning(df["duration"], bins=4, method="frequency")

print(df[["duration", "duration_width", "duration_freq"]])

#a5
def find_best_feature(data):
    ig_scores = {}
    for col in data.columns:
        if col != target:
            try:
                binned = bin_feature(data[col])
                ig_scores[col] = information_gain(data.assign(tmp=binned), "tmp", target)
            except Exception:
                continue
    return max(ig_scores, key=ig_scores.get)

# to build tree
def build_tree(data, depth=0, max_depth=3):
    print("  " * depth + f"Node depth {depth}, samples={len(data)}")

   
    if len(np.unique(data[target])) == 1:
        result = np.unique(data[target])[0]
        print("  " * depth + f"Leaf → {result}")
        return result

   
    if depth >= max_depth:
        result = data[target].mode()[0]
        print("  " * depth + f"Max depth reached → majority={result}")
        return result

    best_feature = find_best_feature(data)
    print("  " * depth + f"Best feature: {best_feature}")

    tree = {best_feature: {}}
    binned = bin_feature(data[best_feature])
    for val in np.unique(binned):
        subset = data[binned == val]
        if subset.empty:
            tree[best_feature][val] = data[target].mode()[0]
        else:
            tree[best_feature][val] = build_tree(subset, depth+1, max_depth)
    return tree


tree = build_tree(df, max_depth=3)

print("Final Decision Tree ->")
print(tree)

#a6
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt

model = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=0)
model.fit(X, y)
plt.figure(figsize=(18, 10))
plot_tree(model,feature_names=X.columns,class_names=y.unique(),filled=True,rounded=True,fontsize=9)
plt.title("Decision Tree Visualization ")
plt.show()

#a7
from sklearn.preprocessing import LabelEncoder
feature_x = "min_flowiat"
feature_y = "max_flowiat"

X_values = df[[feature_x, feature_y]].values
y_labels = df[target].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_labels)

decision_tree = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=0)
decision_tree.fit(X_values, y_encoded)

x_range_min, x_range_max = 0, 1e7
y_range_min, y_range_max = 0, 1e5

xx_grid, yy_grid = np.meshgrid(
    np.linspace(x_range_min, x_range_max, 300),
    np.linspace(y_range_min, y_range_max, 300)
)

grid_points = np.c_[xx_grid.ravel(), yy_grid.ravel()]
Z_predictions = decision_tree.predict(grid_points)
Z_predictions = Z_predictions.astype(float).reshape(xx_grid.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx_grid, yy_grid, Z_predictions, alpha=0.3, cmap="tab20")

for class_id in np.unique(y_encoded):
    plt.scatter(
        X_values[y_encoded == class_id, 0], X_values[y_encoded == class_id, 1],
        label=label_encoder.inverse_transform([class_id])[0], edgecolor="k", s=40
    )

plt.xlim(x_range_min, x_range_max)
plt.ylim(y_range_min, y_range_max)
plt.xlabel(feature_x)
plt.ylabel(feature_y)
plt.title("Decision Boundary with Zoomed Axes (Q7 - Option 2)")
plt.legend(title="Class")
plt.show()

    
    