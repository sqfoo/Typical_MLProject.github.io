import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

data = np.genfromtxt("Modified.txt", skip_header=22)

attribute = data[:len(data), :len(data[0])-1]
target = data[:len(data), len(data[0])-1]
# Split Data
X_train, X_test, y_train, y_test = train_test_split(attribute, target, random_state=42)


tree = DecisionTreeRegressor().fit(X_train, y_train)
print("Decision Tree Regressor: ")
print(tree.score(X_train, y_train))
print(tree.score(X_test, y_test))