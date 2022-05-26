import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = np.genfromtxt("Modified.txt", skip_header=22)

attribute = data[:len(data), :len(data[0])-1]
target = data[:len(data), len(data[0])-1]
# Split Data
X_train, X_test, y_train, y_test = train_test_split(attribute, target, random_state=42)

lr = LinearRegression().fit(X_train, y_train)
print("Linear Regression Model:")
print("Score for training data is ", lr.score(X_train, y_train))
print("Score for test data is ",lr.score(X_test, y_test))