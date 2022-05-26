from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = np.genfromtxt("Modified.txt", skip_header=22)

attribute = data[:len(data), :len(data[0])-1]
target = data[:len(data), len(data[0])-1]
# Split Data
X_train, X_test, y_train, y_test = train_test_split(attribute, target, random_state=42)

training_accuracy_n = []
test_accuracy_n = []
NumEstimators = range(1,30)
for n in NumEstimators:
    forest = RandomForestRegressor(n_estimators=n).fit(X_train, y_train)
    training_accuracy_n.append(forest.score(X_train, y_train))
    test_accuracy_n.append(forest.score(X_test, y_test))
plt.plot(NumEstimators, training_accuracy_n, label="training accuracy")
plt.plot(NumEstimators, test_accuracy_n, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Num of Estimators")
plt.legend()
plt.show()