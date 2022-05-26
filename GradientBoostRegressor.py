import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

data = np.genfromtxt("Modified.txt", skip_header=22)

attribute = data[:len(data), :len(data[0])-1]
target = data[:len(data), len(data[0])-1]
# Split Data
X_train, X_test, y_train, y_test = train_test_split(attribute, target, random_state=42)

depths = range(1, 10)
training_accuracy = []
test_accuracy = []

for depth in depths:
    gradient = GradientBoostingRegressor(max_depth=depth).fit(X_train, y_train)
    training_accuracy.append(gradient.score(X_train, y_train))
    test_accuracy.append(gradient.score(X_test, y_test))

plt.plot(depths, training_accuracy, label="training accuracy")
plt.plot(depths, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Num of Estimators")
plt.legend()
plt.show()