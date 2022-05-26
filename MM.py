import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = np.genfromtxt("Modified.txt", skip_header=22)

attribute = data[:len(data), :len(data[0])-1]
target = data[:len(data), len(data[0])-1]
# Split Data
X_train, X_test, y_train, y_test = train_test_split(attribute, target, random_state=42)


# k-Neighbour Regression #
from sklearn.neighbors import KNeighborsRegressor

training_accuracy_n = []
test_accuracy_n = []
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsRegressor(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy_n.append(clf.score(X_train, y_train))
    # record generalization accuracy
    test_accuracy_n.append(clf.score(X_test, y_test))
plt.plot(neighbors_settings, training_accuracy_n, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy_n, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

# Linear Regression #
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train, y_train)
print("Linear Regression Model:")
print("Score for training data is ", lr.score(X_train, y_train))
print("Score for test data is ",lr.score(X_test, y_test))

# Ridge Regression #
from sklearn.linear_model import Ridge

alphas = np.linspace(0.001, 5, 20)
training_accuracy_alpha = []
test_accuracy_alpha = []

for a in alphas:
    r = Ridge(alpha=a).fit(X_train, y_train)
    training_accuracy_alpha.append(r.score(X_train, y_train))
    test_accuracy_alpha.append(r.score(X_test, y_test))

plt.plot(alphas, training_accuracy_alpha, label="training accuracy")
plt.plot(alphas, test_accuracy_alpha, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("alpha")
plt.legend()
plt.show()

# L1 Regression #
from sklearn.linear_model import Lasso
training_accuracy_alpha = []
test_accuracy_alpha = []

for a in alphas:
    l1 = Lasso(alpha=a).fit(X_train, y_train)
    training_accuracy_alpha.append(l1.score(X_train, y_train))
    test_accuracy_alpha.append(l1.score(X_test, y_test))

plt.plot(alphas, training_accuracy_alpha, label="training accuracy")
plt.plot(alphas, test_accuracy_alpha, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("alpha")
plt.legend()
plt.show()

# Decision Tree Regressor #
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor().fit(X_train, y_train)
print("\nDecision Tree Regressor: ")
print(tree.score(X_train, y_train))
print(tree.score(X_test, y_test))

# Random Forest Regressor #
from sklearn.ensemble import RandomForestRegressor

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

from sklearn.ensemble import GradientBoostingRegressor
gradient = GradientBoostingRegressor().fit(X_train, y_train)
print(forest.score(X_train, y_train))
print(forest.score(X_test, y_test))