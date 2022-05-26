import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

data = np.genfromtxt("Modified.txt", skip_header=22)

attribute = data[:len(data), :len(data[0])-1]
target = data[:len(data), len(data[0])-1]
# Split Data
X_train, X_test, y_train, y_test = train_test_split(attribute, target, random_state=42)

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