import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

data = np.genfromtxt("Modified.txt", skip_header=22)

attribute = data[:len(data), :len(data[0])-1]
target = data[:len(data), len(data[0])-1]
# Split Data
X_train, X_test, y_train, y_test = train_test_split(attribute, target, random_state=42)

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
