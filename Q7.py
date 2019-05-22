import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from knn import Knn

# Question b
data = pd.read_csv('spam.data.txt', header=None, sep=" ")
y = data.iloc[:, -1]
x = data.drop(data.columns[len(data.columns) - 1], axis=1)
for i in range(10):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1000)
    model = LogisticRegression().fit(x_train, y_train)
    results = model.predict_proba(x_test)[:, 0]
    prob_rate = np.argsort(results)
    y_test = np.array(y_test)[prob_rate]
    positive_indices = np.where(y_test == 1)[0]
    NP = positive_indices.size
    positive_indices = positive_indices + np.ones(NP)
    NN = y_test.size - NP
    TPR = (np.arange(1, NP + 1) / NP)
    FPR = (positive_indices - np.arange(1, NP + 1)) / NN
    TPR = np.append(np.append(np.array([0]), TPR), np.array([1]))
    FPR = np.append(np.append(np.array([0]), FPR), np.array([1]))
    plt.plot(FPR, TPR, lw=1, alpha=0.6)
plt.title('spam logistic regression classification ROC curve - times 10')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

# Question c
for k in (1, 2, 5, 10, 100):
    model = Knn(k)
    error = 0
    for i in range(10):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1000)
        model.fit(x_train, y_train)
        prediction = np.apply_along_axis(model.predict, 1, x_test)
        error += (np.sum(np.absolute(y_test - prediction)) / y_test.size)
    print(k, ' neighbours test error: ', error / 10)
