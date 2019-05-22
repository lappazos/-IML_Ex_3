import numpy as np


class Knn:
    def __init__(self, k):
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, x):
        e_dis = np.linalg.norm(np.abs(self.X - x), axis=1)
        neighbours = np.argsort(e_dis)
        y = np.array(self.y)[neighbours]
        k_nearest = y[:self.k]
        if np.sum(k_nearest) >= k_nearest.size:
            return 1
        else:
            return 0
