import numpy as np
from multi_linear_regression import MultiLinearRegression


class ClosedFormSolution(MultiLinearRegression):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)

    def train(self):
        self.a = np.matmul(np.matmul(np.linalg.inv(
            np.matmul(self.X_train.T, self.X_train)), self.X_train.T), self.y_train.reshape(-1, 1))
        self.a = self.a.reshape(-1,)
        self.trained = True
