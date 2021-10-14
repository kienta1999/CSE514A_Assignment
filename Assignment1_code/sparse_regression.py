from multi_linear_regression import MultiLinearRegression
import numpy as np


class SparseRegression(MultiLinearRegression):
    def __init__(self, X_train, y_train, X_test, y_test, penalty=0.00003):
        super().__init__(X_train, y_train, X_test, y_test)
        self.penalty = penalty

    def train(self):
        itr = 0
        while itr < self.itr:
            y_pred = np.matmul(self.X_train, self.a.reshape(-1, 1))
            self.a = self.a - self.step_size * \
                np.matmul(y_pred.reshape(1, -1) - self.y_train.reshape(1, -1),
                          self.X_train).reshape(-1,) * 2 / self.n_train - self.penalty * np.sign(self.a)
            itr += 1
        self.trained = True
        return itr
