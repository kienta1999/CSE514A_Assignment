from multi_linear_regression import MultiLinearRegression
import numpy as np


class PolynomialRegression(MultiLinearRegression):
    def __init__(self, X_train, y_train, X_test, y_test, step_size=0.0000000000007):
        X_train_squared = np.c_[X_train, X_train ** 2]
        X_test_squared = np.c_[X_test, X_test ** 2]
        super().__init__(X_train_squared, y_train,
                         X_test_squared, y_test, step_size=step_size)
