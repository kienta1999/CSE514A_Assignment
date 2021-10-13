# from data import X_train, y_train, X_test, y_test
import numpy as np
import random


class MultiLinearRegression:
    def __init__(self, X_train, y_train, X_test, y_test, step_size=0.0000001):
        self.y_train = y_train
        self.y_test = y_test
        self.m = np.shape(X_train)[1]
        self.n_train = X_train.shape[0]
        self.n_test = X_test.shape[0]
        self.X_train = np.c_[np.ones(self.n_train), X_train]
        self.X_test = np.c_[np.ones(self.n_test), X_test]
        random.seed(8)
        self.a = np.full(self.m + 1, 0)
        self.trained = False
        self.step_size = step_size

    def train(self):
        itr = 0
        while itr < 10000:
            y_pred = np.matmul(self.X_train, self.a.reshape(-1, 1))
            self.a = self.a - self.step_size * \
                np.matmul(y_pred.reshape(1, -1) - self.y_train.reshape(1, -1),
                          self.X_train).reshape(-1,) * 2 / self.n_train
            itr += 1
        self.trained = True
        return itr

    def loss(self, type):
        if type == 'train':
            return np.sum((np.matmul(self.X_train, self.a.reshape(-1, 1)).reshape(-1,) - self.y_train) ** 2) / self.n_train
        elif type == 'test':
            return np.sum((np.matmul(self.X_test, self.a.reshape(-1, 1)).reshape(-1,) - self.y_test) ** 2) / self.n_test
        print('only accept type "train" or "test"')

    def coef(self):
        if not self.trained:
            return None
        return self.a

    def predict(self, X):
        X_appeded = np.c_[np.ones(np.shape(X)[0]), X]
        return np.matmul(X_appeded, self.a.reshape(-1, 1)).reshape(-1,)
