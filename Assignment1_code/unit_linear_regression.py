# from data import X_train, y_train, X_test, y_test
import numpy as np
import random
import matplotlib.pyplot as plt


class UnitLinearRegression:
    def __init__(self, x_train, y_train, x_test, y_test, step_size=0.0000000001):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        random.seed(8)
        self.m = 0
        self.b = 0
        self.trained = False
        self.step_size = step_size
        self.n_train = self.x_train.shape[0]
        self.n_test = self.x_test.shape[0]

    def train(self):
        prev_loss = float('inf')
        loss = self.loss('train')
        itr = 0
        threshold = 0.01
        while prev_loss - loss > threshold:
            y_pred = self.m * self.x_train + self.b
            self.m = self.m - self.step_size * \
                np.sum(self.x_train * (y_pred - self.y_train)) * \
                2 / self.n_train
            self.b = self.b - self.step_size * \
                np.sum(y_pred - self.y_train) * 2 / self.n_train
            prev_loss = loss
            loss = self.loss('train')
            itr += 1
        self.trained = True
        return itr

    def loss(self, type):
        if type == 'train':
            return np.sum((self.m * self.x_train + self.b - self.y_train) ** 2) / self.n_train
        elif type == 'test':
            return 0.5 * np.sum((self.m * self.x_test + self.b - self.y_test) ** 2) / self.n_test

    def coef(self):
        if not self.trained:
            return None, None
        return self.m, self.b

    def score(self):
        if not self.trained:
            return None, None
        ss_res = self.loss('test')
        y_test_avg = np.mean(self.y_test)
        ss_tot = np.sum((self.y_test - y_test_avg) ** 2)
        return 1 - ss_res / ss_tot

    def predict(self, x):
        return self.m * x + self.b

    def plot(self, feature, feature_name):
        plt.scatter(self.x_train, self.y_train)
        x_range = np.arange(np.min(self.x_train), np.max(self.x_train), 0.1)
        y_range = self.predict(x_range)
        plt.plot(x_range, y_range, c='orange')
        # plt.show()
        plt.savefig(f"./plot/{feature}_regression.png")
        plt.clf()
