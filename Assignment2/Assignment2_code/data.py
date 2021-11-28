import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer

header = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
data = pd.read_csv("breast-cancer-wisconsin.data", names=header, na_values="?")
data.drop(0, axis=1, inplace=True)
X = np.array(data.drop(10, axis=1))
y = np.array(data[10])
y = np.where(y == 2, 0, 1)

# impute missing values
imputer = KNNImputer(n_neighbors=1, weights="uniform")
X = imputer.fit_transform(X)

X_train, X_val, y_train_1d, y_val_1d = train_test_split(X, y, test_size=0.1, random_state=42)

y_train = y_train_1d.reshape(-1, 1)
y_val = y_val_1d.reshape(-1, 1)

y_train_onehot = np.array(pd.get_dummies(y_train_1d))
y_val_onehot = np.array(pd.get_dummies(y_val_1d))
print(X.shape)