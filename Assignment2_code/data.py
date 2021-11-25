import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

header = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
data = pd.read_csv("breast-cancer-wisconsin.data", names=header, na_values="?")
data.drop(0, axis=1, inplace=True)
data.dropna(inplace=True)
X = np.array(data.drop(10, axis=1))
y = np.array(data[10]).reshape(-1, 1)
# y = np.where(y == 2, -1, 1).reshape(-1, 1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
