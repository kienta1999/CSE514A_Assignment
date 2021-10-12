import pandas as pd
import numpy as np

data = pd.read_excel("data/Concrete_Data.xls")
columns = data.columns
train = np.array(data[:900])
test = np.array(data[900:])
X_train, y_train = train[:, :-1], train[:, -1]
X_test, y_test = test[:, :-1], test[:, -1]
