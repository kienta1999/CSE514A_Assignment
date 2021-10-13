from data import X_train, y_train, X_test, y_test, columns
import numpy as np
from unit_linear_regression import UnitLinearRegression
from multi_linear_regression import MultiLinearRegression

print('------------------------------------------------------')
print('Uni-variate linear regression')
for feature, feature_name in enumerate(columns[:-1]):
    print('------------------------------------------------------')
    print(f'feature chosen: index {feature + 1}, {feature_name}')
    x_train = X_train[:, feature]
    x_test = X_test[:, feature]
    model = UnitLinearRegression(x_train, y_train, x_test, y_test)
    itr = model.train()
    w, b = model.coef()
    print('Number of iteration: ', itr)
    print(f"Training loss: { model.loss('train') }")
    print(f"Test loss: { model.loss('test') }")
    model.plot(feature + 1, feature_name)

print('------------------------------------------------------')
print('Multi-variate linear regression')
multi_model = MultiLinearRegression(X_train, y_train, X_test, y_test)
itr = multi_model.train()
a = multi_model.coef()
print('Number of iteration: ', itr)
print(f"Training loss: { multi_model.loss('train') }")
print(f"Test loss: { multi_model.loss('test') }")
