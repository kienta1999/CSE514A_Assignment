from data import X_train, y_train, X_test, y_test, columns
import numpy as np
from unit_linear_regression import UnitLinearRegression
from multi_linear_regression import MultiLinearRegression

sorted_models = []

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
    mse_train = model.loss('train')
    mse_test = model.loss('test')
    print('Number of iteration: ', itr)
    print(f"Training loss: { mse_train }")
    print(f"Test loss: { mse_test }")
    sorted_models.append(
        {'index': feature + 1, 'feature_name': feature_name,
            'mse_train': mse_train, 'mse_test': mse_test}
    )
    model.plot(feature + 1, feature_name)

print('------------------------------------------------------')
print('Multi-variate linear regression')
multi_model = MultiLinearRegression(X_train, y_train, X_test, y_test)
itr = multi_model.train()
a = multi_model.coef()
mse_train = multi_model.loss('train')
mse_test = multi_model.loss('test')
print('Number of iteration: ', itr)
print(f"Training loss: { mse_train }")
print(f"Test loss: { mse_test }")
print(f"Coef: {multi_model.coef()}")
sorted_models.append(
    {'index': 'all', 'feature_name': 'all',
     'mse_train': multi_model.loss('train'), 'mse_test': multi_model.loss('test')}
)

sorted_models = sorted(sorted_models, key=lambda model: model['mse_train'])
print('------------------------------------------------------')
print('Result summary, sorted by mse on train')
for model_summary in sorted_models:
    print(model_summary)
