from data import X_train, y_train, X_test, y_test, columns
import numpy as np
from unit_linear_regression import UnitLinearRegression
from multi_linear_regression import MultiLinearRegression
from closed_form_solution import ClosedFormSolution
from polynomial_regression import PolynomialRegression
from sparse_regression import SparseRegression

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
    m, b = model.coef()
    mse_train = model.loss('train')
    mse_test = model.loss('test')
    print('Number of iteration: ', itr)
    print(f'Coef m={m} and b={b}')
    print(f"Training loss: { mse_train }")
    print(f"Test loss: { mse_test }")
    sorted_models.append(
        {'index': feature + 1, 'feature_name': feature_name,
            'mse_train': mse_train, 'mse_test': mse_test, 'm_uni': m}
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
for res in sorted_models:
    if type(res['index']) == int:
        res['m_multi'] = a[res['index']]
print('------------------------------------------------------')
print('Result summary, sorted by mse on train')
for model_summary in sorted_models:
    print(model_summary)

print('------------------------------------------------------')
print('------------------------------------------------------')
print('Extra Credit - Closed Form Solution')
closed_form = ClosedFormSolution(X_train, y_train, X_test, y_test)
# Not really train, just calculate result
closed_form.train()
print(f"Training loss: { closed_form.loss('train') }")
print(f"Test loss: { closed_form.loss('test') }")
print(f"Coef: {closed_form.coef()}")

print('------------------------------------------------------')
print('Extra Credit - Polynomial Regression')
polynomial_model = PolynomialRegression(X_train, y_train, X_test, y_test)
polynomial_model.train()
print(f"Training loss: { polynomial_model.loss('train') }")
print(f"Test loss: { polynomial_model.loss('test') }")
print(f"Coef: {polynomial_model.coef()}")

print('------------------------------------------------------')
print('Extra Credit - Sparse Regression')
sparse_model = SparseRegression(X_train, y_train, X_test, y_test)
sparse_model.train()
print(f"Training loss: { sparse_model.loss('train') }")
print(f"Test loss: { sparse_model.loss('test') }")
print(f"Coef: {sparse_model.coef()}")
