from data import X_train, X_val, y_train, y_val
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

def accuracy(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return np.mean(y_pred == y_test)

NUM_FOLDS = 10
y_train_1d = y_train.reshape(-1,)
y_val_1d = y_val.reshape(-1,)
best_models = []


print('--------------------------------------------------------------------------------------')
print('Model: KNN')

# k-nearest neighbors classifier
from sklearn.neighbors import KNeighborsClassifier
MAX_K = 30
accuracies = np.zeros(MAX_K)
# n_neighbors from 1 --> 20 inclusive
for n_neighbors in range(1, MAX_K + 1):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    kf = KFold(n_splits=NUM_FOLDS)
    mean_accuracy = 0
    for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_test_fold = y_train_1d[train_index], y_train_1d[test_index]
        model.fit(X_train_fold, y_train_fold)
        mean_accuracy += accuracy(model, X_test_fold, y_test_fold)
    mean_accuracy /= NUM_FOLDS
    accuracies[n_neighbors - 1] = mean_accuracy

# Plot accuracy
plt.plot(range(1, MAX_K + 1), accuracies)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Cross validation result of KNN')
plt.savefig('./plots/KNN.png')
plt.clf()

# Save best model
best_k = np.argmax(accuracies) + 1
print('best k value for KNN', best_k)
best_models.append({
    'name': 'KNN',
    'model': KNeighborsClassifier(n_neighbors=best_k),
})

print('--------------------------------------------------------------------------------------')
print('Model: Decision Tree')
from sklearn.tree import DecisionTreeClassifier

MAX_MAX_DEPTH = 30
accuracies = np.zeros(MAX_MAX_DEPTH)
for max_depth in range(1, MAX_MAX_DEPTH + 1):
    model = DecisionTreeClassifier(random_state=0, max_depth=max_depth)
    kf = KFold(n_splits=NUM_FOLDS)
    mean_accuracy = 0
    for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_test_fold = y_train_1d[train_index], y_train_1d[test_index]
        model.fit(X_train_fold, y_train_fold)
        mean_accuracy += accuracy(model, X_test_fold, y_test_fold)
    mean_accuracy /= NUM_FOLDS
    accuracies[max_depth - 1] = mean_accuracy
# Plot accuracy
plt.plot(range(1, MAX_MAX_DEPTH + 1), accuracies)
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.title('Cross validation result of Decision Tree')
plt.savefig('./plots/Decision-Tree.png')
plt.clf()

# Save best model
best_max_depth = np.argmax(accuracies) + 1
print('best max_depth value for Decision Tree', best_max_depth)
best_models.append({
    'name': 'Decision Tree',
    'model': DecisionTreeClassifier(random_state=0, max_depth=best_max_depth),
})