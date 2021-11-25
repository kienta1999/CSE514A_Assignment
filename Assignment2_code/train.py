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
print('best k value for KNN', best_k, 'with mean accuracy', max(accuracies))
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
print('best max_depth value for Decision Tree', best_max_depth, 'with mean accuracy', max(accuracies))
best_models.append({
    'name': 'Decision Tree',
    'model': DecisionTreeClassifier(random_state=0, max_depth=best_max_depth),
})

print('--------------------------------------------------------------------------------------')
print('Model: Random Forest')
from sklearn.ensemble import RandomForestClassifier

MAX_MAX_DEPTH = 30
accuracies = np.zeros(MAX_MAX_DEPTH)
for max_depth in range(1, MAX_MAX_DEPTH + 1):
    model = RandomForestClassifier(random_state=0, max_depth=max_depth)
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
plt.title('Cross validation result of Random Forest')
plt.savefig('./plots/Random-Forest.png')
plt.clf()

# Save best model
best_max_depth = np.argmax(accuracies) + 1
print('best max_depth value for Random Forest', best_max_depth, 'with mean accuracy', max(accuracies))
best_models.append({
    'name': 'Random Forest',
    'model': RandomForestClassifier(random_state=0, max_depth=best_max_depth),
})

print('--------------------------------------------------------------------------------------')
print('Model: SVM using the polynomial kernel')

from sklearn.svm import SVC
MAX_DEG = 30
accuracies = np.zeros(MAX_DEG)
for degree in range(1, MAX_DEG + 1):
    model = SVC(kernel='poly', degree=degree)
    kf = KFold(n_splits=NUM_FOLDS)
    mean_accuracy = 0
    for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_test_fold = y_train_1d[train_index], y_train_1d[test_index]
        model.fit(X_train_fold, y_train_fold)
        mean_accuracy += accuracy(model, X_test_fold, y_test_fold)
    mean_accuracy /= NUM_FOLDS
    accuracies[degree - 1] = mean_accuracy
# Plot accuracy
plt.plot(range(1, MAX_DEG + 1), accuracies)
plt.xlabel('Degree')
plt.ylabel('Accuracy')
plt.title('Cross validation result of SVM using the polynomial kernel')
plt.savefig('./plots/Polynomial-SVM.png')
plt.clf()

# Save best model
best_degree = np.argmax(accuracies) + 1
print('best degree value for Polynomial SVM', best_degree, 'with mean accuracy', max(accuracies))
best_models.append({
    'name': 'Polynomial SVM',
    'model': SVC(kernel='poly', degree=best_degree),
})

print('--------------------------------------------------------------------------------------')
print('Model: SVM using the RBF kernel')

from sklearn.svm import SVC
C_values = np.arange(0.1,3.1,0.1)
accuracies = np.zeros(len(C_values))
for i, C in enumerate(C_values):
    model = SVC(kernel='rbf', C=C)
    kf = KFold(n_splits=NUM_FOLDS)
    mean_accuracy = 0
    for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_test_fold = y_train_1d[train_index], y_train_1d[test_index]
        model.fit(X_train_fold, y_train_fold)
        mean_accuracy += accuracy(model, X_test_fold, y_test_fold)
    mean_accuracy /= NUM_FOLDS
    accuracies[i] = mean_accuracy
# Plot accuracy
plt.plot(C_values, accuracies)
plt.xlabel('C - the lower C is, the higher strength of regularization')
plt.ylabel('Accuracy')
plt.title('Cross validation result of SVM using the rbf kernel')
plt.savefig('./plots/RBF-SVM.png')
plt.clf()

# Save best model
best_C = C_values[np.argmax(accuracies)]
print('best C value for RBF SVM', best_C, 'with mean accuracy', max(accuracies))
best_models.append({
    'name': 'RBF SVM',
    'model': SVC(kernel='rbf', C=best_C),
})

