from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
from data import X_train, X_val, y_train, y_val, y_train_1d, y_val_1d, y_train_onehot, y_val_onehot
import time
import csv

def accuracy(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return np.mean(y_pred == y_test)

final_output = []

sklearn_models = [
    {
        'model': KNeighborsClassifier(n_neighbors=7),
        'name': 'KNN',
    },
    {
        'model': DecisionTreeClassifier(random_state=0, max_depth=2),
        'name': 'Decision Tree',
    },
    {
        'model': RandomForestClassifier(random_state=0, max_depth=4),
        'name': 'Random Forest',
    },
    {
        'model': SVC(kernel='poly', degree=1),
        'name': 'Polynomial SVM',
    },
    {
        'model': SVC(kernel='rbf', C=1.3),
        'name': 'RBF SVM',
    }
]


num_train_samples = X_train.shape[0]
num_val_samples = X_val.shape[0]
for model_infor in sklearn_models:
    # train timer start
    start_time_train = time.time()
    model = model_infor['model']
    model.fit(X_train, y_train_1d)
    # train timer end
    duration_train = time.time() - start_time_train
    start_time_val = time.time()
    validation_accuracy = accuracy(model, X_val, y_val_1d)
    duration_val = time.time() - start_time_val
    training_accuracy = accuracy(model, X_train, y_train_1d)
    final_output.append({
        'name': model_infor['name'],
        'duration_train_ms': duration_train * 1000,
        'duration_val_ms': duration_val * 1000,
        'training_accuracy': training_accuracy,
        'validation_accuracy': validation_accuracy,
        'num_train_samples': num_train_samples,
        'num_val_samples': num_val_samples,
    })


# sigmoid
sigmoid_model = keras.Sequential(name="sigmoid_model")
sigmoid_model.add(layers.Dense(28, activation="sigmoid"))
sigmoid_model.add(layers.Dense(28, activation="sigmoid"))
sigmoid_model.add(layers.Dense(28, activation="sigmoid"))
sigmoid_model.add(layers.Dense(2, activation="softmax"))
sigmoid_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#relu
relu_model = keras.Sequential(name="relu_model")
relu_model.add(layers.Dense(28, activation="relu"))
relu_model.add(layers.Dense(28, activation="relu"))
relu_model.add(layers.Dense(28, activation="relu"))
relu_model.add(layers.Dense(2, activation="softmax"))
relu_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
tensorflow_NN_models = [
    {
        'model': sigmoid_model,
        'name': 'NN with Sigmoid',
    }, 
    {
        'model': relu_model,
        'name': 'NN with ReLU',
    }
]

for model_infor in tensorflow_NN_models:
    # train timer start
    start_time_train = time.time()
    model = model_infor['model']
    model.fit(X_train, y_train_onehot, epochs=25)
    # train timer end
    duration_train = time.time() - start_time_train
    start_time_val = time.time()
    _, validation_accuracy = model.evaluate(X_val, y_val_onehot, verbose=2)
    duration_val = time.time() - start_time_val
    _, training_accuracy = model.evaluate(X_train, y_train_onehot, verbose=2)
    final_output.append({
        'name': model_infor['name'],
        'duration_train_ms': duration_train * 1000,
        'duration_val_ms': duration_val * 1000,
        'training_accuracy': training_accuracy,
        'validation_accuracy': validation_accuracy,
        'num_train_samples': num_train_samples,
        'num_val_samples': num_val_samples,
    })

csv_columns = final_output[0].keys()
csv_file = "final_validation_output.csv"
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in final_output:
            writer.writerow(data)
except IOError:
    print("I/O error")