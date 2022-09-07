import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import pickle

# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

epochs = 1000
batch_size = 32
no_folds = 5

seed = 10

np.random.seed(seed)
tf.random.set_seed(seed)

#read train data
train_input = np.genfromtxt('ctg_data_cleaned.csv', delimiter= ',')
x, y = train_input[1:, :21], train_input[1:,-1].astype(int)
x = scale(x, np.min(x, axis=0), np.max(x, axis=0))
y = y-1

no_data = len(x)
nf = no_data//no_folds

accuracies = {}
timings = []

def get_mean_accuracy(fold_accuracies):
    res =[]
    for i in range(epochs):
        sum=0
        for j in range(no_folds):
            sum+=fold_accuracies[j][i]
        res.append(sum/no_folds)

    return res

#Neuron count =5
timings_model_5 = []
accuracies_model_5 = []
for fold in range(no_folds):

    histories = {}
    start, end = fold*nf, (fold+1)*nf
    x_test, y_test = x[start:end], y[start:end]
    x_train  = np.append(x[:start], x[end:], axis=0)
    y_train = np.append(y[:start], y[end:], axis=0)

    # Create the model
    model_5 = keras.Sequential([
        keras.layers.Dense(input_shape=(21,), units=5, activation='relu', kernel_regularizer=l2(0.000001)),
        keras.layers.Dense(3, activation='softmax')
    ])

    #Compile the model
    model_5.compile(optimizer='SGD',
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])



    # Train the model
    tic = time.clock()
    history =model_5.fit(x_train, y_train,
                                            validation_data = (x_test, y_test),
                                            epochs=epochs,
                                            verbose = 0,
                                            batch_size=batch_size)

    toc = time.clock()
    print('Training complete for fold {}'.format(fold+1))

    time_per_epoch = (toc-tic)/1000
    timings_model_5.append(time_per_epoch)
    accuracies_model_5.append(history.history['accuracy'])
model_5_accuracy =  get_mean_accuracy(accuracies_model_5)
accuracies['model_5'] = model_5_accuracy
timings.append(np.mean(timings_model_5))

#Neuron count =10
timings_model_10 = []
accuracies_model_10 = []
for fold in range(no_folds):

    histories = {}
    start, end = fold*nf, (fold+1)*nf
    x_test, y_test = x[start:end], y[start:end]
    x_train  = np.append(x[:start], x[end:], axis=0)
    y_train = np.append(y[:start], y[end:], axis=0)

    # Create the model
    model_10 = keras.Sequential([
        keras.layers.Dense(input_shape=(21,), units=10, activation='relu', kernel_regularizer=l2(0.000001)),
        keras.layers.Dense(3, activation='softmax')
    ])

    #Compile the model
    model_10.compile(optimizer='SGD',
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])



    # Train the model
    tic = time.clock()
    history =model_10.fit(x_train, y_train,
                                            validation_data = (x_test, y_test),
                                            epochs=epochs,
                                            verbose = 0,
                                            batch_size=batch_size)

    toc = time.clock()
    print('Training complete for fold {}'.format(fold+1))

    time_per_epoch = (toc-tic)/1000
    timings_model_10.append(time_per_epoch)
    accuracies_model_10.append(history.history['accuracy'])
model_10_accuracy =  get_mean_accuracy(accuracies_model_10)
accuracies['model_10'] = model_10_accuracy
timings.append(np.mean(timings_model_10))

#Neuron count =15
timings_model_15 = []
accuracies_model_15 = []
for fold in range(no_folds):

    histories = {}
    start, end = fold*nf, (fold+1)*nf
    x_test, y_test = x[start:end], y[start:end]
    x_train  = np.append(x[:start], x[end:], axis=0)
    y_train = np.append(y[:start], y[end:], axis=0)

    # Create the model
    model_15 = keras.Sequential([
        keras.layers.Dense(input_shape=(21,), units=15, activation='relu', kernel_regularizer=l2(0.000001)),
        keras.layers.Dense(3, activation='softmax')
    ])

    #Compile the model
    model_15.compile(optimizer='SGD',
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])



    # Train the model
    tic = time.clock()
    history =model_15.fit(x_train, y_train,
                                            validation_data = (x_test, y_test),
                                            epochs=epochs,
                                            verbose = 0,
                                            batch_size=batch_size)

    toc = time.clock()
    print('Training complete for fold {}'.format(fold+1))

    time_per_epoch = (toc-tic)/1000
    timings_model_15.append(time_per_epoch)
    accuracies_model_15.append(history.history['accuracy'])
model_15_accuracy =  get_mean_accuracy(accuracies_model_15)
accuracies['model_15'] = model_15_accuracy
timings.append(np.mean(timings_model_15))

#Neuron count =20
timings_model_20 = []
accuracies_model_20 = []
for fold in range(no_folds):

    histories = {}
    start, end = fold*nf, (fold+1)*nf
    x_test, y_test = x[start:end], y[start:end]
    x_train  = np.append(x[:start], x[end:], axis=0)
    y_train = np.append(y[:start], y[end:], axis=0)

    # Create the model
    model_20 = keras.Sequential([
        keras.layers.Dense(input_shape=(21,), units=20, activation='relu', kernel_regularizer=l2(0.000001)),
        keras.layers.Dense(3, activation='softmax')
    ])

    #Compile the model
    model_20.compile(optimizer='SGD',
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])



    # Train the model
    tic = time.clock()
    history =model_20.fit(x_train, y_train,
                                            validation_data = (x_test, y_test),
                                            epochs=epochs,
                                            verbose = 0,
                                            batch_size=batch_size)

    toc = time.clock()
    print('Training complete for fold {}'.format(fold+1))

    time_per_epoch = (toc-tic)/1000
    timings_model_20.append(time_per_epoch)
    accuracies_model_20.append(history.history['accuracy'])
model_20_accuracy =  get_mean_accuracy(accuracies_model_20)
accuracies['model_20'] = model_20_accuracy
timings.append(np.mean(timings_model_20))

#Neuron count =25
timings_model_25 = []
accuracies_model_25 = []
for fold in range(no_folds):

    histories = {}
    start, end = fold*nf, (fold+1)*nf
    x_test, y_test = x[start:end], y[start:end]
    x_train  = np.append(x[:start], x[end:], axis=0)
    y_train = np.append(y[:start], y[end:], axis=0)

    # Create the model
    model_25 = keras.Sequential([
        keras.layers.Dense(input_shape=(21,), units=25, activation='relu', kernel_regularizer=l2(0.000001)),
        keras.layers.Dense(3, activation='softmax')
    ])

    #Compile the model
    model_25.compile(optimizer='SGD',
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])



    # Train the model
    tic = time.clock()
    history =model_25.fit(x_train, y_train,
                                            validation_data = (x_test, y_test),
                                            epochs=epochs,
                                            verbose = 0,
                                            batch_size=batch_size)

    toc = time.clock()
    print('Training complete for fold {}'.format(fold+1))

    time_per_epoch = (toc-tic)/1000
    timings_model_25.append(time_per_epoch)
    accuracies_model_25.append(history.history['accuracy'])
model_25_accuracy =  get_mean_accuracy(accuracies_model_25)
accuracies['model_25'] = model_25_accuracy
timings.append(np.mean(timings_model_25))

res = {}
res['model_5'] = accuracies['model_5']
res['model_10'] = accuracies['model_10']
res['model_15'] = accuracies['model_15']
res['model_20'] = accuracies['model_20']
res['model_25'] = accuracies['model_25']
res['timings'] = timings

pickle.dump(res, open( "a3.p", "wb" ))
print("Job complete")
