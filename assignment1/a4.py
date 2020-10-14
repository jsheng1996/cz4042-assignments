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
num_neurons = 25
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

#Decay = 0
timings_model_0 = []
accuracies_model_0 = []
for fold in range(no_folds):

    histories = {}
    start, end = fold*nf, (fold+1)*nf
    x_test, y_test = x[start:end], y[start:end]
    x_train  = np.append(x[:start], x[end:], axis=0)
    y_train = np.append(y[:start], y[end:], axis=0)

    # Create the model
    model_0 = keras.Sequential([
        keras.layers.Dense(input_shape=(21,), units=num_neurons, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
    ])

    #Compile the model
    model_0.compile(optimizer='SGD',
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])



    # Train the model
    tic = time.clock()
    history =model_0.fit(x_train, y_train,
                                            validation_data = (x_test, y_test),
                                            epochs=epochs,
                                            verbose = 0,
                                            batch_size=batch_size)

    toc = time.clock()
    print('Training complete for fold {}'.format(fold+1))

    time_per_epoch = (toc-tic)/1000
    timings_model_0.append(time_per_epoch)
    accuracies_model_0.append(history.history['accuracy'])
model_0_accuracy =  get_mean_accuracy(accuracies_model_0)
accuracies['model_0'] = model_0_accuracy
timings.append(np.mean(timings_model_0))

#Decay = 1e-3
timings_model_3 = []
accuracies_model_3 = []
for fold in range(no_folds):

    histories = {}
    start, end = fold*nf, (fold+1)*nf
    x_test, y_test = x[start:end], y[start:end]
    x_train  = np.append(x[:start], x[end:], axis=0)
    y_train = np.append(y[:start], y[end:], axis=0)

    # Create the model
    model_3 = keras.Sequential([
        keras.layers.Dense(input_shape=(21,), units=num_neurons, activation='relu', kernel_regularizer=l2(1e-3)),
        keras.layers.Dense(3, activation='softmax')
    ])

    #Compile the model
    model_3.compile(optimizer='SGD',
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])



    # Train the model
    tic = time.clock()
    history =model_3.fit(x_train, y_train,
                                            validation_data = (x_test, y_test),
                                            epochs=epochs,
                                            verbose = 0,
                                            batch_size=batch_size)

    toc = time.clock()
    print('Training complete for fold {}'.format(fold+1))

    time_per_epoch = (toc-tic)/1000
    timings_model_3.append(time_per_epoch)
    accuracies_model_3.append(history.history['accuracy'])
model_3_accuracy =  get_mean_accuracy(accuracies_model_3)
accuracies['model_3'] = model_3_accuracy
timings.append(np.mean(timings_model_3))

#Decay = 1e-6
timings_model_6 = []
accuracies_model_6 = []
for fold in range(no_folds):

    histories = {}
    start, end = fold*nf, (fold+1)*nf
    x_test, y_test = x[start:end], y[start:end]
    x_train  = np.append(x[:start], x[end:], axis=0)
    y_train = np.append(y[:start], y[end:], axis=0)

    # Create the model
    model_6 = keras.Sequential([
        keras.layers.Dense(input_shape=(21,), units=num_neurons, activation='relu', kernel_regularizer=l2(1e-6)),
        keras.layers.Dense(3, activation='softmax')
    ])

    #Compile the model
    model_6.compile(optimizer='SGD',
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])



    # Train the model
    tic = time.clock()
    history =model_6.fit(x_train, y_train,
                                            validation_data = (x_test, y_test),
                                            epochs=epochs,
                                            verbose = 0,
                                            batch_size=batch_size)

    toc = time.clock()
    print('Training complete for fold {}'.format(fold+1))

    time_per_epoch = (toc-tic)/1000
    timings_model_6.append(time_per_epoch)
    accuracies_model_6.append(history.history['accuracy'])
model_6_accuracy =  get_mean_accuracy(accuracies_model_6)
accuracies['model_6'] = model_6_accuracy
timings.append(np.mean(timings_model_6))


#Decay = 1e-9
timings_model_9 = []
accuracies_model_9 = []
for fold in range(no_folds):

    histories = {}
    start, end = fold*nf, (fold+1)*nf
    x_test, y_test = x[start:end], y[start:end]
    x_train  = np.append(x[:start], x[end:], axis=0)
    y_train = np.append(y[:start], y[end:], axis=0)

    # Create the model
    model_9 = keras.Sequential([
        keras.layers.Dense(input_shape=(21,), units=num_neurons, activation='relu', kernel_regularizer=l2(1e-9)),
        keras.layers.Dense(3, activation='softmax')
    ])

    #Compile the model
    model_9.compile(optimizer='SGD',
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])



    # Train the model
    tic = time.clock()
    history =model_9.fit(x_train, y_train,
                                            validation_data = (x_test, y_test),
                                            epochs=epochs,
                                            verbose = 0,
                                            batch_size=batch_size)

    toc = time.clock()
    print('Training complete for fold {}'.format(fold+1))

    time_per_epoch = (toc-tic)/1000
    timings_model_9.append(time_per_epoch)
    accuracies_model_9.append(history.history['accuracy'])
model_9_accuracy =  get_mean_accuracy(accuracies_model_9)
accuracies['model_9'] = model_9_accuracy
timings.append(np.mean(timings_model_9))

#Decay = 1e-12
timings_model_12 = []
accuracies_model_12 = []
for fold in range(no_folds):

    histories = {}
    start, end = fold*nf, (fold+1)*nf
    x_test, y_test = x[start:end], y[start:end]
    x_train  = np.append(x[:start], x[end:], axis=0)
    y_train = np.append(y[:start], y[end:], axis=0)

    # Create the model
    model_12 = keras.Sequential([
        keras.layers.Dense(input_shape=(21,), units=num_neurons, activation='relu', kernel_regularizer=l2(1e-12)),
        keras.layers.Dense(3, activation='softmax')
    ])

    #Compile the model
    model_12.compile(optimizer='SGD',
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])



    # Train the model
    tic = time.clock()
    history =model_12.fit(x_train, y_train,
                                            validation_data = (x_test, y_test),
                                            epochs=epochs,
                                            verbose = 0,
                                            batch_size=batch_size)

    toc = time.clock()
    print('Training complete for fold {}'.format(fold+1))

    time_per_epoch = (toc-tic)/1000
    timings_model_12.append(time_per_epoch)
    accuracies_model_12.append(history.history['accuracy'])
model_12_accuracy =  get_mean_accuracy(accuracies_model_12)
accuracies['model_12'] = model_12_accuracy
timings.append(np.mean(timings_model_12))

res = {}
res['model_0'] = accuracies['model_0']
res['model_3'] = accuracies['model_3']
res['model_6'] = accuracies['model_6']
res['model_9'] = accuracies['model_9']
res['model_12'] = accuracies['model_12']
res['timings'] = timings

pickle.dump(res, open( "a3.p", "wb" ))
print("Job complete")
