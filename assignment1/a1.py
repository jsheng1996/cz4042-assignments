import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle

#Scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

epochs = 1000
batch_size = 32
num_neurons = 10

seed = 10

np.random.seed(seed)
tf.random.set_seed(seed)

#Read train data
train_input = np.genfromtxt('ctg_data_cleaned.csv', delimiter= ',')
x, y = train_input[1:, :21], train_input[1:,-1].astype(int)
x = scale(x, np.min(x, axis=0), np.max(x, axis=0))
y = y-1

#Split data into training and testing data
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.3, train_size=0.7, random_state=seed)


#Create the model
model = keras.Sequential([
    keras.layers.Dense(input_shape=(21,), units=num_neurons, activation='relu', kernel_regularizer=l2(0.000001)),
    keras.layers.Dense(units=3, activation='softmax')
])


model.compile(optimizer='SGD',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

histories ={}
#Train the model
histories['model'] = model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=epochs, verbose=0, batch_size=batch_size)

print('Training complete')
res = {}
res['accuracy']=histories["model"].history['accuracy']
res['val_accuracy']=histories["model"].history['val_accuracy']
pickle.dump(res, open( "a1.p", "wb" ))
print("Job complete")
