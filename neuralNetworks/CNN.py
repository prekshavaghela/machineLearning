from keras.datasets import cifar10
from skimage.color import rgb2gray
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from keras import models, layers, optimizers, regularizers
from keras.layers import Conv2D, Flatten, MaxPooling2D
import numpy as np
from sklearn import model_selection, preprocessing


from keras.models import Sequential
from keras.layers import Dense

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# load the dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# convert to grayscale images
X_train = rgb2gray(X_train)
X_test = rgb2gray(X_test)

#### CNN 

# Use one of the five batches of the training data as a validation set.
lenTrain = len(X_train)

#randomizing the training and testing set 
trainingSetX, validationSetX, trainingSetY, validationSetY = model_selection.train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# HYPERPARAMETERS: layers, nodes per layer, activation function, dropout, weight regularization, etc. 

print("--------------------------")
print("MODEL 1")
print("--------------------------")

secondsStart = time.time()

# Define 2 groups of layers: features layer (convolutions) and classification layer


common_features = [Conv2D(32, kernel_size=6, activation='relu', input_shape=(32,32,1)), 
            MaxPooling2D(pool_size=(4,4)),
            Conv2D(64, kernel_size=4, activation='relu'),
            MaxPooling2D(pool_size=(2,2)), Flatten(),]
classifier = [Dense(32, activation='relu'), Dense(10, activation='softmax'),]

cnn_model = Sequential(common_features+classifier)

print(cnn_model.summary())  # Compare number of parameteres against FFN
cnn_model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['accuracy'],)

train_images_3d = trainingSetX.reshape(40000,32,32,1)
test_images_3d = validationSetX.reshape(10000,32,32,1)

##train model 
flatten_images_X = X_train.reshape(50000,32,32,1)
vals = cnn_model.fit(flatten_images_X, tf.keras.utils.to_categorical(y_train),validation_split=0.2, epochs=30, batch_size=128,)


print( vals.history.keys())
train_loss = vals.history['loss']
val_loss = vals.history['val_loss']
acc_train = vals.history['accuracy']
acc_val = vals.history['val_accuracy']

# performance = model.evaluate(flatten_train_images, tf.keras.utils.to_categorical(trainingSetY))
print("Model 1 Loss on training set samples: {0}".format(train_loss[len(train_loss)-1 ]))
print("Model 1 Accuracy on training set samples: {0}".format(acc_train[len(acc_train)-1 ]))

# performance = model.evaluate(flatten_test_images, tf.keras.utils.to_categorical(validationSetY))
print("Model 1 Loss on validation set samples: {0}".format(val_loss[len(val_loss)-1 ]))
print("Model 1 Accuracy on validation set samples: {0}".format(acc_val[len(acc_val)-1 ]))

print("SECONDS FOR MODEL 1: ", time.time() - secondsStart)

epochs = range(1, 31)
plt.plot(epochs, train_loss, 'r', label="Loss on Train Set")
plt.plot(epochs, val_loss, 'b', label="Loss on Validation Set")
plt.plot(epochs, acc_train, 'g', label="Accuracy on Train Set")
plt.plot(epochs, acc_val, label="Accuracy on Validation Set")
plt.title("Loss and Accuracy on Training and Validation Set Model 1")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
print()


print("SECONDS FOR MODEL 1: ", time.time() - secondsStart)
print()


print("--------------------------")
print("MODEL 2")
print("--------------------------")

secondsStart = time.time()

# Define 2 groups of layers: features layer (convolutions) and classification layer


common_features = [Conv2D(128, kernel_size=8, activation='tanh', strides =(2,2), input_shape=(32,32,1)), 
            MaxPooling2D(pool_size=(5,5)),
            Conv2D(64, kernel_size=2, activation='tanh', strides =(2,2)),
            MaxPooling2D(pool_size=(1,1)), Flatten(),]
classifier = [Dense(32, activation='tanh'), Dense(10, activation='softmax'),]

cnn_model = Sequential(common_features+classifier)

print(cnn_model.summary())  # Compare number of parameteres against FFN
cnn_model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['accuracy'],)

train_images_3d = trainingSetX.reshape(40000,32,32,1)
test_images_3d = validationSetX.reshape(10000,32,32,1)


##train model 
flatten_images_X = X_train.reshape(50000,32,32,1)
vals = cnn_model.fit(flatten_images_X, tf.keras.utils.to_categorical(y_train),validation_split=0.2, epochs=35, batch_size=128,)


print( vals.history.keys())
train_loss = vals.history['loss']
val_loss = vals.history['val_loss']
acc_train = vals.history['accuracy']
acc_val = vals.history['val_accuracy']

# performance = model.evaluate(flatten_train_images, tf.keras.utils.to_categorical(trainingSetY))
print("Model 2 Loss on training set samples: {0}".format(train_loss[len(train_loss)-1 ]))
print("Model 2 Accuracy on training set samples: {0}".format(acc_train[len(acc_train)-1 ]))

# performance = model.evaluate(flatten_test_images, tf.keras.utils.to_categorical(validationSetY))
print("Model 2 Loss on validation set samples: {0}".format(val_loss[len(val_loss)-1 ]))
print("Model 2 Accuracy on validation set samples: {0}".format(acc_val[len(acc_val)-1 ]))

print("SECONDS FOR MODEL 2: ", time.time() - secondsStart)

epochs = range(1, 36)
plt.plot(epochs, train_loss, 'r', label="Loss on Train Set")
plt.plot(epochs, val_loss, 'b', label="Loss on Validation Set")
plt.plot(epochs, acc_train, 'g', label="Accuracy on Train Set")
plt.plot(epochs, acc_val, label="Accuracy on Validation Set")
plt.title("Loss and Accuracy on Training and Validation Set Model 2")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
print()

print("--------------------------")
print("MODEL 3")
print("--------------------------")

secondsStart = time.time()

# Define 2 groups of layers: features layer (convolutions) and classification layer


common_features = [Conv2D(128, kernel_size=8, activation='relu', strides =(2,2), input_shape=(32,32,1)), 
            MaxPooling2D(pool_size=(5,5)),
            Conv2D(64, kernel_size=2, activation='relu', strides =(2,2)),
            Conv2D(32, kernel_size=1, activation='relu', strides =(1,1)),
            MaxPooling2D(pool_size=(1,1)), Flatten(),
            ]

classifier = [Dense(32, activation='sigmoid'), Dense(10, activation='softmax'),]

cnn_model = Sequential(common_features+classifier)

print(cnn_model.summary())  # Compare number of parameteres against FFN
cnn_model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['accuracy'],)

train_images_3d = trainingSetX.reshape(40000,32,32,1)
test_images_3d = validationSetX.reshape(10000,32,32,1)


##train model 
flatten_images_X = X_train.reshape(50000,32,32,1)
vals = cnn_model.fit(flatten_images_X, tf.keras.utils.to_categorical(y_train),validation_split=0.2, epochs=35, batch_size=128,)


print( vals.history.keys())
train_loss = vals.history['loss']
val_loss = vals.history['val_loss']
acc_train = vals.history['accuracy']
acc_val = vals.history['val_accuracy']

# performance = model.evaluate(flatten_train_images, tf.keras.utils.to_categorical(trainingSetY))
print("Model 3 Loss on training set samples: {0}".format(train_loss[len(train_loss)-1 ]))
print("Model 3 Accuracy on training set samples: {0}".format(acc_train[len(acc_train)-1 ]))

# performance = model.evaluate(flatten_test_images, tf.keras.utils.to_categorical(validationSetY))
print("Model 3 Loss on validation set samples: {0}".format(val_loss[len(val_loss)-1 ]))
print("Model 3 Accuracy on validation set samples: {0}".format(acc_val[len(acc_val)-1 ]))

print("SECONDS FOR MODEL 3: ", time.time() - secondsStart)

epochs = range(1, 36)
plt.plot(epochs, train_loss, 'r', label="Loss on Train Set")
plt.plot(epochs, val_loss, 'b', label="Loss on Validation Set")
plt.plot(epochs, acc_train, 'g', label="Accuracy on Train Set")
plt.plot(epochs, acc_val, label="Accuracy on Validation Set")
plt.title("Loss and Accuracy on Training and Validation Set Model 3")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
print()