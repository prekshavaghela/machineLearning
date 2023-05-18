from keras.datasets import cifar10
from skimage.color import rgb2gray
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from keras import models, layers, optimizers, regularizers
import numpy as np
from sklearn import model_selection, metrics


from keras.models import Sequential
from keras.layers import Dense

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# load the dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# convert to grayscale images
X_train = rgb2gray(X_train)
X_test = rgb2gray(X_test)

#### “multilayer perceptron” to perform the object classification
#### FNN comprises of all the pixels of the image.

# Use one of the five batches of the training data as a validation set.
lenTrain = len(X_train)

#randomizing the training and testing set 
trainingSetX, validationSetX, trainingSetY, validationSetY = model_selection.train_test_split(X_train, y_train, test_size=0.2, random_state=42)


### method for splitting: not randomly 
# Validation set of first 1/5th
# validationSetX = X_train[0:lenTrain//5]
# validationSetY = y_train[0:lenTrain//5]

# # training set of last 4/5th of train data
# trainingSetX = X_train[lenTrain//5::]
# trainingSetY = y_train[lenTrain//5::]

# HYPERPARAMETERS: layers, nodes per layer, activation function, dropout, weight regularization, etc. 

##### MODEL 3
secondsStart = time.time()

print("--------------------------")
print("MODEL 3")
print("--------------------------")


model = Sequential([Dense(812,input_shape=(1024, ), activation='relu'),
                    Dense(341, activation='sigmoid'),
                    Dense(123, activation='sigmoid'),
                    Dense(54, activation='sigmoid'),
                    Dense(25, activation='sigmoid'),
                    Dense(10, activation='softmax')])

# Compile model

# the loss on the train and validation set across the epochs of the FNN training
model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['accuracy'],)

print("Train image shape: ", trainingSetX.shape)
print(trainingSetY.shape)

# Flatten the images into vectors (1D) for feed forward network
model.summary()

#FIX ME
flatten_train_images = trainingSetX.reshape((-1, 32*32))
print(flatten_train_images )
flatten_test_images = validationSetX.reshape((-1, 32*32))

print("Train image shape: ", trainingSetX.shape, "Flattened image shape: ", flatten_train_images.shape)
print(trainingSetY.shape)

print(type(flatten_train_images[0,0]))
print(type(trainingSetY[0]))

# Train model
flatten_images_X = X_train.reshape((-1, 32*32))
vals = model.fit(flatten_images_X, tf.keras.utils.to_categorical(y_train),validation_split=0.2, epochs=30, batch_size=128,)


print("SECONDS FOR MODEL 3 fit: ", time.time() - secondsStart)

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

epochs = range(1, 31)
plt.plot(epochs, train_loss, 'r', label="Loss on Train Set")
plt.plot(epochs, val_loss, 'b', label="Loss on Validation Set")
plt.plot(epochs, acc_train, 'g', label="Accuracy on Train Set")
plt.plot(epochs, acc_val, label="Accuracy on Validation Set")
plt.title("Loss and Accuracy on Training and Validation Set Model 3")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

print()

print("SECONDS FOR MODEL 3 : ", time.time() - secondsStart)

flatten_test_images_REAL = X_test.reshape((-1, 32*32))
performance = model.evaluate(flatten_test_images_REAL, tf.keras.utils.to_categorical(y_test))
print("Model 3 Loss on test set samples: {0}".format(performance[0]))
print("Model 3 Accuracy on test set: {0}".format(performance[1]))

##CONFUSION MATRIX
print(len(y_test))
flatten_test_images_REAL = X_test.reshape((-1, 32*32))
predictions = model.predict(flatten_test_images_REAL)
valsPredicted=np.argmax(predictions,axis=1)
print(valsPredicted)
print(len(valsPredicted))

confusion_matrix_Test = metrics.confusion_matrix(y_test, valsPredicted)
print( "CONFUSION MATRIX")
print(confusion_matrix_Test)

sum =0
for i in confusion_matrix_Test:
    sum +=i

print(sum)

print("SECONDS FOR MODEL 3: ", time.time() - secondsStart)
plt.show()



