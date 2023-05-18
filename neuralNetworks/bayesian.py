from hyperopt import hp,  fmin, tpe, STATUS_OK, Trials
from keras.datasets import cifar10
from skimage.color import rgb2gray
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from keras import models, layers, optimizers, regularizers
from keras.layers import Conv2D, Flatten, MaxPooling2D
import numpy as np
from sklearn import model_selection, metrics

from keras.models import Sequential
from keras.layers import Dense, Dropout

from hpsklearn import HyperoptEstimator
from hpsklearn import any_classifier
from hpsklearn import any_preprocessing

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# load the dataset

# HYPERPARAMETERS: layers, nodes per layer, activation function, dropout, weight regularization, etc. 

#BAYESIAN OPTIMIZATION ON HYPERPARAMETER SPACE WITH VALIDATION SET
# model = HyperoptEstimator(classifier=any_classifier('cla'), 
#                           preprocessing=any_preprocessing('pre'), 
#                           algo=tpe.suggest, 
#                           max_evals=15, 
#                           trial_timeout=30)

def optimize_cnn(hyperparameter):
  # Define model using hyperparameters 
  cnn_model = Sequential([Conv2D(32, kernel_size=hyperparameter['conv_kernel_size'], activation='relu', input_shape=(32,32,1)), 
            MaxPooling2D(pool_size=(2,2)),
            Dropout(hyperparameter['dropout_prob']),
            Conv2D(64, kernel_size=hyperparameter['conv_kernel_size'], activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Dropout(hyperparameter['dropout_prob']), 
            Flatten(),
            Dense(32, activation='relu'), 
            Dense(10, activation='softmax'),])

  cnn_model.compile(optimizer=hyperparameter['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])

  cnn_model.fit(trainingSetX, tf.keras.utils.to_categorical(trainingSetY), epochs=2, batch_size=256, verbose=0)
  # Evaluate accuracy on validation data
  performance = cnn_model.evaluate(validationSetX, tf.keras.utils.to_categorical(validationSetY), verbose=0)

  print("Hyperparameters: ", hyperparameter, "Accuracy: ", performance[1])
  print("----------------------------------------------------")
  # We want to minimize loss i.e. negative of accuracy
  return({"status": STATUS_OK, "loss": -1*performance[1], "model":cnn_model})
  


(X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # convert to grayscale images
X_train = rgb2gray(X_train)
X_test = rgb2gray(X_test)


    #randomizing the training and testing set 
trainingSetX, validationSetX, trainingSetY, validationSetY = model_selection.train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# Define search space for hyper-parameters
space = {
    # The kernel_size for convolutions:
    'conv_kernel_size': hp.choice('conv_kernel_size', [1, 4, 5]),
    # Uniform distribution in finding appropriate dropout values
    'dropout_prob': hp.uniform('dropout_prob', 0.15, 0.3),
    # Choice of optimizer 
    'optimizer': hp.choice('optimizer', ['Adam', 'sgd']),
}

trials = Trials()

# Find the best hyperparameters
best = fmin(
        optimize_cnn,
        space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=15,
    )

print("==================================")
print("Best Hyperparameters", best)

test_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']

performance = test_model.evaluate(X_test, tf.keras.utils.to_categorical(y_test))

print("==================================")
print("Test Accuracy: ", performance[1])

#     # Define search space for hyper-parameters
#     model = HyperoptEstimator(classifier=any_classifier('cla'), 
#                             preprocessing=any_preprocessing('pre'), 
#                             algo=tpe.suggest, 
#                             max_evals=10, 
#                             trial_timeout=30)

#     print(trainingSetY)

#     newTrainData = trainingSetX.reshape(40000, 32*32)
#     newTrainDataY = trainingSetY.reshape(-1,1)

#     ##ERROR: Not getting the right shape to fit the data 
#     model.fit(newTrainData, tf.keras.utils.to_categorical(newTrainDataY)) 


#     # Results
#     performance = model.evaluate(X_test, tf.keras.utils.to_categorical(y_test))
#     print("Model Loss on test set samples: {0}".format(performance[0]))
#     print("Model Accuracy on test set: {0}".format(performance[1]))


#     print(f"Train score: {model.score(trainingSetX, trainingSetY)}")
#     print("TEST SET SCORE",model.score(validationSetX, validationSetY))

# #SEE ACCURACY ON A TEST SET 

# if __name__ == '__main__':

  
#     optimize()