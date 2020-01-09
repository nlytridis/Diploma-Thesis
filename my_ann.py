# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 17:32:47 2020

@author: nikol
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# http://www.bbci.de/competition/iv/desc_1.html Description and format of the dataset
# sampling rate: fs = 100, channels = 59

#cnt: the continuous EEG signals, size [time x channels]. The array is stored in datatype INT16.  
cnt = pd.read_csv('Dataset/BCICIV/BCICIV_calib_ds1a_cnt.csv')
raw_data = cnt.iloc[:,:].values

# pos: vector of positions of the cue in the EEG signals given in unit sample, length #cues 
pos = pd.read_csv('Dataset/BCICIV/BCICIV_calib_ds1a_pos.csv')
pos = np.array(pos).T

# y vector of target classes (-1 for class one or 1 for class two), length #cues 
y = pd.read_csv('Dataset/BCICIV/BCICIV_calib_ds1a_y.csv')
y = np.array(y).T
for i in list(range(200)):
    if y[i,0]<0: y[i,0] = 0
    
    
# MI trials starts at time pos[0] so data input must start at pos[0] time too.
# Χ : input data 200 trials X time X channels
X = np.zeros((200,800,59))

for i in list(range(200)):
    X[i] = raw_data [ pos[i,0] : (pos[i,0] + 800),:]  
            
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# To scale 3D array we reshape  to 2d 
x_trials, time_step, channels = X_train.shape
y_trials, time_step, channels = X_test.shape
X_train = X_train.reshape(-1,channels)
X_test = X_test.reshape(-1,channels)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#and afterward we reshape back to 3D
X_train = np.reshape(X_train, ( x_trials , time_step , channels ))
X_test = np.reshape(X_test, ( y_trials , time_step , channels ))

X_train = X_train.reshape(x_trials, time_step * channels)
X_test = X_test.reshape(y_trials, time_step * channels)



# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 35, kernel_initializer = 'uniform', activation = 'relu', input_dim = time_step*channels))

# Adding the second hidden layer
classifier.add(Dense(units = 25, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 100, epochs = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu', input_dim = channels*time_step))
    classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 100, epochs = 50)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN
# Dropout Regularization to reduce overfitting if needed

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu', input_dim = time_step*channels))
    classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [10, 50],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_



