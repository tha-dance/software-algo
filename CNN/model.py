# This program is used to build the basic model structure
# This program refers to some ideas from 
# https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
# Data is fetched from: http://archive.ics.uci.edu/ml/machine-learning-databases/iris/

import numpy as np 
import pandas

# import os; 
# os.environ['KERAS_BACKEND'] = 'theano'

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

# Suppose the input is stored in csv format just like the online dataset 
dataframe = pandas.read_csv('input/iris.csv', header=0)
dataset = dataframe.values
len_data = len(dataset[0])
properties = dataset[:, 0:len_data-1].astype(float) # property
labels = dataset[:, len_data-1] # label

# one-hot encoding for labels 
encoder = LabelEncoder()
encoder.fit(labels)
encoder_label = encoder.transform(labels)

# 0 ==> [1,0,0] 1 ==> [0,1,0] 2==> [0,0,1]
one_hot_label = np_utils.to_categorical(encoder_label)

# Use simply fully connected feedforward neural network
def fully_connected_model():
    # create model
    model = Sequential()
    # build layers 
    model.add(Dense(8, input_dim=len_data-1, activation='relu'))
    model.add(Dense(3, activation='softmax')) # use softmax to represent predicted probability
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

estimator = KerasClassifier(build_fn=fully_connected_model, epochs=200, batch_size=5, verbose=0)

seed = 11
np.random.seed(seed)

kfold = KFold(n_splits=10, shuffle=True, random_state=seed) 
results = cross_val_score(estimator, properties, one_hot_label, cv=kfold)

print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


