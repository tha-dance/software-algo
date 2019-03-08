'''
    This program is used to build the basic model structure
    This program refers to some ideas from 
    https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
    Data is fetched from: http://archive.ics.uci.edu/ml/machine-learning-databases/iris/
'''

'''
TODO:
1. use one-hot encoding 
2. use k-fold
'''
import numpy as np 
import pandas

# import os; 
# os.environ['KERAS_BACKEND'] = 'theano'

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.models import model_from_json
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from keras.backend import manual_variable_initialization
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn import cross_validation
from sklearn import datasets

# Suppose the input is stored in csv format just like the online dataset 
dataframe = pandas.read_csv('input/iris.csv', header=0)
dataset = dataframe.values
label_names = datasets.load_iris().target_names

len_data = len(dataset[0])
feature = dataset[:, 0:len_data-1].astype(float) # property
label = dataset[:, len_data-1] # label

'''
    The reason that I convert numerical data then to one hot encoding is that 
    using numerical data may result in making model assume a natural ordering between categories
    like category 1 data will be in front of category 2 data 
    which will result in poor performance
'''
# one-hot encoding for labels 
encoder = LabelEncoder()
encoder.fit(label)
encoder_label = encoder.transform(label)
# 0 ==> [1,0,0] 1 ==> [0,1,0] 2==> [0,0,1]
one_hot_label = np_utils.to_categorical(encoder_label)

feature_train, feature_test, label_train, label_test = cross_validation.train_test_split(feature, label, test_size=0.2, random_state=4)

# Use simply fully connected feedforward neural network
def fully_connected_model():
    # create model
    model = Sequential()
    # build layers 
    model.add(Dense(8, input_dim=len_data-1, activation='relu'))
    model.add(Dense(3, activation='softmax')) # use softmax to represent predicted probability

    # If the effect is not so good, add more hidden layers to increase the accuracy level
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

estimator = KerasClassifier(build_fn=fully_connected_model, epochs=200, batch_size=119, verbose=0)

# Confusion matrix as the evaluation method 
model = fully_connected_model()
print(feature_train.shape)
# model.fit(feature_train, label_train, batch_size=119, epochs=200, verbose=0)
# label_pred = model.predict_classes(feature_test)
# print(label_pred)
estimator.fit(feature_train, label_train)
label_pred = estimator.predict(feature_test)
print(label_test)
print(label_pred)

# Use K-Fold validation
# Another evaluation method ==> accuracy score
seed = 11
np.random.seed(seed)

kfold = KFold(n_splits=10, shuffle=True, random_state=seed) 
results = cross_val_score(estimator, feature, one_hot_label, cv=kfold)

print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# model.fit(feature_train, label_train, epochs=200, batch_size=5, verbose=0)
model = estimator.model
model_label_pred = model.predict_classes(feature_test)
print(label_test)
print(model_label_pred)


result = confusion_matrix(label_test, label_pred)
print(result)
# the testing result :
# [[17  0  0]
#  [ 0  5  1]
#  [ 0  0  7]]

one_line_label_pred = estimator.predict(feature_test[0:1])
print(one_line_label_pred)

one_line_result = confusion_matrix(label_test[0:1], one_line_label_pred)
print(one_line_result)

model = fully_connected_model()
weights = model.get_weights()
print(weights)

manual_variable_initialization(True)

# model.save("model/normal_nn.h5")
model_json = model.to_json()
with open('model/nn_structure.json', 'w') as f:
    f.write(model_json)
model.save_weights("model/weights.h5")
print('Model saved to disk. ')

# For actual classification scenario, the model will be loaded and used directly 
# loaded_model = load_model('model/normal_nn.h5')
json_file = open("model/nn_structure.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model/weights.h5")
print(loaded_model.get_weights())
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
loaded_labels_pred = loaded_model.predict_classes(feature_test)
# print(label_test)
# print(loaded_labels_pred)
# for label in loaded_labels_pred:
#     print(label_names[label])

print(loaded_model.get_weights())



