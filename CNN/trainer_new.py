'''
This will be the program to train the model during training scenario
'''

import pandas
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, SpatialDropout2D, MaxPooling2D
from keras.models import load_model
from keras.models import model_from_json
from keras.utils import np_utils, to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import cross_validation, preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.optimizers import SGD, Adam
from sklearn.model_selection import cross_val_score, StratifiedKFold

dataframe = pandas.read_csv('processed_train.csv', header=0)
dataset = dataframe.values

label_reference = [1,2,3,4,5,6,7,8,9,10,11]

len_data = len(dataset[0])
feature = dataset[:, 0:len_data-1].astype(float)
label = dataset[:, len_data-1].astype(int)

NUM_FEATURE = len_data-1
print('num of feature is ' + str(NUM_FEATURE))
NUM_LABEL = len(label_reference)+1

# The key is the scaler, previously the preprocessing.normalize() scale the values too low 
# and the model cannot detect the difference between different values. Hence cannot get effective prediction 
scaler = preprocessing.StandardScaler()
scaler.fit(feature)
feature = scaler.transform(feature)
# label = preprocessing.normalize(label)
feature = feature.reshape(len(feature), 9, 6, 1)

feature_train, feature_test, label_train, label_test = cross_validation.train_test_split(feature, label, test_size=0.2, random_state=4)
# feature_test = to_categorical(feature_test)
# feature_train = to_categorical(feature_train)

def fully_connected_model():
    # print(num_feature, num_label)
    # create model 
    model = Sequential()

    # build layers
    # model.add(Dense(32, input_dim=num_feature, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dropout(0.25))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dropout(0.25))
    # model.add(Dense(32, activation='relu'))

    # We can build a CNN to increase accuracy (hopefully) 
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(9,6,1)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(SpatialDropout2D(0.25))
    model.add(Flatten())
    model.add(Dropout(0.25))

    # The number of neurons in the last layer == number of classes 
    model.add(Dense(NUM_LABEL, activation='softmax')) # use softmax to represented predicted probabilty

    # compile model
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    return model
print(feature_train.shape)
# print(label_train.shape[0])

'''
model got more parameters ==> more representation power(high capacity) ==> easy to get overfit since representation power is high
different techniques to avoid overfitting issues 
1. L1 or L2 regularization 
2. decay learning rate
3. add dropout layer ==> find optimal dropout rate
'''

estimator = KerasClassifier(build_fn=fully_connected_model, epochs=20, batch_size=200, verbose=1)
estimator.fit(feature_train, label_train)

# seed = 11
# np.random.seed(seed)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed) 
# results = cross_val_score(estimator, feature, label, cv=kfold)

model = estimator.model 
# model = fully_connected_model(len_data-1, len(label_reference)+1)
# model.fit(feature_train, label_train, epochs=200, batch_size=100, verbose=1)
label_pred_index = model.predict_classes(feature_test)
print(label_pred_index)
print(max(label_pred_index))
label_names = [1,2,3,4,5,6,7,8,9,10,11]
label_pred = []
for index in label_pred_index:
    label_pred.append(label_names[index])

matrix = confusion_matrix(label_test, label_pred)
accuracy = accuracy_score(label_test, label_pred)
print(matrix)
print(accuracy)

model_json = model.to_json()
with open('model/cnn_structure.json', 'w') as f:
    f.write(model_json)
model.save_weights("model/cnn_weights.h5")
print('Model saved to disk. ')

