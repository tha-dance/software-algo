'''
This will be the program to train the model during training scenario
'''

import pandas

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.models import model_from_json
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix, accuracy_score

dataframe = pandas.read_csv('input/HARDataset/hard.csv', header=0)
dataset = dataframe.values

len_data = len(dataset[0])
feature = dataset[:, 0:len_data-1].astype(float)
label = dataset[:, len_data-1].astype(int)

feature_train, feature_test, label_train, label_test = cross_validation.train_test_split(feature, label, test_size=0.2, random_state=4)

def fully_connected_model():
    # create model 
    model = Sequential()

    # build layers
    model.add(Dense(8, input_dim=len_data-1, activation='relu'))
    model.add(Dense(32, input_dim=8, activation='relu'))

    # The number of neurons in the last layer == number of classes 
    model.add(Dense(6, activation='softmax')) # use softmax to represented predicted probabilty

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

estimator = KerasClassifier(build_fn=fully_connected_model, epochs=200, batch_size=10, verbose=0)
estimator.fit(feature_train, label_train)

model = estimator.model
label_pred_index = model.predict_classes(feature_test)
print(label_pred_index)
label_names = [1,2,3,4,5,6]
label_pred = []
for index in label_pred_index:
    label_pred.append(label_names[index])

matrix = confusion_matrix(label_test, label_pred)
accuracy = accuracy_score(label_test, label_pred)
print(matrix)
print(accuracy)

model_json = model.to_json()
with open('model/nn_structure.json', 'w') as f:
    f.write(model_json)
model.save_weights("model/weights.h5")
print('Model saved to disk. ')

