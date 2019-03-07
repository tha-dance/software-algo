'''
Single label multi-class classification 

This model run through the 
'''
import pandas

from keras.models import Sequential
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import load_model
from keras.utils import np_utils, to_categorical
from keras.optimizers import Adam
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn import cross_validation

# same process for all the programs to read the input 
dataframe = pandas.read_csv('input/iris.csv', header=0)
dataset = dataframe.values
print(dataset)

def convolutional_model(width, height, depth, finalAct='softmax'): 
    # final act use softmax as the activation in the end of the neural network structure
    model = Sequential()
    inputShape = (height, width, depth)
    dimension = -1

    # Conv => Relu => Max Pooling layer
    model.add(Conv2D(64, (3,3), padding='same', input_shape=inputShape)) 
    # the number of neurons depends on the size of the input dataset 
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3,3), activation='relu'))
    
    model.add(BatchNormalization(axis=dimension))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.2)) # drop out 1/5 of the connections to avoid overfitting 

    # Add two more set of layers from above architecture

    return model

height = 28
width = 28
depth = 1
model = convolutional_model(width, height, depth)

# Besides csv file, pandas.read_csv can also read txt file 
dataframe = pandas.read_csv('input/iris.csv', header=0)

