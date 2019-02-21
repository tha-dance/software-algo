import pandas
import numpy as np 

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split

dataframes = pandas.read_csv('iris.csv')
dataset = dataframes.values
len_data = len(dataset[0])
properties = dataset[:, 0:len_data-1].astype(float) # property
labels = dataset[:, len_data-1] # label

properties_train, properties_test, labels_train, labels_test = train_test_split(properties, labels, test_size=0.2, random_state=4)

rf = RandomForestClassifier(n_estimators=100) # the parameter n_estimators defines the number of trees in the forest 

rf.fit(properties_train, labels_train)
labels_pred = rf.predict(properties_test)
result = confusion_matrix(labels_test, labels_pred)
print(result)
