'''
This will be the program to predict the class of each dance move during actual scenario
'''
import pandas
from keras.models import model_from_json
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix, accuracy_score

json_file = open("model/nn_structure.json")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model/weights.h5")

print('Model loaded from disk. ')

dataframe = pandas.read_csv('input/HARDataset/hard.csv', header=0)
dataset = dataframe.values
len_data = len(dataset[0])
feature = dataset[:, 0:len_data-1].astype(float)
label = dataset[:, len_data-1]
label_names = [1,2,3,4,5,6]

feature_train, feature_test, label_train, label_test = cross_validation.train_test_split(feature, label, test_size=0.2, random_state=4)

label_pred_index = loaded_model.predict_classes(feature_test)
label_pred = []
for index in label_pred_index:
    label_pred.append(label_names[index])

matrix = confusion_matrix(label_pred, label_test)
accur = accuracy_score(label_pred, label_test)
print(matrix)
print(accur)
