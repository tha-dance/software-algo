'''
This program achieves multi-class classification using KNN
'''
from sklearn.datasets import load_iris # IRIS dataset is a pre-defined dataset in sklearn
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

import matplotlib.pyplot as plt

iris = load_iris()

# iris contains the columns of its features and also columns of its target
print(iris.data)
print(iris.target)

X = iris.data
y = iris.target

feature_train, feature_test, label_train, label_test = train_test_split(X, y, test_size=0.2, random_state=4)

print(feature_train.shape)
print(feature_test.shape)
print(label_train.shape)
print(label_test.shape)

k_limit = 27
scores_list = []

for k in range(1, k_limit):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(feature_train, label_train)
    label_pred = knn.predict(feature_test)
    score = metrics.accuracy_score(label_test, label_pred)
    scores_list.append(score)

k_range = range(1, 27)
plt.plot(k_range, scores_list)
plt.xlabel('Value of K for this KNN')
plt.ylabel('Accuracy for testing dataset')
plt.show()

# TODO: choose the best K from the dataset 
# and use that to predict test dataset and get the final accuracy


