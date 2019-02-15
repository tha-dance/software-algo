'''
This program achieves multi-class classification using KNN
'''
from sklearn.datasets import load_iris # IRIS dataset is a pre-defined dataset in sklearn
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

import matplotlib.pyplot as plt

iris = load_iris()
print(iris)

# iris contains the columns of its features and also columns of its target
data_len = iris.data.shape[0]
X = iris.data
y = iris.target

# print(y.shape)

feature_train, feature_test, label_train, label_test = train_test_split(X, y, test_size=0.2, random_state=4)

# print(feature_train.shape)
# print(feature_test.shape)
# print(label_train.shape)
# print(label_test.shape)

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
# plt.show(block=False)

# TODO: choose the best K from the dataset 
# and use that to predict test dataset and get the final accuracy
max = 0
for i in range(len(scores_list)):
    if scores_list[max] > scores_list[i]:
        max = i

print('optimal is : ' + str(max))

knn = KNeighborsClassifier(n_neighbors=i)
knn.fit(feature_train, label_train)

# X_unseen = iris.data[int(0.8*data_len)+1:, :]
# y_unseen = iris.target[int(0.8*data_len)+1:]

y_predict = knn.predict(feature_test)

score = metrics.accuracy_score(label_test, y_predict)
print('score : %f%%' % (score*100))
count = 0
correct = 0
for i in range(len(y_predict)): 
    if y_predict[i] == label_test[i]:
        correct += 1
    count += 1
print(count)
print(correct)
print('Accuracy %.2f%%' % (correct * 100 / count))
