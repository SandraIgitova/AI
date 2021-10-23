import numpy as np 
import random
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn import neighbors
import parser
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

def avg(x):
    s = 0.0
    for t in x:
        s += t
    return (s/len(x))*100.0

def compare(list1, list2):
    count = 0
    for i in range(len(list1)):
        if list1[i] == list2[i]:
            count += 1
    return (count / len(list1)) * 100


def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

def normalize(data):
    minimum = []
    maximum = []
    for i in range(len(data[0])):
        tmp_1 = min(data[1: len(data) - 1][i])
        minimum.append(tmp_1)
        tmp_2 = max(data[1: len(data) - 1][i])
        maximum.append(tmp_2)
    for i in range(len(data) - 1):
        for j in range(len(data[0])):
            data[i][j] = abs((data[i][j] - minimum[j]) / (maximum[j] - minimum[j]))
    return data


def del_last_col(data):
    new_data = []
    for i in range (len(data) - 1):
        tmp = list(map(lambda x: float(x), data[i][0:3]))
        new_data.append(tmp)

    return new_data

def get_colomn(data, col):
    l = []
    for i in range(len(data) - 1):
        tmp = 1 if data[i][col] == 'True' else 0
        l.append(tmp)
    return l

data = parser.csv_reader('nasa.csv')
data = parser.pars_data(data)

target = get_colomn(data, 7)
target = np.array(target)
test = del_last_col(data)
test = normalize(test)
test = np.array(test)
target.reshape((896, 1))


train, new_test = splitDataset(data, 0.9)

train1 = del_last_col(train)
target1 = get_colomn(train, 7)
target1 = np.array(target1)
target1.reshape((len(target1), 1))


knn = neighbors.KNeighborsClassifier(n_neighbors = 5, weights='uniform')
score = cross_val_score(knn, test, target, cv=10)

print('KNN: ' + str(avg(score)))

gnb = GaussianNB()
model = gnb.fit(train1, target1)
preds = gnb.predict(train1)
print('Naive bayes: ' + str(compare(preds, target1)))