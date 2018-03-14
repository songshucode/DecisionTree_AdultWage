#  time:        2018-3-14
#  author:      songshu
#  function:    predict the wage of adult according to the data from census
#  version:     Python: 3.6.1
#               Anaconda: 4.4.0
#               numpy: 1.12.1
#               scikit-learn: 0.18.1
#               scipy: 0.19.0
import numpy as np
import csv
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from scipy.sparse import dok_matrix
from sklearn.metrics import accuracy_score

# data pre-processing
# read the data file
raw_train_file = open(r'adultTrain.csv', newline='')
raw_test_file = open(r'adultTest.csv', newline='')
raw_train_data = csv.reader(raw_train_file, delimiter=',', quotechar='|')
raw_test_data = csv.reader(raw_test_file, delimiter=',', quotechar='|')
# output the special file need
out_train_file = open(r'out_train.csv', 'w', newline='')
out_test_file = open(r'out_test.csv', 'w', newline='')
out_train_data = csv.writer(out_train_file)
out_test_data = csv.writer(out_test_file)
# remove the error data and fill another data 'NA'
# transform the continue data to disperse data
#   the train data
for data in raw_train_data:
    if ' ?' in data:
        for j in range(len(data)):
            if data[j] == ' ?':
                data[j] = 'NA'
    # deal with the age
    if data[0] != 'NA':
        age = int(int(data[0])/20)
        data[0] = str(age)
    # deal with the fnlwgt
    if data[2] != 'NA':
        fnlwgt = int(int(data[2])/100000)
        data[2] = fnlwgt
    # deal with the education-num
    if data[4] != 'NA':
        education_num = int(int(data[4])/5)
        data[4] = education_num
    # deal with the capital-gain:
    if data[10] != 'NA':
        capital_gain = int(int(data[10])/5000)
        data[10] = capital_gain
    # deal with the capital-loss
    if data[11] != 'NA':
        capital_loss = int(int(data[11])/500)
        data[11] = capital_loss
    # deal with the hours-per-week
    if data[12] != 'NA':
        hours_per_week = int(int(data[12])/50)
        data[12] = hours_per_week
    out_train_data.writerow(data)
#  the test data
for data in raw_test_data:
    if ' ?' in data:
        for j in range(len(data)):
            if data[j] == ' ?':
                data[j] = 'NA'
                # deal with the age
    if data[0] != 'NA':
        age = int(int(data[0]) / 20)
        data[0] = str(age)
    # deal with the fnlwgt
    if data[2] != 'NA':
        fnlwgt = int(int(data[2]) / 100000)
        data[2] = fnlwgt
    # deal with the education-num
    if data[4] != 'NA':
        education_num = int(int(data[4]) / 5)
        data[4] = education_num
    # deal with the capital-gain:
    if data[10] != 'NA':
        capital_gain = int(int(data[10]) / 5000)
        data[10] = capital_gain
    # deal with the capital-loss
    if data[11] != 'NA':
        capital_loss = int(int(data[11]) / 500)
        data[11] = capital_loss
    # deal with the hours-per-week
    if data[12] != 'NA':
        hours_per_week = int(int(data[12]) / 50)
        data[12] = hours_per_week
    out_test_data.writerow(data)
# close the file used
raw_train_file.close()
raw_test_file.close()
out_train_file.close()
out_test_file.close()

# load the data
train_file = open(r'out_train.csv', newline='')
test_file = open(r'out_test.csv', newline='')
train_data = csv.reader(train_file, delimiter=',', quotechar='|')
test_data = csv.reader(test_file, delimiter=',', quotechar='|')

# set the header of the data
headers = ['age','workclass',
           'fnlwgt','education',
           'education-num','marital-status',
           'occupation','relationship',
           'race','sex',
           'capital-gain','capital-loss',
           'hours-per-week','native-country']
# load the label list and feature list from the train data file
labelList = []
featureList = []
for data in train_data:
    labelList.append(data[-1])
    rowDict = {}
    for index in range(0,len(data) -1):
        rowDict[headers[index]] = data[index]
    featureList.append(rowDict)

# transform the feature list to the suitable form
vec = DictVectorizer()
X = vec.fit_transform(featureList).toarray()

# transform the label list to the suitable form
lb = preprocessing.LabelBinarizer()
Y = lb.fit_transform(labelList)

# using the decisong tree for classification
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)
clf = clf.fit(X,Y)

# visualize model as pdf file
with open('decision.dot', 'w') as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)

# transform the test data to the suitable form
L = len(vec.get_feature_names())
L_h = len(headers)
test_label_list = []
predict_label_list = []
for data in test_data:
    test_label_list.append(data[-1])
    temp_test_data = dok_matrix((1, L), dtype=np.int)
    for x in range(L_h):
        temp = headers[x] + '=' + data[x]
        temp = vec.vocabulary_[temp]
        temp_test_data[0, temp] = 1
    temp_test_data = temp_test_data.toarray()
    predict_label_list.append(clf.predict(temp_test_data))
test_label_list = lb.fit_transform(test_label_list)
test_label_list = test_label_list.tolist()

# caculate the accuracy of the prediction
print(accuracy_score(test_label_list, predict_label_list))