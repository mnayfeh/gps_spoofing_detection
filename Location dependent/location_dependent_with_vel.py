import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import RocCurveDisplay
import time

input_file = "Data.csv"
input_file_test = "Testing_Data.csv"

method = 'NB'


# Clean= 0
# Static= 1
# Dynamic= 2

x= pd.read_csv(input_file, usecols=[2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 26])
y= pd.read_csv(input_file, usecols=[0])
y= np.ravel(y)

x_test= pd.read_csv(input_file_test, usecols=[2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 26])
y_test= pd.read_csv(input_file_test, usecols=[0])
y_test= np.ravel(y_test)

scaler = StandardScaler()
#scaler = MinMaxScaler(feature_range=(-1, 1), copy=True, clip=False)
x = scaler.fit_transform(x)

x_test = scaler.fit_transform(x_test)

train_accuracy = 0
train_precision = 0
train_recall = 0
train_fscore = 0
train_con_matrix = 0

train_time = 0
predict_time = 0

val_accuracy = 0
val_precision = 0
val_recall = 0
val_fscore = 0
val_con_matrix = 0

test_accuracy = 0
test_precision = 0
test_recall = 0
test_fscore = 0
test_con_matrix = 0

for i in range(10):

    X_train, X_val, y_train, y_val = train_test_split(x, y, train_size=0.7, random_state=i)


    if method == 'RF':
        clf = RandomForestClassifier(criterion='entropy', n_estimators=757, max_depth=394, min_samples_split=489,
                                     min_samples_leaf=39, ccp_alpha=0.00114063)
        
    elif method == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=538, leaf_size=728, weights='distance', algorithm='ball_tree',
                                   metric='manhattan', p=6)

    elif method == 'ANN':
        clf = MLPClassifier(hidden_layer_sizes=(221, 170), activation='logistic', solver='lbfgs', max_iter=954,
                            alpha=0.001761192, early_stopping=True)
        
    elif method == 'LR':
        clf = LogisticRegression(penalty='l2', solver='sag', C=7.210172, max_iter=459)

    elif method == 'DT':
        clf = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=394, min_samples_split=489,
                                     min_samples_leaf=39, ccp_alpha=0.001140636)

    elif method == 'SVM':
        clf = LinearSVC(penalty='l1', loss='squared_hinge', dual=False, max_iter=777, C=3.50658)

    elif method == 'NB':
        clf = GaussianNB(var_smoothing=0.433139)

        
    t1 = time.time()
    clf.fit(X_train, y_train)
    train_time_i = time.time()-t1
    train_time = train_time + train_time_i
    
    y_pred_train = clf.predict(X_train)

    train_accuracy_i = metrics.accuracy_score(y_train, y_pred_train)
    train_precision_i = metrics.precision_score(y_train, y_pred_train, average='weighted')
    train_recall_i = metrics.recall_score(y_train, y_pred_train, average='weighted')
    train_fscore_i = metrics.f1_score(y_train, y_pred_train, average='weighted')
    train_con_matrix_i = confusion_matrix(y_train, y_pred_train)

    train_accuracy = train_accuracy + train_accuracy_i
    train_precision = train_precision + train_precision_i
    train_recall = train_recall + train_recall_i
    train_fscore = train_fscore + train_fscore_i
    train_con_matrix = train_con_matrix + train_con_matrix_i

    
    y_pred_val = clf.predict(X_val)

    val_accuracy_i = metrics.accuracy_score(y_val, y_pred_val)
    val_precision_i = metrics.precision_score(y_val, y_pred_val, average='weighted')
    val_recall_i = metrics.recall_score(y_val, y_pred_val, average='weighted')
    val_fscore_i = metrics.f1_score(y_val, y_pred_val, average='weighted')
    val_con_matrix_i = confusion_matrix(y_val, y_pred_val)

    val_accuracy = val_accuracy + val_accuracy_i
    val_precision = val_precision + val_precision_i
    val_recall = val_recall + val_recall_i
    val_fscore = val_fscore + val_fscore_i
    val_con_matrix = val_con_matrix + val_con_matrix_i

    t2 = time.time()
    y_pred_test = clf.predict(x_test)
    predict_time_i = time.time()-t2
    predict_time = predict_time + predict_time_i

    test_accuracy_i = metrics.accuracy_score(y_test, y_pred_test)
    test_precision_i = metrics.precision_score(y_test, y_pred_test, average='weighted')
    test_recall_i = metrics.recall_score(y_test, y_pred_test, average='weighted')
    test_fscore_i = metrics.f1_score(y_test, y_pred_test, average='weighted')
    test_con_matrix_i = confusion_matrix(y_test, y_pred_test)

    test_accuracy = test_accuracy + test_accuracy_i
    test_precision = test_precision + test_precision_i
    test_recall = test_recall + test_recall_i
    test_fscore = test_fscore + test_fscore_i
    test_con_matrix = test_con_matrix + test_con_matrix_i

train_time = train_time / 10

predict_time = predict_time / 10

print("\nTraining time is (ms)\n")
print(train_time*1000)
print("\nPredection time is (ms)\n")
print(predict_time*1000)

train_con_matrix = np.ravel(train_con_matrix / 10)

TP_train_1 = train_con_matrix[0]
FP_train_1 = train_con_matrix[3] + train_con_matrix[6]
FN_train_1 = train_con_matrix[1] + train_con_matrix[2]
TN_train_1 = train_con_matrix[4] + train_con_matrix[8]

FAR_train_1 = FP_train_1/(FP_train_1+TN_train_1)
MDR_train_1 = FN_train_1/(TP_train_1+FN_train_1)

TP_train_2 = train_con_matrix[4]
FP_train_2 = train_con_matrix[1] + train_con_matrix[7]
FN_train_2 = train_con_matrix[3] + train_con_matrix[5]
TN_train_2 = train_con_matrix[0] + train_con_matrix[8]

FAR_train_2 = FP_train_2/(FP_train_2+TN_train_2)
MDR_train_2 = FN_train_2/(TP_train_2+FN_train_2)

TP_train_3 = train_con_matrix[8]
FP_train_3 = train_con_matrix[2] + train_con_matrix[5]
FN_train_3 = train_con_matrix[6] + train_con_matrix[7]
TN_train_3 = train_con_matrix[0] + train_con_matrix[4]

FAR_train_3 = FP_train_3/(FP_train_3+TN_train_3)
MDR_train_3 = FN_train_3/(TP_train_3+FN_train_3)

FAR_train = (FAR_train_1 + FAR_train_2 + FAR_train_3) / 3
MDR_train = (MDR_train_1 + MDR_train_2 + MDR_train_3) / 3

train_score = np.array([train_accuracy, train_precision, train_recall, train_fscore, FAR_train*10, MDR_train*10])
train_score = train_score * 10

val_con_matrix = np.ravel(val_con_matrix / 10)

TP_val_1 = val_con_matrix[0]
FP_val_1 = val_con_matrix[3] + val_con_matrix[6]
FN_val_1 = val_con_matrix[1] + val_con_matrix[2]
TN_val_1 = val_con_matrix[4] + val_con_matrix[8]

FAR_val_1 = FP_val_1/(FP_val_1+TN_val_1)
MDR_val_1 = FN_val_1/(TP_val_1+FN_val_1)

TP_val_2 = val_con_matrix[4]
FP_val_2 = val_con_matrix[1] + val_con_matrix[7]
FN_val_2 = val_con_matrix[3] + val_con_matrix[5]
TN_val_2 = val_con_matrix[0] + val_con_matrix[8]

FAR_val_2 = FP_val_2/(FP_val_2+TN_val_2)
MDR_val_2 = FN_val_2/(TP_val_2+FN_val_2)

TP_val_3 = val_con_matrix[8]
FP_val_3 = val_con_matrix[2] + val_con_matrix[5]
FN_val_3 = val_con_matrix[6] + val_con_matrix[7]
TN_val_3 = val_con_matrix[0] + val_con_matrix[4]

FAR_val_3 = FP_val_3/(FP_val_3+TN_val_3)
MDR_val_3 = FN_val_3/(TP_val_3+FN_val_3)

FAR_val = (FAR_val_1 + FAR_val_2 + FAR_val_3) / 3
MDR_val = (MDR_val_1 + MDR_val_2 + MDR_val_3) / 3

val_score = np.array([val_accuracy, val_precision, val_recall, val_fscore, FAR_val*10, MDR_val*10])
val_score = val_score * 10


test_con_matrix = np.ravel(test_con_matrix / 10)

TP_test_1 = test_con_matrix[0]
FP_test_1 = test_con_matrix[3] + test_con_matrix[6]
FN_test_1 = test_con_matrix[1] + test_con_matrix[2]
TN_test_1 = test_con_matrix[4] + test_con_matrix[8]

FAR_test_1 = FP_test_1/(FP_test_1+TN_test_1)
MDR_test_1 = FN_test_1/(TP_test_1+FN_test_1)

TP_test_2 = test_con_matrix[4]
FP_test_2 = test_con_matrix[1] + test_con_matrix[7]
FN_test_2 = test_con_matrix[3] + test_con_matrix[5]
TN_test_2 = test_con_matrix[0] + test_con_matrix[8]

FAR_test_2 = FP_test_2/(FP_test_2+TN_test_2)
MDR_test_2 = FN_test_2/(TP_test_2+FN_test_2)

TP_test_3 = test_con_matrix[8]
FP_test_3 = test_con_matrix[2] + test_con_matrix[5]
FN_test_3 = test_con_matrix[6] + test_con_matrix[7]
TN_test_3 = test_con_matrix[0] + test_con_matrix[4]

FAR_test_3 = FP_test_3/(FP_test_3+TN_test_3)
MDR_test_3 = FN_test_3/(TP_test_3+FN_test_3)

FAR_test = (FAR_test_1 + FAR_test_2 + FAR_test_3) / 3
MDR_test = (MDR_test_1 + MDR_test_2 + MDR_test_3) / 3

test_score = np.array([test_accuracy, test_precision, test_recall, test_fscore, FAR_test*10, MDR_test*10])
test_score = test_score * 10


columns= ['Accuracy', 'Precision', 'Recall', 'F-Score', 'FAR', 'MDR']
index= ['Training', 'Validation', 'Testing']

pd.DataFrame(np.array([train_score, val_score, test_score]), index=index, columns=columns).to_csv(method + " score.csv", index = True, header = True)

target_names = ['Clean', 'Static', 'Dynamic']

ConfusionMatrixDisplay.from_estimator(clf, X_val, y_val,display_labels=target_names,normalize=None, cmap="GnBu")
plt.title(method + ' Validation confusion matrix')
plt.savefig(method + ' Validation confusion matrix.png', dpi=600)

ConfusionMatrixDisplay.from_estimator(clf, x_test, y_test,display_labels=target_names,normalize=None, cmap="GnBu")
plt.title(method + ' Testing confusion matrix')
plt.savefig(method + ' Testing confusion matrix.png', dpi=600)

plt.show()
