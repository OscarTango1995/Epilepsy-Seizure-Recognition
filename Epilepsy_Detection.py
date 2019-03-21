import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

accuracy_list=[]
Kappa_list=[]
precision_list=[]




'''|--------------------------------Data Reading Phase-------------------------------------|'''

#reading the csv file

df=pd.read_csv('data.csv' ,low_memory=False)
print("The data has {} observations and {} features".format(df.shape[0], df.shape[1]))

#Checking for null values in the dataframe
null_count = df.apply(lambda df: sum(df.isnull()))
print('Number of columns with null values:', len(null_count[null_count != 0]))
#print(df.head(5))
'''|--------------------------------Data Reading Phase-------------------------------------|'''

'''|--------------------------------Data Cleaning Phase-------------------------------------|'''
#Dropping Un-Necessary Attributes
df.drop('Unnamed: 0', inplace=True, axis=1)

#Converting class label from 5 to binary, 0 for Normal Brain, 1 for Epilepsy

df['y'] = df['y'].apply(lambda x: 1 if x == 1 else 0)


'''|--------------------------------Data Cleaning Phase-------------------------------------|'''

'''|--------------------------------Data Preparation Phase-------------------------------------|'''

class_label=df.y
features=df.drop('y',axis=1)

#Spliting into 30% test 70% train

X_train, X_test, y_train, y_test = train_test_split(features, class_label,test_size=0.3,shuffle=True)

'''|--------------------------------Data Preparation Phase-------------------------------------|'''

'''|--------------------------------Classifiers Application-------------------------------------|'''


'''|********************************K Nearest Neighbour*****************************************|'''
def applyKNN(neighbours):
    probs = []
    print("\n***K-Nearest Neighbour***")
    model = KNeighborsClassifier(n_neighbors=neighbours)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    probs.append(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy of KNN with " + str(neighbours) + " neighbours = %.2f%%" % (accuracy * 100.0))
    accuracy_list.append((accuracy * 100.0))
    Matrix_Confusion(y_test, y_pred, "K Nearest Neighbours with n="+str(neighbours))
    rocCurve(probs, "KNN")

'''|********************************K Nearest Neighbour*****************************************|'''

'''|********************************Naive Bayes*************************************************|'''
def applyBernouliNaiveBayes():
    probs=[]
    print("\n***Bernouli Naive Bayes***")
    model=BernoulliNB(binarize=True)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    probs.append(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print("\nAccuracy of BernNB:%.2f%%\n" % (round(accuracy * 100, 2)))
    accuracy_list.append((accuracy * 100.0))
    Matrix_Confusion(y_test,y_pred,"Bernouli Naive Bayes")
    rocCurve(probs,"Bernouli Naive Bayes")

def applyGausianNaiveBayes():
    probs = []
    print("***Gaussian Naive Bayes***")
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    probs.append(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy of GaussNB:%.2f%%" % (round(accuracy * 100, 2)))
    accuracy_list.append((accuracy * 100.0))
    Matrix_Confusion(y_test, y_pred, "Gaussian Naive Bayes")
    rocCurve(probs, "Gaussian Naive Bayes")

'''|********************************Support Vector Machine*****************************************|'''
def applySVM():
    probs = []
    print("\n***Support Vector Machine***")
    model =svm.SVC(kernel='rbf', probability=True)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    probs.append(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy of SVM:%.2f%%"%(round((accuracy*100),2)))
    accuracy_list.append((accuracy * 100.0))
    Matrix_Confusion(y_test, y_pred, "Support Vector Machine")
    rocCurve(probs, "SVM")


'''|********************************Random Forest**************************************************|'''
def applyRandomForest():
    probs = []
    print("\n***Random Forest***")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    probs.append(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy of RandomForest:%.2f%%" % (round((accuracy * 100), 2)))
    accuracy_list.append((accuracy * 100.0))
    Matrix_Confusion(y_test, y_pred, "Random Forest")
    rocCurve(probs, "Random Forest")

'''|********************************Logistic Regression********************************************|'''
def applyLogisticRegression():
    probs = []
    print("\n***Logistic Regression***")
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    probs.append(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy of LGR:%.2f%%" % (round((accuracy * 100), 2)))
    accuracy_list.append((accuracy * 100.0))
    Matrix_Confusion(y_test, y_pred, "Logistic Regression")
    rocCurve(probs, "Logistic Regression")


'''|--------------------------------Classifiers Application-------------------------------------|'''
def check_best_neighbour():
    k_range=range(1, 26)
    scores=[]
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        scores.append((accuracy_score(y_test,y_pred))*100)
    plt.plot(k_range,scores)
    plt.xlabel("Value of K")
    plt.ylabel("Accuracy")
    plt.show()

'''|--------------------------------Data Visualization------------------------------------------|'''
'''Confusion Matrix'''
def Matrix_Confusion(test,pred,name):
    cm = confusion_matrix(test, pred)
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    ALL=TP+FP+FN+TN
    OA=(TP+TN)/ALL
    AC=((((TP+FP)/ALL)*((TP+FN)/ALL))+(((FN+TN)/ALL)*((FP+TN)/ALL)))
    Kappa=((OA-AC)/(1-AC))*100
    classification_error = (FP + FN) / float(TP + TN + FP + FN)
    sensitivity = TP / float(FN + TP)
    specificity = TN / (TN + FP)
    Kappa_list.append(Kappa)
    roc_auc = roc_auc_score(y_test, pred)
    print("TP:%s   FP:%s\nFN:%s    TN:%s"%(TP,FP,FN,TN))
    target_names = ['class 0', 'class 1']
    print("******Classification Report*******\n",classification_report(y_test, pred,target_names=target_names))
    print("Classification_Error                 Sensitivity                  Specifity             ")
    print("%.2f                                 %.2f                         %.2f                  " %(classification_error,sensitivity,specificity))
    print("Kappa Statistic: %.2f%%" % Kappa)
    precision_list.append(round((TP/(TP+FP))*100,2))
    print("Area Under the Receiver Operating Characteristic Curve: %.2f" % roc_auc)
    graph_plot(cm,name)

'''|--------------------------------Graph-------------------------------------------------------|'''
def graph_plot(matrix,name):
    plt.matshow(matrix)
    plt.title('%s\nConfusion matrix'%(name))
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def rocCurve(probabilities,name):
    preds = probabilities[-1]
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic (%s)'%(name))
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def accuracyPlot():
    objects = ('Knn', 'BernNB', 'GaussNB', 'SVM', 'RandomForest','LGR')
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, accuracy_list, align='center', alpha=0.7)
    plt.xticks(y_pos, objects)
    plt.ylabel('Accuracy(%)')
    plt.title('Classifiers')
    plt.show()
    plt.bar(y_pos, Kappa_list, align='center', alpha=0.7)
    plt.xticks(y_pos, objects)
    plt.ylabel('Kappa Statistic')
    plt.title('Classifiers')
    plt.show()
    plt.bar(y_pos, precision_list , align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Precision(%)')
    plt.title('Classifiers')
    plt.show()



applyKNN(5)
applyBernouliNaiveBayes()
applyGausianNaiveBayes()
applySVM()
applyRandomForest()
applyLogisticRegression()
accuracyPlot()




