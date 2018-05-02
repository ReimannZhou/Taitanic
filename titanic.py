import pandas as pd
import numpy as np
from data_view import *
from sklearn.cluster import *

#字符串替换
def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if big_string.find(substring) != -1:
            return substring
    print(big_string)
    return np.nan


#把 titles 用 mr, mrs, miss, master 替换
def replace_titles(x):
    title = x['Title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title == 'Dr':
        if x['Sex'] == 'Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title


def replace_Embarked(x):
    if x == 'S':
        return 0
    elif x == 'Q':
        return 1
    else:
        return 2


def replace_Title(x):
    if x == 'Mr':
        return 0
    elif x == 'Mrs':
        return 1
    elif x == 'Miss':
        return 2
    else:
        return 3


def replace_Deck(x):
    if x == 'A':
        return 0
    elif x == 'B':
        return 1
    elif x == 'C':
        return 2
    elif x == 'D':
        return 3
    elif x == 'E':
        return 4
    elif x == 'F':
        return 5
    elif x == 'T':
        return 6
    elif x == 'G':
        return 7
    else:
        return 8




#读取并处理数据
def loadData(csvPath):
    dataFrame = pd.read_csv(csvPath, na_values = None, low_memory = False)
    dataFrame['Age'].fillna(value = int(np.mean(dataFrame['Age'])), inplace = True)
    dataFrame['Fare'].fillna(value = 0, inplace = True)
    dataFrame['Embarked'].fillna(value = dataFrame['Embarked'].value_counts(dropna = True).idxmax(), inplace = True)
    dataFrame['Cabin'].fillna(value = 'Unknown', inplace = True)
    #通过对 Name 属性的分析新增 Title 属性，属性值为 mr, mrs, miss, master
    title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                  'Dr', 'Ms', 'Mlle', 'Col', 'Capt', 'Mme', 'Countess',
                  'Don', 'Jonkheer']
    dataFrame['Title'] = dataFrame['Name'].map(lambda x: substrings_in_string(x, title_list))
    dataFrame['Title'] = dataFrame.apply(replace_titles, axis = 1)
    #通过对 Cabin 属性的分析新增 Deck 属性
    cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
    dataFrame['Deck'] = dataFrame['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
    #增加 FamilySize 属性
    dataFrame['Family_Size'] = dataFrame['SibSp'] + dataFrame['Parch']
    #增加 FarePerPerson 属性
    dataFrame['Fare_Per_Person'] = dataFrame['Fare'] / (dataFrame['Family_Size'] + 1)
    # 男性为0，女性为1
    dataFrame['Sex'] = dataFrame['Sex'].apply(lambda x: 0 if x == 'male' else 1)
    dataFrame['Embarked'] = dataFrame['Embarked'].map(replace_Embarked)
    dataFrame['Title'] = dataFrame['Title'].map(replace_Title)
    dataFrame['Deck'] = dataFrame['Deck'].map(replace_Deck)
    dropCols = ['Name', 'Ticket', 'Cabin']
    return dataFrame.drop(dropCols, axis = 1)


# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# SVM Classifier
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel = 'rbf', probability = True)
    model.fit(train_x, train_y)
    return model


# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha = 0.01)
    model.fit(train_x, train_y)
    return model


if __name__ == '__main__':
    trainPath = './input/train.csv'
    testPath = './input/test.csv'
    models = {}

    trainData = loadData(trainPath)
    testData = loadData(testPath)
    categoryCols = ['Survived', 'Pclass', 'Sex', 'Embarked', 'Title', 'Deck']
    valueCols = ['Age', 'SibSp', 'Parch', 'Fare', 'Family_Size', 'Fare_Per_Person']
    train_x = trainData[list(set(categoryCols + valueCols).difference(set(['Survived', ])))]
    train_y = trainData['Survived']
    test_x = testData[list(set(categoryCols + valueCols).difference(set(['Survived', ])))]

    #训练数据可视化
    #统计标称属性频数
    count(trainData, categoryCols)
    describe(trainData, valueCols)
    #绘制直方图
    histogram(trainData, categoryCols)
    #绘制qq图
    qqplot(trainData, valueCols)
    #绘制盒图
    boxplot(trainData, valueCols)

    #使用分类模型对测试集分类
    test_classifiers = ['NB', 'KNN', 'SVM', 'DT']
    classifiers = {'NB': naive_bayes_classifier,
                   'KNN': knn_classifier,
                   'DT': decision_tree_classifier,
                   'SVM': svm_classifier
                   }
    for classifier in test_classifiers:
        model = classifiers[classifier](train_x, train_y)
        models[classifier] = model
        count(test_x, list(set(categoryCols).difference(set(['Survived', ]))))
        describe(test_x, valueCols)
        predict = model.predict(test_x)
        save_result1(predict, classifier)

    #使用聚类模型对测试集聚类
    test_clusters = ['HC', 'KM']
    clusters = {
        'HC': AgglomerativeClustering,
        'KM': KMeans
    }
    for cluster in test_clusters:
        if cluster == 'HC':
            model = clusters[cluster](n_clusters = 2, linkage = 'complete', affinity = 'cosine')
        else:
            model = clusters[cluster](n_clusters = 2)
        model.fit(test_x)
        labels = model.labels_
        #对于测试数据，考虑泰坦尼克号的实际情况，死亡的人数远大于或者的人数，设定数目多的标签为死亡(0)，少的为生存(1)
        labels = change(labels)
        save_result2(labels, cluster)

    #对比分类模型和聚类模型在测试集上的结果，以少数服从多数的原则假设结果，算出准确率
    compare_results()