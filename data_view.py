import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pandas import DataFrame
import numpy as np
import os
import csv


def set_display():
    '''显示设置'''
    pd.set_option('display.max_columns', 2000)
    pd.set_option('display.max_rows', 2000)
    pd.set_option('display.width', 2000)


def count(dataFrame, columns, dropna = False, format_width=30):
    '''标称属性，给出每个可能取值的频数'''
    file = open('./output/out.txt', 'w+')
    format_text = '{{:<{0}}}{{:<{0}}}'.format(format_width)
    for col in columns:
        print('标称属性 <{}> 频数统计'.format(col), file = file)
        print(format_text.format('value', 'count'), file = file)
        print('--' * format_width, file = file)

        counts = pd.value_counts(dataFrame[col].values, dropna = dropna)
        for i, index in enumerate(counts.index):
            # 计算NaN的数目
            if pd.isnull(index):
                print(format_text.format('-NaN-', counts.values[i]), file = file)
            else:
                print(format_text.format(index, counts[index]), file = file)
        print('--' * format_width, file = file)
        print('\n', file = file)
    file.close()


def describe(dataFrame, columns):
    '''数值属性，给出最大、最小、均值、中位数、四分位数及缺失值的个数'''
    file = open('./output/out.txt', 'a+')
    desc = dataFrame[columns].describe()
    statistic = DataFrame()
    statistic['max'] = desc.loc['max']
    statistic['min'] = desc.loc['min']
    statistic['mean'] = desc.loc['mean']
    statistic['50%'] = desc.loc['50%']
    statistic['25%'] = desc.loc['25%']
    statistic['75%'] = desc.loc['75%']
    statistic['NaN'] = dataFrame[columns].isnull().sum()
    print(statistic, file = file)
    file.close()


# 绘图配置
row_size = 2
col_size = 3
cell_size = row_size * col_size


def histogram(dataFrame, columns):
    '''直方图'''
    counts = 0
    for i, col in enumerate(columns):
        if i % cell_size == 0:
            fig = plt.figure()
        ax = fig.add_subplot(col_size, row_size, (i % cell_size) + 1)
        dataFrame[col].hist(ax = ax, grid = False, figsize = (15, 15), bins = 50)
        plt.title(col)
        if (i + 1) % cell_size == 0 or i + 1 == len(columns):
            counts += 1
            plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
            plt.savefig('./output/histogram' + str(counts) + '.png')
            plt.show()


def qqplot(dataFrame, columns):
    '''qq图'''
    counts = 0
    for i, col in enumerate(columns):
        if i % cell_size == 0:
            fig = plt.figure(figsize = (15, 15))
        ax = fig.add_subplot(col_size, row_size, (i % cell_size) + 1)
        sm.qqplot(dataFrame[col], ax = ax)
        ax.set_title(col)
        if (i + 1) % cell_size == 0 or i + 1 == len(columns):
            counts += 1
            plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
            plt.savefig('./output/qqplot' + str(counts) + '.png')
            plt.show()


def boxplot(dataFrame, columns):
    '''盒图'''
    counts = 0
    for i, col in enumerate(columns):
        if i % cell_size == 0:
            fig = plt.figure()
        ax = fig.add_subplot(col_size, row_size, (i % cell_size) + 1)
        dataFrame[col].plot.box(ax = ax, figsize = (15, 15))
        if (i + 1) % cell_size == 0 or i + 1 == len(columns):
            counts += 1
            plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
            plt.savefig('./output/boxplot' + str(counts) + '.png')
            plt.show()


def save_result1(Survived, name):
    '''保存分类模型预测结果'''
    PassengerId = pd.read_csv('./input/test.csv', usecols = ['PassengerId', ])
    with open('./output/classifier_' + name + '.csv', 'w+') as file:
        w = csv.writer(file)
        w.writerow(['PassengerId', 'Survived'])
        for i in range(len(Survived)):
            w.writerow([PassengerId.iat[i, 0], Survived[i]])


def save_result2(Survived, name):
    '''保存聚类模型预测结果'''
    PassengerId = pd.read_csv('./input/test.csv', usecols = ['PassengerId', ])
    with open('./output/cluster_' + name + '.csv', 'w+') as file:
        w = csv.writer(file)
        w.writerow(['PassengerId', 'Survived'])
        for i in range(len(Survived)):
            w.writerow([PassengerId.iat[i, 0], Survived[i]])


def change(labels):
    '''修改聚类标签'''
    count0 = 0
    count1 = 0
    for i in range(len(labels)):
        if labels[i] == 0:
            count0 += 1
        else:
            count1 += 1
    print('count0 = ', count0)
    print('count1 = ', count1)
    if count0 >= count1:
        return labels
    else:
        for i in range(len(labels)):
            if labels[i] == 0:
                labels[i] = 1
            else:
                labels[i] = 0
        return labels


def compare_results():
    '''将所有结果合并到一个.csv文件，并且计算准确率'''
    path = './output'
    all_counts = []
    all_paths = []
    all_names = []
    list = os.listdir(path)
    for i in range(len(list)):
        if list[i].find('.csv') != -1 and list[i].find('summary') == -1:
            all_paths.append(os.path.join(path, list[i]))
            all_names.append(list[i][:-4])
            all_counts.append(0)
    print('all_paths = ', all_paths)
    print('all_names = ', all_names)
    data  = DataFrame()
    for i in range(len(all_names)):
        if i == 0:
            data = pd.read_csv(all_paths[i])
            data.rename(columns = {'Survived' : all_names[i]}, inplace = True)
        else:
            data = pd.merge(data, pd.read_csv(all_paths[i]), how = 'left', on = 'PassengerId', sort = False)
            data.rename(columns = {'Survived': all_names[i]}, inplace=True)

    def create_supposed(x):
        count0 = 0
        count1 = 0
        for i in range(len(all_names)):
            if x[all_names[i]] == 0:
                count0 += 1
            else:
                count1 += 1
        #只要有一半结果判断活下来了那么就假定为活下来了
        if count0 > count1:
            return 0
        else:
            return 1

    data['Supposed'] = 0
    data['Supposed'] = data.apply(create_supposed, axis = 1)
    err_count = [0 for x in range(len(all_names))]
    length = data.shape[0]
    for i in range(length):
        for j in range(len(all_names)):
            if(data[all_names[j]].ix[i] != data['Supposed'].ix[i]):
                err_count[j] += 1
    err_percent = [1 - x / length for x in err_count]
    with open('./output/summary.csv', 'w+') as file:
        w = csv.writer(file)
        w.writerow(data.columns)
        w.writerow([None, ] + err_percent + [None, ])
        for i in range(length):
            w.writerow(data.ix[i])