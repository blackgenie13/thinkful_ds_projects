# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 15:54:15 2016

@author: Michael Lin_2
"""

import pandas as pd
import numpy as np
import random
from sklearn import datasets

temp = datasets.load_iris()
iris = pd.DataFrame()
iris['sepal_length'] = temp.data[:,0]
iris['sepal_width'] = temp.data[:,1]
iris['petal_length'] = temp.data[:,2]
iris['petal_width'] = temp.data[:,3]
iris['target'] = temp.target
iris['target_flower'] = ''
# iris['target_flower'].replace(0, 'setosa', inplace = True)
# iris['target_flower'].replace(1, 'versicolor', inplace = True)
# iris['target_flower'].replace(2, 'virginica', inplace = True)

for i in range(len(iris['target'])):
    if iris['target'][i] == 0:
        iris['target_flower'][i] = 'setosa'
    elif iris['target'][i] == 1:
        iris['target_flower'][i] = 'versicolor'
    else:
        iris['target_flower'][i] = 'virginica'
        
s_length = iris['sepal_length'].tolist()
s_width = iris['sepal_width'].tolist()

x1 = np.average(iris['sepal_length'].tolist())
x2 = np.average(iris['sepal_width'].tolist())

k = 11
length = random.uniform(min(iris['sepal_length']), max(iris['sepal_length']))
width = random.uniform(min(iris['sepal_width']), max(iris['sepal_width']))

prediction = k_neighbor (data = iris, x1 = length, x2 = width, K=k)

print('the prediction for length {0:.2f} and width {1:.2f} using K = {2} is likely to be {3}'.format(length, width, k, prediction))

######################### K-Neighbor Function ########################

def k_neighbor (data, x1, x2, K = 10):
    data['distance'] = 0.00

    len_dis = 0
    wid_dis = 0

    for i in range(len(data['target'])):
        len_dis = (data['sepal_length'][i] - x1)**2
        wid_dis = (data['sepal_width'][i] - x2)**2
        data['distance'][i] = np.sqrt(len_dis + wid_dis)

    data_sorted = data.sort_values(by='distance', ascending=False)
    data_sorted = data_sorted.reset_index()

    setosa_count = 0
    versicolor_count = 0
    virginica_count = 0

    for i in range(K):
        if data_sorted['target_flower'][i] == 'setosa':
            setosa_count += 1
        elif data_sorted['target_flower'][i] == 'versicolor':
            versicolor_count += 1
        else:
            virginica_count += 1
        
    prediction = 'unknown'
    if (setosa_count > versicolor_count) and (setosa_count > virginica_count):
        prediction = 'setosa'
    elif (versicolor_count > setosa_count) and (versicolor_count > virginica_count):
        prediction = 'versicolor'
    elif (virginica_count > setosa_count) and (virginica_count > versicolor_count):
        prediction = 'virginica'
    else:
        prediction = 'unknown'
    
    return prediction
    

    