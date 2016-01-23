# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 18:30:37 2016

@author: Michael Lin_2
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('C:/Users/Michael Lin_2/Desktop/Thinkful/Data/ideal_weight.csv')

df.columns = df.columns.map(lambda x: x.replace("'",""))
df['sex'] = [i.replace("'","") for i in df['sex']]


plt.figure()
plt.hist([df['actual'], df['ideal']])
plt.show()

plt.figure()
plt.hist(df['actual']-df['ideal'])
plt.show()

df['sex_c'] = df['sex'].astype('category')

print("")
print("Count of Females and Males in the Dataset:")
print(df['sex_c'].value_counts())

predictors = df[['actual', 'ideal', 'diff']]
target = df['sex_c']

nbg = GaussianNB()
model = nbg.fit(predictors, target)

model_output = model.predict(predictors)
diff = 0
same = 0
for i in range(0,len(target)):
    if target[i] == model_output[i]:
        same += 1
    else:
        diff += 1

print("There are a total of %d points in the dataset and %d of them are mislabeled" %(len(target),diff))

pred1 = pd.DataFrame({'actual': 145, 'ideal': 160, 'diff': -15}, index=[1])
print(model.predict(pred1))

pred2 = pd.DataFrame({'actual': 160, 'ideal': 145, 'diff': 15}, index=[1])
print(model.predict(pred2))
