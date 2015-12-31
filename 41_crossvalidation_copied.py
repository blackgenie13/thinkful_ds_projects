# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 16:04:46 2015

@author: Michael Lin_2
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import sklearn.linear_model as ln
from sklearn.cross_validation import train_test_split, cross_val_score

### Added to avoid Deprecation Warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

# Clean Interest.Rate Data
InterestRate = loansData['Interest.Rate'].map(lambda x: round(float(x.rstrip('%')) / 100, 4))

# Clean Loan.Length Data
LoanLength = loansData['Loan.Length'].map(lambda x: int(x.rstrip(' months')))

# Clean FICO.Range Data and Convert it into FICOScore
temp = loansData['FICO.Range'].map(lambda x: x.split('-'))
temp = temp.map(lambda x: [int(n) for n in x])
FICORange = [x[0] for x in temp.values]


# Convert them into new column of the dataframe
loansData['InterestRate'] = InterestRate
loansData['LoanLength'] = LoanLength
loansData['FICOScore'] = FICORange

# FICOScore Histogram
## plt.figure()
## p = loansData['FICOScore'].hist()
## plt.savefig('FICOScore Hist.png')

# Scatter Plot of all Columns
## plt.figure()
## a = pd.scatter_matrix(loansData, alpha=0.05, figsize=(10,10), diagonal='hist')
## plt.savefig('Scatter Plot.png')

intrate = loansData['InterestRate']
loanamt = loansData['Amount.Requested']
fico = loansData['FICOScore']

# The dependent variable
y = np.matrix(intrate).transpose()
# The independent variables shaped as columns
x1 = np.matrix(fico).transpose()
x2 = np.matrix(loanamt).transpose()
# Put the two columns together to create an input matrix
x = np.column_stack([x1,x2])

# Create the Linear Model
X = sm.add_constant(x)
model = sm.OLS(y,X)
f = model.fit()

## print(f.summary())

###### Adding from here...

#Subset data into independent and dependent
X1 = loansData[['Amount.Requested', 'FICOScore']]
Y1 = loansData['InterestRate']
#Training and test sets
X_train, X_test, y_train, y_test = train_test_split(X1, Y1)
#Size of training and test sets
X_train.shape
#(1875, 2)
X_test.shape
#(625, 1)
y_train.shape
#(1875, 2)
y_test.shape
#(625, 1)

#Model building 

#Other OLS method with scikitlearn
#Create linear regression object
regr = ln.LinearRegression()
#Train model with training sets
regr.fit(X_train, y_train)

#Evaluation

#Cross validation score for MSE 10 folds
MSE_Scores = cross_val_score(regr, X_train, y_train, scoring = 'mean_squared_error', cv = 10)
#Take average of all cross validation folds
mean_MSE = np.mean(MSE_Scores)
#-0.0006163776021605567
#Cross validation score for MAE 10 fold
MAE_Scores = cross_val_score(regr, X_train, y_train, scoring = 'mean_absolute_error', cv = 10)
#Take average of all cross validation folds
mean_MAE = np.mean(MAE_Scores)
#-0.01959622777354547
#Cross validation score for R2 10 folds
R2_Scores = cross_val_score(regr, X_train, y_train, scoring = 'r2', cv = 10)
#Take average of all cross validation folds
mean_R2 = np.mean(R2_Scores)
#0.6505887138996289

print(mean_MSE)
print(mean_MAE)
print(mean_R2)