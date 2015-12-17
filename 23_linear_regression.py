# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 16:04:46 2015

@author: Michael Lin_2
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

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
plt.figure()
p = loansData['FICOScore'].hist()
plt.savefig('FICOScore Hist.png')

# Scatter Plot of all Columns
plt.figure()
a = pd.scatter_matrix(loansData, alpha=0.05, figsize=(10,10), diagonal='hist')
plt.savefig('Scatter Plot.png')

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

print(f.summary())