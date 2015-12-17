# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 13:23:12 2015

@author: Michael Lin_2
"""

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pylab
import numpy as np
import matplotlib.pyplot as plt
import collections
from scipy import stats

# To use previously cleaned up data:
# loansData.to_csv('loansData_clean.csv', header=True, index=False)

# Load the reduced version of the Lending Club Dataset
df = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')
# Drop null rows
df.dropna(inplace=True)

# Clean Interest.Rate Data
InterestRate = df['Interest.Rate'].map(lambda x: round(float(x.rstrip('%')) / 100, 4))

# Clean Loan.Length Data
LoanLength = df['Loan.Length'].map(lambda x: int(x.rstrip(' months')))

# Clean FICO.Range Data and Convert it into FICOScore
temp = df['FICO.Range'].map(lambda x: x.split('-'))
temp = temp.map(lambda x: [int(n) for n in x])
FICORange = [x[0] for x in temp.values]

# Establish Annual Income Based off Monthly Income
AnnualIncome = df['Monthly.Income'].map(lambda x: x*12)

# Convert them into new column of the dataframe
df['InterestRate'] = InterestRate
df['LoanLength'] = LoanLength
df['FICOScore'] = FICORange
df['AnnualIncome'] = AnnualIncome

print(pd.unique(df['Home.Ownership']))
df['HomeOwnership'] = pd.Categorical(df['Home.Ownership']).codes

# Check the Data
print(df.head())

# Fit the Model
est_a = smf.ols(formula="InterestRate ~ AnnualIncome", data=df).fit()
print(est_a.summary())
est_b = smf.ols(formula='InterestRate ~ AnnualIncome + HomeOwnership', data=df).fit()
print(est_b.summary())
est_c = smf.ols(formula='InterestRate ~ AnnualIncome + HomeOwnership + AnnualIncome*HomeOwnership', data=df).fit()
# This is the same as: est_b = smf.ols(formula='InterestRate ~ AnnualIncome*HomeOwnership', data=df).fit()
print(est_c.summary())


## Alternatively using sm: but will need to add constant.  Same Result.
X1 = df['AnnualIncome']
X2 = df[['AnnualIncome', 'HomeOwnership']]
df['Interaction'] = df['AnnualIncome']*df['HomeOwnership']
X3 = df[['AnnualIncome', 'HomeOwnership', 'Interaction']]
y = df['InterestRate']
X1 = sm.add_constant(X1)
X2 = sm.add_constant(X2)
X3 = sm.add_constant(X3)
est1 = sm.OLS(y, X1).fit()
print(est1.summary())
est2 = sm.OLS(y, X2).fit()
print(est2.summary())
est3 = sm.OLS(y, X3).fit()
print(est3.summary())


