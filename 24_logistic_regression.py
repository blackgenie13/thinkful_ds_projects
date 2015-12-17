# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 15:28:29 2015

@author: Michael Lin_2
"""

import pandas as pd
import statsmodels.api as sm
import pylab
import numpy as np
import matplotlib.pyplot as plt
import collections
from scipy import stats

# To use previously cleaned up data:
# loansData.to_csv('loansData_clean.csv', header=True, index=False)

# Load the reduced version of the Lending Club Dataset
loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')
# Drop null rows
loansData.dropna(inplace=True)

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

# Add the IF_TF stating 0 if interest is < 12% and 1 otherwise
IR_TF = loansData['InterestRate'].map (lambda x: 0 if x < 0.12 else 1)
loansData['IR_TF'] = IR_TF

# Verify IR_IF is correct
loansData.iloc[:,[2,17]][0:10]

# Add Constant Column
Intercept = loansData['IR_TF'].map(lambda x: 1.0)
loansData['Intercept'] = Intercept

ind_vars = [ 'Intercept', 'FICOScore', 'Amount.Requested']

logit = sm.Logit(loansData['IR_TF'], loansData[ind_vars])
result = logit.fit()
coeff = result.params
print(coeff)

def logistic_function(ficoscore, loanamount):
    p = 1/( 1 + np.exp(coeff[0] + coeff[1]*(ficoscore) + coeff[2]*(loanamount)))  
    return p

print('chance of getting loan for 720 Fico Score and $10,000 loan is: {0}' .format(logistic_function(720, 10000)))

plt.figure()
x = np.linspace(0,950,951)
y = 1/(1 + np.exp(coeff[0] + coeff[1]*x + coeff[2]*10000))
plt.plot(x, y)
plt.savefig('2.4.3 10000 loan.png')


