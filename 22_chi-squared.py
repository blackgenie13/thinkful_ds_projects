# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 19:39:59 2015

@author: Michael Lin_2
"""

import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import collections

# Load the reduced version of the Lending Club Dataset
loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')
# Drop null rows
loansData.dropna(inplace=True)

freq = collections.Counter(loansData['Open.CREDIT.Lines'])

plt.figure()
plt.bar(freq.keys(), freq.values(), width=1)
plt.savefig('2.2.3 opencredit.png')

templist = []

for value in freq.values():
    templist.append(value)

chi, p = stats.chisquare(templist)

print('chi is {0}' .format(chi))
print('p is {0}' .format(p))
