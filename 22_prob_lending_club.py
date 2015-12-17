# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 18:57:19 2015

@author: Michael Lin_2
"""

import matplotlib.pyplot as plt
import pandas as pd

loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

loansData.dropna(inplace=True)

plt.figure()
loansData.boxplot(column='Amount.Requested')
plt.savefig('2.2.2 request boxplot.png')

plt.figure()
loansData.hist(column='Amount.Requested')
plt.savefig('2.2.2 request histogram.png')

plt.figure()
graph = stats.probplot(loansData['Amount.Requested'], dist="norm", plot=plt)
plt.savefig('2.2.2 request qqplot.png')

# The distribution of 'Amount.Requested' and 'Amount.Funded.By.Investors' are
# pretty similar based on all three graphs.  This is not surprising as people
# are funding the amount of what lenders are asking (I'm assuming...)