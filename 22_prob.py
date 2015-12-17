# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 18:34:39 2015

@author: Michael Lin_2
"""

import collections
import matplotlib.pyplot as plt
import numpy as np 
import scipy.stats as stats

freq_data = [1, 4, 5, 6, 9, 9, 9]

c = collections.Counter(freq_data)

# calculate the number of instances in the list
count_sum = sum(c.values())
for k,v in c.items():
  print("The frequency of number " + str(k) + " is " + str(float(v) / count_sum))
 
data = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 4, 4, 4, 5, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 9, 9]
plt.figure()
plt.boxplot(data)
plt.savefig("2.2.1 boxplot.png")
plt.figure()
plt.hist(data, histtype='bar')
plt.savefig("2.2.1 histogram.png")

plt.figure()
test_data = np.random.normal(size=1000)   
graph1 = stats.probplot(test_data, dist="norm", plot=plt)
plt.savefig("2.2.1 norm_qq.png") #this will generate the first graph
plt.figure()
test_data2 = np.random.uniform(size=1000)   
graph2 = stats.probplot(test_data2, dist="norm", plot=plt)
plt.savefig("2.2.1 uniform_qq.png") #this will generate the second graph