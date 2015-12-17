# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 17:20:51 2015

@author: Michael Lin_2
"""

import pandas as pd
from scipy import stats

data = '''Region,Alcohol,Tobacco
North,6.47,4.03
Yorkshire,6.13,3.76
Northeast,6.19,3.77
East Midlands,4.89,3.34
West Midlands,5.63,3.47
East Anglia,4.52,2.92
Southeast,5.89,3.20
Southwest,4.79,2.71
Wales,5.27,3.53
Scotland,6.08,4.51
Northern Ireland,4.02,4.56'''

# dataframe construction
data = data.splitlines() # make a list for each row
data = [i.split(',') for i in data] # make columns for each list
column_names = data[0] # this is the first row
data_rows = data[1::] # these are all the following rows of data
df = pd.DataFrame(data_rows, columns=column_names)

# change data type
df['Alcohol'] = df['Alcohol'].astype(float)
df['Tobacco'] = df['Tobacco'].astype(float)

# print: mean, median, mode, range, variance, and standard deviation with full text

a_mean = df['Alcohol'].mean() 
a_median = df['Alcohol'].median() 
a_mode = stats.mode(df['Alcohol']) 
a_range = max(df['Alcohol']) - min(df['Alcohol'])
a_var = df['Alcohol'].var() 
a_sd= df['Alcohol'].std() 

t_mean = df['Tobacco'].mean() 
t_median = df['Tobacco'].median() 
t_mode = stats.mode(df['Tobacco']) 
t_range = max(df['Tobacco']) - min(df['Tobacco'])
t_var = df['Tobacco'].std()
t_sd = df['Tobacco'].var() 

desc = ['mean', 'median', 'mode', 'range', 'variance', 'standard deviation']
a = [a_mean, a_median, a_mode, a_range, a_var, a_sd]
t = [t_mean, t_median, t_mode, t_range, t_var, t_sd]

for n in range (0, 5):
    print ('The {0} for Alcohol of dataset is {1}'.format(desc[n], a[n]))
    print ('The {0} for Tobacco of dataset is {1}'.format(desc[n], t[n]))
    