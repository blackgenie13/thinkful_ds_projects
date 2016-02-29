# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 17:30:55 2016

@author: Michael Lin_2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as skm
import pylab as pl

### DATA IMPORTS

df1 = pd.read_csv('C:/Users/Michael Lin_2/Desktop/Thinkful/Data/lending-club-project/LoanStats3a_securev1.csv', skiprows=1)
df2 = pd.read_csv('C:/Users/Michael Lin_2/Desktop/Thinkful/Data/lending-club-project/LoanStats3b_securev1.csv', skiprows=1)
df3 = pd.read_csv('C:/Users/Michael Lin_2/Desktop/Thinkful/Data/lending-club-project/ZIP_2010.csv')

df1.info() #names of all columns
df1['loan_status'][39784:39800] #check out the gap - need to close it
df1['loan_status'].unique() #check out all unique column names

# Getting rid of rows with status = "does not meet the credit policy" - no longer valid now.
df1 = df1[df1.loan_status.str.contains("Does not meet the credit policy.") == False]
# df2 = df2[df2.issue_d != 'Dec-2013']
df3 = df3.rename(columns={'Zip': 'zip_code'})

df12 = pd.concat([df1, df2])
df12.shape
# Here are some code that would term strings of time into time variables - however, we decided to 
# exclude all time base variables as predictors as they do not make sense for new borrowers of later time.
# df12['issue_d'] = df12[pd.to_datetime(df12['issue_d'])
# df12['issue_year'] = pd.DatetimeIndex(df12['issue_d']).year
# df12['issue_month'] = pd.DatetimeIndex(df12['issue_d']).month


### DATA CLEANING

# Only extract useful predictors
predictors = ['loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade', 'emp_title', \
              'emp_length', 'home_ownership', 'annual_inc', 'verification_status', 'loan_status', \
              'pymnt_plan', 'purpose', 'title', 'zip_code', 'addr_state', 'dti', 'delinq_2yrs', \
              'earliest_cr_line', 'fico_range_low', 'fico_range_high', 'inq_last_6mths', \
              'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec', \
              'revol_bal', 'revol_util', 'total_acc', 'initial_list_status', 'application_type', \
              'total_rev_hi_lim']

df12 = df12[predictors]

# Dropping 'title' as the columns holds too many different strings - they are messy and
# likely contributes no values as a predictors.  Also many spelling errors.
df12.drop('title', axis=1, inplace=True)

# Dropping 'application_type' because there is only one value, which is "INDIVIDUAL"
df12.drop('application_type', axis=1, inplace=True)

# Dropping 'earliest_cr_line' because this is a time based predictors, it does not make sense to use this
# as predictors because this predictors, even if significant, would not apply to the new borrowers.
df12.drop('earliest_cr_line', axis=1, inplace=True)

# Strip '%' from 'int_rate' and 'revol_util' and reformat them as float type.
df12.int_rate = pd.Series(df12.int_rate).str.replace('%', '').astype(float)
df12.revol_util = pd.Series(df12.revol_util).str.replace('%', '').astype(float)

# We looked at the first word of all 'emp_title' - and found that similar to 'title', the free text 
# field is too messy to contain any useful information as predictor, even if it's just the first word
df12.emp_title = df12.emp_title.str.lower()
temp = df12['emp_title'].str[0:].str.split(' ', return_type='frame')
df12['emp_title'] = temp[0]
print(df12.emp_title.value_counts())
# Dropping 'emp_title" for similar reasons as dropping 'title'
df12.drop('emp_title', axis=1, inplace=True)

# We need to clean up the 'emp_length' and convert it into numeric format
print(df12.emp_length.value_counts())
df12.replace('n/a', np.nan, inplace=True)
df12.emp_length.fillna(value=0, inplace=True)
df12['emp_length'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
df12['emp_length'] = df12['emp_length'].astype(int)
print(df12.emp_length.value_counts())

# Exclude all loans that are 60 months (retaining only 36-month loans) and drop the field as predictor
df12 = df12[df12['term']==' 36 months']
print(df12.term.value_counts())
df12.drop('term', axis=1, inplace=True)

# The two 'fico_range_xxx" variables give primiarly the same range - either in range of 4 or 5;
# therefore, we'll only retain one of the fico score along with the range and drop the other fico score
df12['fico_range'] = df12['fico_range_high'] - df12['fico_range_low']
print(df12.fico_range.value_counts())
df12.drop('fico_range_low', axis=1, inplace=True)

### Merge the zip code median income dataset with Lending Club's dataset
# Join zip_code dataset with lending club dataset
df = pd.merge(df12, df3, on='zip_code')
print(df.zip_code.value_counts())

# Adding a few more calculated predictors...
df['Median_range'] = df['Max_Median'] - df['Min_Median']
df['Dif_median_from_zip'] = df['annual_inc'] - df['Avg_Median']
df['Dif_mean_from_zip'] = df['annual_inc'] - df['Est_Mean']
df['loan_over_income'] = df['loan_amnt']/df['annual_inc']
df['loan_over_median'] = df['loan_amnt']/df['Avg_Median']

# Finally we categorize the target variable 'target' based on 'loan_status'
# and then drop the 'loan_status'
print(df.loan_status.value_counts())
df['target'] = np.nan
bad_loan = ["Late (16-30 days)", "Late (31-120 days)", "Default", "Charged Off"]
good_loan = ["Fully Paid", "Current", "In Grace Period"]
# target = 0 indicates a bad loan
df.ix[df.loan_status.isin(bad_loan), 'target'] = 0
# target = 1 indicates a good loan
df.ix[df.loan_status.isin(good_loan), 'target'] = 1
print(df.groupby(['loan_status', 'target']).loan_status.count().groupby(level=['loan_status','target']).value_counts())
df.drop('loan_status', axis=1, inplace=True)


### DATA ANALYSIS & SOME MORE CLEANING

# 'Est_tot_income' and 'Est_household' were created to estimate the mean income of each zip code based on
# 'Pop' and 'income' so they are 100% correlated with exiting variable such as 'Pop'.  Here we drop both.
df.drop('Est_tot_income', axis=1, inplace=True)
df.drop('Est_household', axis=1, inplace=True)

# It would seem that 'Avg_Median' and 'Est_Mean' are highly correlated at 0.951.
# As a result, we decided to drop 'Est_Mean'
plt.scatter(df.Avg_Median, df.Est_Mean)
np.corrcoef(df.Avg_Median, df.Est_Mean)
df.drop('Est_Mean', axis=1, inplace=True)

# It would seem that 'Dif_median_from_zip' and 'Dif_mean_from_zip' are highly correlated at 0.994.
# As a result, we decided to drop 'Dif_mean_from_zip'
plt.scatter(df.Dif_median_from_zip, df.Dif_mean_from_zip)
np.corrcoef(df.Dif_median_from_zip, df.Dif_mean_from_zip)
df.drop('Dif_mean_from_zip', axis=1, inplace=True)

# 'Avg_Median' and 'Median_range" don't seem to be strongly correlated (at 0.584)
plt.scatter(df.Avg_Median, df.Median_range)
np.corrcoef(df.Avg_Median, df.Median_range)
# 'loan_over_income' and 'loan_over_median' don't seem to be strongly correlated (at 0.549)
plt.scatter(df.loan_over_income, df.loan_over_median)
np.corrcoef(df.loan_over_income, df.loan_over_median)
# 'Min_Median' and 'Median_range' don't seem to be correlated at all (at -0.077)
plt.scatter(df.Min_Median, df.Median_range)
np.corrcoef(df.Min_Median, df.Median_range)
# 'total_acc" and "open_acc" seem to have some correlation at 0.673, we'll retain both as predictors for now.
plt.scatter(df.total_acc, df.open_acc)
np.corrcoef(df.total_acc, df.open_acc)
# 'fico_range_high' and 'int_rate' seem to be somewhat negatively correlated at -0.670, we'll retain both as predictors for now.
plt.scatter(df.fico_range_high, df.int_rate)
np.corrcoef(df.fico_range_high, df.int_rate)
# 'loan_amnt" and "loan_over_median" seem to have somewhat strong correlation at 0.883, we'll retain both as predictors for now.
plt.scatter(df.loan_amnt, df.loan_over_median)
np.corrcoef(df.loan_amnt, df.loan_over_median)

# 'Max_Median' and 'Median_range' seem to be highly correlated at 0.930.
# As a result, we decided to drop "Max_Median" and retain "Median_range"
plt.scatter(df.Max_Median, df.Median_range)
np.corrcoef(df.Max_Median, df.Median_range)
df.drop('Max_Median', axis=1, inplace=True)

# Note: there may be some potential outliers in the variable 'total_rev_hi_lim'.  In addition, the varaible isn't complete 
# and it seems to be strongly correlated with 'revol_bal'; therefore, we'll drop it as well.
plt.scatter(df.revol_bal, df.total_rev_hi_lim)
np.corrcoef(df.revol_bal, df.total_rev_hi_lim)
df.drop('total_rev_hi_lim', axis=1, inplace=True)

# Next we identified 4 EXTREME outliers in the predictor 'revol_bal' that is over 0.9-million.  We decided re-assign these four points
# with values that is the fifth highest in the column - 605,627 (at index 155881) so avoid extreme outliers problems.
plt.hist(df['revol_bal'])
plt.boxplot(df['revol_bal'])
df['revol_bal'].order(ascending=0).head(10)
df['revol_bal'][132714] = df['revol_bal'][155881]
df['revol_bal'][50689] = df['revol_bal'][155881]
df['revol_bal'][51162] = df['revol_bal'][155881]
df['revol_bal'][72921] = df['revol_bal'][155881]
# pd.scatter_matrix(df[['loan_amnt','int_rate','sub_grade','annual_inc','open_acc','total_acc']], alpha=0.05, figsize=(10,10), diagonal='hist')
# pd.scatter_matrix(df[['loan_amnt','int_rate','sub_grade','annual_inc','open_acc','total_acc','dti','fico_range_high','revol_bal','revol_util','Avg_Median','Min_Median','Pop','Dif_mean_median','loan_over_median']], alpha=0.05, figsize=(10,10), diagonal='hist')


## Scatter Plot using Heat Map
sns.set(style="white")
# Compute the correlation matrix
corr = df[['loan_amnt','int_rate','sub_grade','annual_inc','open_acc','total_acc','dti','fico_range_high','revol_bal','revol_util','Avg_Median','Min_Median','Pop','Dif_mean_median','loan_over_median']].corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr)


### SPLITTING DATA INTO TRAINING AND TESTING SETS
np.random.seed(2016)
msk = np.random.rand(len(df)) < 0.70
df_train = df[msk]
df_test = df[~msk]

var = ['Intercept','loan_amnt', 'int_rate', 'installment', 'grade', 'sub_grade',
       'emp_length', 'home_ownership', 'annual_inc', 'verification_status',
       'pymnt_plan', 'purpose', 'zip_code', 'addr_state', 'dti', 'delinq_2yrs',
       'fico_range_high', 'inq_last_6mths', 'mths_since_last_delinq',
       'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal',
       'revol_util', 'total_acc', 'initial_list_status', 'fico_range',
       'Avg_Median', 'Min_Median', 'Pop', 'Dif_mean_median', 'Median_range',
       'Dif_median_from_zip', 'loan_over_income', 'loan_over_median']

# Alternatively (but cannot reproduce this result)
#  from sklearn.cross_validation import train_test_split
#  df_train, df_test = train_test_split(df, test_size = 0.7)



            
'''
NEXT STEP:
1. Extract all the fields that make sense (what do investors see when selecting loans to invest?)
2. Add more variables by constructing new variables
3. Data Analysis - and get rid of highly correlated variables.

Plot histogram:

plt.hist(df['loan_amnt']/df['annual_inc'])

Question to ask:

1. how should we treat dates?  Break it into month and year or leave it as ordinary variable?
   also - should I transform date variable into actual date type?
2. how should we treat FICO score range: upper range vs. lower range?
3. what can I do with zip code "123xx"?
4. how can I merge the zip code?
## df = pd.merge(df12, df3, on='zip_code')
5. what are some of the meaningful graph I can do gegraphically?
6. What kind of model should I run?  Random Forrest?

Discussion
Consider only do 36-month term.
Not to date as predictor....
Use both FICO score as two predictors: Check the correlation of these two first.

Check the correlation of Median Income among (average, min, and max)
Consider using RANGE (Max - Min) and check correlation with AVERAGE

Difference between individual income and regional income.

logistic regression, svm, then try random forrest last

distribution of different features

bubble graphs on geographic map (high/low income)

Correlations matrix for all predictors (as visualization)

Confusion Matrix - (how accurate is the model)

Make sure to do cross-validation.

'''