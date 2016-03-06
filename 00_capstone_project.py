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
df3 = pd.read_csv('C:/Users/Michael Lin_2/Desktop/Thinkful/Data/lending-club-project/ZIP_2010-2.csv')

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


#####################################################################################################################
#####################################################################################################################
####### DATA CLEANING, ANALYSIS and SOME MORE CLEANING                                    ###########################
#######                                                                                   ###########################
#####################################################################################################################
#####################################################################################################################

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


#####################################################################################################################
#####################################################################################################################
####### MODEL 1: Logistic Regression With NUMERIC predictors                              ###########################
####### Note that we eliminated insignificant predictors                                  ###########################
#####################################################################################################################
#####################################################################################################################

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
warnings.simplefilter(action = "ignore", category = DeprecationWarning)

## WE FIRST CHECKED THE COEFFICIENTS HERE AND DETERMINED THE SIGNIFICANT PREDICTORS BASED ON P-VALUES
## NOTE that the statsmodels.api is not as robust as we added more predictors - so we has to limit the
## number of predictors going into the first logistic model.  We excluded "annual_inc" and "fico_range",
## which we went back to test those two and they turned out to be insignificant against other predictors.
import statsmodels.api as sm
num_predictors_test = ['loan_amnt', 'int_rate', 'installment', 'dti', 'fico_range_high', 'Avg_Median', 'Pop', \
                  'delinq_2yrs', 'pub_rec', 'revol_bal', 'emp_length', 'Median_range', 'loan_over_median', \
                  'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record', 'loan_over_income', \
                  'open_acc', 'revol_util', 'total_acc','Dif_median_from_zip', 'RU_Ratio', 'Dif_mean_median', \
                  'Min_Median']

df_num_test = df.get(num_predictors_test)
df_num_test['Intercept'] = 1.0
imputed_features = df_num_test.median()
imputed_features[['mths_since_last_delinq','mths_since_last_record']] = 0
df_num_test = df_num_test.fillna(imputed_features)
logit = sm.Logit(df['target'], df_num_test)
result = logit.fit()
print (result.summary())

# Turn the target (responses) into a numpy array
target = df.target.values

## Assigned only significant numeric predictors to "num_predictors" and create a new dataframe
num_predictors = ['loan_amnt', 'int_rate', 'installment', 'emp_length', 'dti', \
                  'delinq_2yrs', 'fico_range_high', 'inq_last_6mths', \
                  'mths_since_last_delinq', 'open_acc', 'pub_rec', 'revol_bal', \
                  'total_acc', 'Min_Median', 'Dif_mean_median', 'loan_over_income', 'RU_Ratio']
df_num = df.get(num_predictors)
df_num['Intercept'] = 1.0

'''
## FOR REFERENCE:
#  Originally, we used all of the numeric predictors in our logistic regression; however,
#  the result was dissapointed as the default regression classified all test data as
#  "good loans" as if there were no bad loans.  Therefore, we looked up the regression
#  coefficients and eliminated the predictors with really low coefficients - see "zipped".
num_predictors = ['loan_amnt', 'int_rate', 'installment', 'emp_length', 'annual_inc', \
                  'dti', 'delinq_2yrs', 'fico_range_high', 'inq_last_6mths', \
                  'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', \
                  'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'fico_range', \
                  'Avg_Median', 'Min_Median', 'Pop', 'Dif_mean_median', 'Median_range',\
                  'Dif_median_from_zip', 'loan_over_income', 'loan_over_median', 'RU_Ratio']
df_num.columns.values
lr.coef_[0]
zipped = zip(df_num.columns.values, lr.coef_[0])
print(list(zipped))

## FOR REFERENCE - THE VERY FIRST DS_NUM SELECTIONS purely based on coefficient values
num_predictors = ['loan_amnt', 'int_rate', 'installment', 'dti', 'fico_range_high', \
                  'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record', \
                  'open_acc', 'revol_util', 'total_acc','Dif_median_from_zip', 'RU_Ratio']
                  
## FOR REFERENCE - THE SECOND DS_NUM SELECTIONS based on p-values - THIS IS THE CURRENT SELECTIONS
num_predictors = ['loan_amnt', 'int_rate', 'installment', 'emp_length', 'dti', \
                  'delinq_2yrs', 'fico_range_high', 'inq_last_6mths', \
                  'mths_since_last_delinq', 'open_acc', 'pub_rec', 'revol_bal', \
                  'total_acc', 'Min_Median', 'Dif_mean_median', 'loan_over_income', 'RU_Ratio']

'''

# histogram plot for reference
plt.hist(df['Dif_mean_median'])
# boxplot for reference
df_num.boxplot(column = 'revol_bal')
# scatterplot for reference - limited the boundaries of y-axis here
plt.ylim([-150000, 200000])
plt.scatter(df['target'], df_num['Dif_mean_median'])

## We need to fill (or impute) the "NaN" values
## We decided to use median() to fill all the missing values with the exceptions of 
## 'mths_since_last_delinq' and 'mths_since_last_record' - for these two, we think
## it's more appropriate to use zeros.
## In fact, the only remaining predictor with NaN was 'revol_util'.
imputed_features = df_num.median()
imputed_features[['mths_since_last_delinq']] = 0
# imputed_features[['mths_since_last_delinq','mths_since_last_record']] = 0
df_num = df_num.fillna(imputed_features)

## Turning the predictor dataframe into an array.  Note that "target" is already an array
df_num_array = df_num.values
target

## Split the Data using train_test_split function:
ar_num_train, ar_num_test, target_train, target_test = train_test_split(df_num_array, target, test_size = 0.30, random_state=0)
ar_num_train.shape
ar_num_test.shape

## Fit the logistic regression model
lr = LogisticRegression()
lr.fit(ar_num_train, target_train)

## Predict test set target values
target_predicted = lr.predict(ar_num_test)

## Accessing accuracy (Overall Accuracy Only)
print(accuracy_score(target_test, target_predicted))
print(pd.crosstab(target_test, target_predicted, rownames=['True'], colnames=['Predicted'], margins=True))

## Here is the population ratio of the good loan - note that this is the accuracy we need to beat
print('The default population good loan rate is {}' .format(target_test.mean()))

## Here we tested different 'class_wight' in our model inorder to imporve the accuracy of the "good loan" prediction only
ls = []
for i in range(10):
    lr2 = LogisticRegression(class_weight={0: i})
    lr2.fit(ar_num_train, target_train)
    target_predicted = lr2.predict(ar_num_test)
    ls.append(accuracy_score(target_test, target_predicted))
    print(pd.crosstab(target_test, target_predicted, rownames=['True'], colnames=['Predicted'], margins=True))


#####################################################################################################################
#####################################################################################################################
####### MODEL 2: Logistic Regression With Numeric and Categorical Predictors              ###########################
#####################################################################################################################
#####################################################################################################################

## CONTINUE FROM NUMERIC-ONLY PREDICTOR MODEL - WE ADDED THE CATEGROICAL PREDICTORS
## Right off the bat, we decided to elminate 'papymnt_plan' because one of the value only has a sample size of 2
## Right off the bat, we also decided to eliminate 'zip_code' as there are just too many values
## Note that we only retained the numeric predictors that worked in Model 1
## Finally, we later further eliminate 'sub_grade' because its significant is captured in 'grade' already -
## in particular grade 'A'.

target = df.target.values

logistic_predictors = ['loan_amnt', 'int_rate', 'installment', 'grade', 'sub_grade', \
       'emp_length', 'home_ownership', 'verification_status','purpose', 'addr_state', \
       'dti', 'delinq_2yrs', 'fico_range_high', 'inq_last_6mths', 'mths_since_last_delinq', \
       'open_acc', 'pub_rec', 'revol_bal', 'total_acc', 'initial_list_status', \
       'Min_Median', 'Dif_mean_median', 'loan_over_income', 'RU_Cat', 'RU_Ratio']

df_lr = df.get(logistic_predictors)

'''
## FOR REFERENCE - this would be the full varialbes logistic predictor
logistic_predictors_full = ['loan_amnt', 'int_rate', 'installment', 'grade', 'sub_grade',
       'emp_length', 'home_ownership', 'annual_inc', 'verification_status',
       'purpose', 'addr_state', 'dti', 'delinq_2yrs',
       'fico_range_high', 'inq_last_6mths', 'mths_since_last_delinq',
       'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal',
       'revol_util', 'total_acc', 'initial_list_status', 'fico_range',
       'Avg_Median', 'Min_Median', 'Pop', 'Dif_mean_median', 'Median_range',
       'Dif_median_from_zip', 'loan_over_income', 'loan_over_median', 'RU_Cat', 'RU_Ratio']
'''

## Prep the dataframe with dummy variables (for the categorical predictors)
## Note that each of the first dummy variable is skipped and not retained in the dataframe.
dummified = ['grade', 'sub_grade', 'home_ownership', 'verification_status', 'purpose', 'addr_state', 'initial_list_status', 'RU_Cat']
df_lr = df_lr.drop(dummified, axis=1)

dummy_grade = pd.get_dummies(df['grade'], prefix='d_grade')
df_lr = df_lr.join(dummy_grade.ix[:, :'d_grade_F'])
## As mentioned above, 'subgrade' isn't significant because there is no significant difference within each grade
# dummy_subgrade = pd.get_dummies(df['sub_grade'], prefix='d_sub_grade')
# df_lr = df_lr.join(dummy_subgrade.ix[:, 1:])
dummy_home_ownership = pd.get_dummies(df['home_ownership'], prefix='d_home_ownership')
df_lr = df_lr.join(dummy_home_ownership.ix[:, 1:])
dummy_verification_status = pd.get_dummies(df['verification_status'], prefix='d_verification_status')
df_lr = df_lr.join(dummy_verification_status.ix[:, 1:])
dummy_purpose = pd.get_dummies(df['purpose'], prefix='d_purpose')
df_lr = df_lr.join(dummy_purpose.ix[:, 1:])
dummy_addr_state = pd.get_dummies(df['addr_state'], prefix='d_addr_state')
df_lr = df_lr.join(dummy_addr_state.ix[:, 1:])
dummy_initial_list_status = pd.get_dummies(df['initial_list_status'], prefix='d_initial_list_status')
df_lr = df_lr.join(dummy_initial_list_status.ix[:, 1:])
dummy_RU_Cat = pd.get_dummies(df['RU_Cat'], prefix='d_RU_Cat')
df_lr = df_lr.join(dummy_RU_Cat.ix[:, 1:])

df_lr['intercept'] = 1.0

## Impute median values on numeric features with NaN values with the exception of
## 'mths_since_last_delinq' - as we earlier decided to impute zero instead.
imputed_features = df_lr.median()
imputed_features[['mths_since_last_delinq']] = 0
# imputed_features[['mths_since_last_delinq','mths_since_last_record']] = 0
df_lr = df_lr.fillna(imputed_features)

## Turning the predictor dataframe into an array.  Note that "target" is already an array
df_lr_array = df_lr.values
target

## Split the Data using train_test_split function:
ar_lr_train, ar_lr_test, target_train, target_test = train_test_split(df_lr_array, target, test_size = 0.30, random_state=0)
ar_lr_train.shape
ar_lr_test.shape

## Fit the logistic regression model
lr_full = LogisticRegression()
lr_full.fit(ar_lr_train, target_train)

## Predict test set target values
target_predicted2 = lr_full.predict(ar_lr_test)

## Accessing accuracy (Overall Accuracy Only)
print(accuracy_score(target_test, target_predicted2))
print(pd.crosstab(target_test, target_predicted2, rownames=['True'], colnames=['Predicted'], margins=True))

## Here we tested different 'class_wight' in our model inorder to imporve the accuracy of the "good loan" prediction only
ls = []
for i in range(10):
    lr_full2 = LogisticRegression(class_weight={0: i})
    lr_full2.fit(ar_lr_train, target_train)
    target_predicted2 = lr_full2.predict(ar_lr_test)
    ls.append(accuracy_score(target_test, target_predicted2))
    print(pd.crosstab(target_test, target_predicted2, rownames=['True'], colnames=['Predicted'], margins=True))

## Here we use the statsmodels to check the p-value of each coefficients.  Note that we had to use
## 'method = 'basinhopping' because the default method somehow doesn't work with our model when not using
## 'dummy_subgrade' as predictors... not sure why here.
logit = sm.Logit(df['target'], df_lr)
result = logit.fit(method='basinhopping')
print (result.summary())
























'''
https://www.mercurial-scm.org/
https://github.com/shubhabrataroy/Thinkful/blob/master/Curriculum/SpamFilter.ipynb
https://github.com/ga-students/DAT_SF_13/blob/master/labs/DAT13-lab09-Solution.ipynb
http://blog.yhat.com/posts/logistic-regression-and-python.html
http://machinelearningmastery.com/get-your-hands-dirty-with-scikit-learn-now/


http://www.dataschool.io/logistic-regression-in-python-using-scikit-learn/
http://www.bogotobogo.com/python/scikit-learn/scikit-learn_logistic_regression.php
http://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html
https://github.com/dzorlu/GADS/wiki/Scikit-Learn-and-Logistic-Regression


http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html
http://scikit-learn.org/stable/modules/preprocessing.html
"Multinomial"
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
http://scikit-learn.org/stable/modules/linear_model.html
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html

http://stackoverflow.com/questions/30972029/how-does-the-class-weight-parameter-in-scikit-learn-work



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
       'Dif_median_from_zip', 'loan_over_income', 'loan_over_median', 'RU_Cat', 'RU_Ratio']

# Alternatively (but cannot reproduce this result)
#  from sklearn.cross_validation import train_test_split
#  df_train, df_test = train_test_split(df, test_size = 0.7)

'''
            
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