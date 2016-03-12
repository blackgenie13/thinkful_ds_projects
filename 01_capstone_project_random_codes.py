# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 17:39:38 2016

@author: Michael Lin_2
"""
'''
DEALING WITH UNBALANCE DATA:
-----------------------------------
https://github.com/fmfn/UnbalancedDataset
https://github.com/fmfn/UnbalancedDataset/blob/master/notebook/Notebook_UnbalancedDataset.ipynb


QUESTIONS:
- What are some of the ways we can appropriately select predictors? (Eliminate unwanted ones)
  This was difficult to do given that sklearn doesn't offer p-value for each coefficients.
  TRY THIS: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest
  
- How can I effectively check for interactions among predictors?  Or non-linear relationship?
  TRY THIS: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html

- If there are interaction - how should I refit the model?
  Creating new features if there's interaction.

- Should we consider writing models only for a subset of data?  For example:
  Fit a separate model for Grade A, Grade B, Grade C; or fit a separate model for each state.
  If so, how can I analyze the data to determine which subset is worth the effort to build
  separated models? (i.e. How can I know that different grades actually behave differently?)
  
- How can I configure Random Forrest parameters to make it favoring "bad loans" prediction?
  i.e. more prediction for 'target'=0.  The parameter class_weight={0:10} doesn't seem to be
  working.  I want it so that predicted=0 if there's a leaf that has more than X% of target=0.
  
  --TRY TO MAKE IT BALANCE
  -- SAMPLE WEIGHT INSTEAD OF CLASS WEIGHT
  -- http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.fit
  
    
- Follow-up Question: Do I need to put Continuous Predictors into Bins????? If so, is there a
  package that can easily done this?
  -- NO NEED TO BIN
  
- What other parameters can I tweek between Random Forrest and Logistic Regression?
  
  RANDOM FORREST..............  
  -- NUMBER OF LEAFS
  -- MAXIMUM DEPTH
  -- NUMBER OF SPLITS
  -- http://stackoverflow.com/questions/30102973/how-to-get-best-estimator-on-gridsearchcv-random-forest-classifier-scikit
  
  LOGISTIC REGRESSION.............
  -- http://stackoverflow.com/questions/30102973/how-to-get-best-estimator-on-gridsearchcv-random-forest-classifier-scikit

CHECK THIS OUT WHEN YOU HAVE TIME:  
http://scikit-learn.org/stable/tutorial/machine_learning_map/
'''




#display the breakout
df.groupby(['grade'])['target'].mean()
df.groupby(['sub_grade']).target.count()

## BAD WAY TO CALCULATE ROI
#df.groupby(['grade'])['int_rate'].mean() * df.groupby(['grade'])['target'].mean()
#df.groupby(['grade'])['int_rate'].mean() * df.groupby(['grade'])['target'].mean() - df.groupby(['grade'])['int_rate'].mean()
## BETTER WAY TO CALCULATE ROI
df['roi'] = df.int_rate * df.target
df.groupby(['grade'])['roi'].mean()
df.groupby(['grade'])['roi'].mean() - df.groupby(['grade'])['int_rate'].mean()
df[df['grade'].isin(['E','F','G'])].groupby(['grade']).target.count()

df['roi'].mean()
(df['roi']==0).count()
df[df['roi']!=0].roi.mean()


target = df.target.values

grade
sub_grade
home_ownership
verification_status
pymnt_plan          XX ELIMINATED DUE TO TOO LITTLE SAMPLE SIZE ON ONE VALUE
purpose
zip_code            XX TOO MANY OF THEM!!
addr_state
initial_list_status
RU_Cat


logistic_predictors_full = ['loan_amnt', 'int_rate', 'installment', 'grade', 'sub_grade',
       'emp_length', 'home_ownership', 'annual_inc', 'verification_status',
       'purpose', 'addr_state', 'dti', 'delinq_2yrs',
       'fico_range_high', 'inq_last_6mths', 'mths_since_last_delinq',
       'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal',
       'revol_util', 'total_acc', 'initial_list_status', 'fico_range',
       'Avg_Median', 'Min_Median', 'Pop', 'Dif_mean_median', 'Median_range',
       'Dif_median_from_zip', 'loan_over_income', 'loan_over_median', 'RU_Cat', 'RU_Ratio']
       
logistic_predictors = ['loan_amnt', 'int_rate', 'installment', 'grade', 'sub_grade',
       'emp_length', 'home_ownership', 'verification_status',
       'purpose', 'addr_state', 'dti', 'delinq_2yrs',
       'fico_range_high', 'inq_last_6mths', 'mths_since_last_delinq',
       'open_acc', 'pub_rec', 'revol_bal',
       'total_acc', 'initial_list_status',
       'Min_Median', 'Dif_mean_median', 
       'loan_over_income', 'RU_Cat', 'RU_Ratio']



'''
## FOR REFERENCE - THE VERY FIRST DS_NUM SELECTIONS purely based on coefficient values
num_predictors = ['loan_amnt', 'int_rate', 'installment', 'dti', 'fico_range_high', \
                  'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record', \
                  'open_acc', 'revol_util', 'total_acc','Dif_median_from_zip', 'RU_Ratio']

## FOR REFERENCE - THE SECOND DS_NUM SELECTIONS based on p-values
num_predictors = ['loan_amnt', 'int_rate', 'installment', 'emp_length', 'dti', \
                  'delinq_2yrs', 'fico_range_high', 'inq_last_6mths', \
                  'mths_since_last_delinq', 'open_acc', 'pub_rec', 'revol_bal', \
                  'total_acc', 'Min_Median', 'Dif_mean_median', 'loan_over_income', 'RU_Ratio']
'''

df_lr = df.get(logistic_predictors)

## Get Dummy Variables
dummified = ['grade', 'sub_grade', 'home_ownership', 'verification_status', 'purpose', 'addr_state', 'initial_list_status', 'RU_Cat']
df_lr = df_lr.drop(dummified, axis=1)

dummy_grade = pd.get_dummies(df['grade'], prefix='d_grade')
df_lr = df_lr.join(dummy_grade.ix[:, :'d_grade_F'])
## subgrade turns out to be not significant.
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

## Impute some numeric features
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

ls = []
for i in range(10):
    lr_full2 = LogisticRegression(class_weight={0: i})
    lr_full2.fit(ar_lr_train, target_train)
    target_predicted2 = lr_full2.predict(ar_lr_test)
    ls.append(accuracy_score(target_test, target_predicted2))
    print(pd.crosstab(target_test, target_predicted2, rownames=['True'], colnames=['Predicted'], margins=True))


## We can check p-value here........
import statsmodels.api as sm

logit = sm.Logit(df['target'], df_lr)
result = logit.fit(method='basinhopping')
print (result.summary())


target = df.target.values

'''
# Using only Numeric Values only
num_predictors = ['loan_amnt',
'int_rate',
'installment',
'emp_length',
'annual_inc',
'dti',
'delinq_2yrs',
'fico_range_high',
'inq_last_6mths',
'mths_since_last_delinq',
'mths_since_last_record',
'open_acc',
'pub_rec',
'revol_bal',
'revol_util',
'total_acc',
'fico_range',
'Avg_Median',
'Min_Median',
'Pop',
'Dif_mean_median',
'Median_range',
'Dif_median_from_zip',
'loan_over_income',
'loan_over_median']
'''

num_predictors = ['loan_amnt',
'int_rate',
'installment',
'dti',
'fico_range_high',
'inq_last_6mths',
'mths_since_last_delinq',
'mths_since_last_record',
'open_acc',
'revol_util',
'total_acc',
'Dif_median_from_zip']

df_num = df.get(num_predictors)

# df_num.boxplot(column = 'revol_util')

import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

## Taking care of the NAN values

imputed_features = df_num.median()
imputed_features[['mths_since_last_delinq','mths_since_last_record']] = 0

df_num = df_num.fillna(imputed_features)

## Making it an Array

df_num_array = df_num.values
target

## Split the Data
from sklearn.cross_validation import train_test_split
ar_num_train, ar_num_test, target_train, target_test = train_test_split(df_num_array, target, test_size = 0.70, random_state=0)

ar_num_train.shape
ar_num_test.shape

## Modeling... finally!!
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr2 = LogisticRegression(class_weight={0: 5})
warnings.simplefilter(action = "ignore", category = DeprecationWarning)

lr.fit(ar_num_train, target_train)
target_predicted = lr.predict(ar_num_test)

from sklearn.metrics import accuracy_score

accuracy_score(target_test, target_predicted)
pd.crosstab(target_test, target_predicted, rownames=['True'], colnames=['Predicted'], margins=True)

target_test.mean()







plt.hist(df['Dif_mean_median'])



plt.ylim([-150000, 200000])
plt.scatter(df['target'], df_num['Dif_median_from_zip'])


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