# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 17:39:38 2016

@author: Michael Lin_2
"""

#display the breakout
df.groupby(['grade'])['target'].mean()
df.groupby(['sub_grade']).target.count()


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


'''
QUESTIONS?????????
- What is the appropriate way to eliminate predictors?  This is difficult to determine given that sklearn doesn't 
  offer p-value for each coefficients
- How can I check for interactions among predictors?
- Should we consider writing model for only Grade A, Grade B, Grade C?????  How can I determine whether this is worth
  the effort to explroe?  (How to know different grades actually behave differently?)


















###############################################################################

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
