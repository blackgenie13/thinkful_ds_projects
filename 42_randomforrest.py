# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 16:51:10 2015
@author: sroy
Unfortunately the data does not match with the prescribed link
Please use this location to download the required data:
https://github.com/nborwankar/LearnDataScience/tree/master/datasets/samsung
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as skm
import pylab as pl

### Added to avoid Deprecation Warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


samtrain = pd.read_csv('C:/Users/Michael Lin_2/Desktop/Thinkful/Data/samsung-accelerometer/samtrain.csv')
samval = pd.read_csv('C:/Users/Michael Lin_2/Desktop/Thinkful/Data/samsung-accelerometer/samval.csv')
samtest = pd.read_csv('C:/Users/Michael Lin_2/Desktop/Thinkful/Data/samsung-accelerometer/samtest.csv')
samsungdata = pd.read_csv('C:/Users/Michael Lin_2/Desktop/Thinkful/Data/samsung-accelerometer/samsungdata.csv')
samsungmin = pd.read_csv('C:/Users/Michael Lin_2/Desktop/Thinkful/Data/samsung-accelerometer/samsungmin.csv')


# We use the Python RandomForest package from the scikits.learn collection of algorithms. 

# For this we need to convert the target column ('activity') to integer values 
# because the Python RandomForest package requires that.  

# We map activity to an integer according to
# laying = 1, sitting = 2, standing = 3, walk = 4, walkup = 5, walkdown = 6
# Code also uses library randomforest.py (uploaded in git)

def remap_col(df,colname, mapping = None):
  """'mapping' is a dict sending old col elements to new ones, perhaps changing data types even.
      very useful for mapping factor columns from R to integer columns for Python
  """
  map_dict = {'laying':1, 'sitting':2, 'standing':3, 'walk':4, 'walkup':5, 'walkdown':6} 
  if not mapping:
    mapping = map_dict.copy()
    
  df[colname] = df[colname].map(lambda x: mapping[x]) 
  return df

samtrain = remap_col(samtrain,'activity')
samval = remap_col(samval,'activity')
samtest = remap_col(samtest,'activity')


rfc = RandomForestClassifier(n_estimators=500, oob_score=True)
train_data = samtrain[samtrain.columns[1:-2]]
train_target = samtrain['activity']
model = rfc.fit(train_data, train_target)

# use the OOB (out of band) score which is an estimate of accuracy of our model.
rfc.oob_score_
#model.oob_score_

# use "feature importance" scores to see what the top 10 important features are
## Change the value 0.04 which we picked empirically to give us 10 variables
fi = enumerate(rfc.feature_importances_)
cols = samtrain.columns
[(value,cols[i]) for (i,value) in fi if value > 0.04]

# pandas data frame adds a spurious unknown column in 0 position hence starting at col 1
# not using subject column, activity ie target is in last columns hence -2 i.e dropping last 2 cols
val_data = samval[samval.columns[1:-2]]
val_target = samval['activity']
val_pred = rfc.predict(val_data)
#val_pred = model.predict(val_data)

test_data = samtest[samtest.columns[1:-2]]
test_target = samtest['activity']
test_pred = rfc.predict(test_data)
#test_pred = model.predct(test_data)

print("mean accuracy score for validation set = %f" %(rfc.score(val_data, val_target)))
print("mean accuracy score for test set = %f" %(rfc.score(test_data, test_target)))

# use the confusion matrix to see how observations were misclassified as other activities and visualize it
test_cm = skm.confusion_matrix(test_target,test_pred)


pl.matshow(test_cm)
pl.title('Confusion matrix for test data')
pl.colorbar()
pl.show()

# compute a number of other common measures of prediction goodness
# Accuracy
print("Accuracy = %f" %(skm.accuracy_score(test_target,test_pred)))
# Precision
print("Precision = %f" %(skm.precision_score(test_target,test_pred)))
# Recall
print("Recall = %f" %(skm.recall_score(test_target,test_pred)))
# F1 Score
print("F1 score = %f" %(skm.f1_score(test_target,test_pred)))