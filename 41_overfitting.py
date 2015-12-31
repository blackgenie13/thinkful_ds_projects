# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 19:07:20 2015

@author: Michael Lin_2
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Set seed for reproducible results
np.random.seed(414)

# Gen toy data
X = np.linspace(0, 15, 1000)
y = 3 * np.sin(X) + np.random.normal(1 + X, .2, 1000)

train_X, train_y = X[:700], y[:700]
test_X, test_y = X[700:], y[700:]

train_df = pd.DataFrame({'X': train_X, 'y': train_y})
test_df = pd.DataFrame({'X': test_X, 'y': test_y})

# Linear Fit
poly_1 = smf.ols(formula='y ~ 1 + X', data=train_df).fit()
coeff1 = poly_1.params
poly_1.summary()
print("Linear Fit has a R-Squared of {0}" .format(poly_1.rsquared))

# Quadratic Fit
poly_2 = smf.ols(formula='y ~ 1 + X + I(X**2)', data=train_df).fit()
coeff2 = poly_2.params
poly_2.summary()
print("Quadratic Fit has a R-Squared of {0}" .format(poly_2.rsquared))

# 
poly_3 = smf.ols(formula='y ~ 1 + X + I(X**2) + I(X**3)', data=train_df).fit()
coeff3 = poly_3.params
poly_3.summary()
print("To the 3rd Fit has a R-Squared of {0}" .format(poly_3.rsquared))

# 
poly_4 = smf.ols(formula='y ~ 1 + X + I(X**2) + I(X**3) + I(X**4)', data=train_df).fit()
coeff4 = poly_4.params
poly_4.summary()
print("To the 4th Fit has a R-Squared of {0}" .format(poly_4.rsquared))