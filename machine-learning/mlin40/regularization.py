import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn import linear_model,metrics

"""
lib: sklearn
model: linear_regression
opt: 1.LASSO 2.ridge
"""

# STEP1
#
stats = pd.read_table('dataset/regression.csv')
point = stats.iloc[:,4] / 38
positional_rating = stats.iloc[:,[0,1,2,3]]

# STEP2
#
linear = linear_model.LinearRegression()
linear.fit(positional_rating, point)

# STEP-BRANCH1 4
#
print("The coefficients of linear regression is: \n", linear.coef_, linear.intercept_)
linear_pred = linear.predict(positional_rating)
linear_error = metrics.mean_squared_error(point, linear_pred)

# STEP-BRANCH2 3
#
lasso = linear_model.Lasso(alpha = 0.05)
lasso.fit(positional_rating, point)

# STEP-BRANCH2 4
#
print("The coefficients of LASSO is: \n", lasso.coef_, lasso.intercept_)
lasso_pred = lasso.predict(positional_rating)
lasso_error = metrics.mean_squared_error(point, lasso_pred)

# STEP-BRANCH3 3
#
ridge = linear_model.Ridge(alpha = 0.05)
ridge.fit(positional_rating, point)

# STEP-BRANCH3 4
#
print("The coefficients of ridge regression is: \n", ridge.coef_, ridge.intercept_)
ridge_pred = ridge.predict(positional_rating)
ridge_error = metrics.mean_squared_error(point, ridge_pred)

# STEP-BRANCH-ALL 4
# plt
plt.scatter(range(len(point)), linear_pred, c="b", s=5, label = "RMSE = {}".format(linear_error))
plt.scatter(range(len(point)), lasso_pred, c="g", s=5, label = "RMSE = {}".format(lasso_error))
plt.scatter(range(len(point)), ridge_pred, c="r", s=5, label = "RMSE = {}".format(ridge_error))
plt.legend()
plt.title("Multivariate Linear Regression with Regularization")
plt.show()
