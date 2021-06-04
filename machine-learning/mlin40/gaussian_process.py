import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF

"""
lib: sklearn
model: gaussian process
"""

kernel = 1e-2 * RBF([5e-3])

## STEP-BRANCH1-1
#
#Gaussian Process Regression with regression dataset
stats = pd.read_table('dataset/regression.csv')
point = stats.iloc[:,4] / 38
rating = stats.iloc[:,5].reshape(-1, 1)

x_min, x_max = rating.min() - 0.05, rating.max() + 0.05
xx_regression = np.arange(x_min, x_max, 1e-3).reshape(-1, 1)

## STEP-BRANCH1-2
gpr = gaussian_process.GaussianProcessRegressor(kernel=kernel).fit(rating, point)
gpr_pred, sigma = gpr.predict(xx_regression, return_std=True)

## STEP-BRANCH2-1
#Gaussian Process Classification with linear inseparable dataset
ratio = []

rawstat = pd.read_table('dataset/linear inseparable.csv')
category = rawstat.iloc[:,0]
pass_ratio = rawstat.iloc[:,1] / rawstat.iloc[:,2]
shot_ratio = rawstat.iloc[:,3] / rawstat.iloc[:,4]

ratio.append(pass_ratio)
ratio.append(shot_ratio)
ratio = np.array(ratio)
ratio = ratio.astype('float')

## STEP-BRANCH2-2
#
gpc = gaussian_process.GaussianProcessClassifier(kernel=kernel).fit(ratio.T, category)

x_min, x_max = ratio[0].min() - 0.05, ratio[0].max() + 0.05
y_min, y_max = ratio[1].min() - 0.05, ratio[1].max() + 0.05
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
xy = np.vstack([xx.ravel(), yy.ravel()]).T
Z = gpc.predict(xy).reshape(xx.shape)

## STEP4
#result demonstration
fig = plt.figure()

ax1 = fig.add_subplot(121)  
plt.plot(xx_regression, gpr_pred, c="r", linewidth=4)
plt.scatter(rating, point, c="b", s=5)
plt.fill(np.concatenate([xx_regression, xx_regression[::-1]]),
         np.concatenate([gpr_pred - 1.96 * sigma, (gpr_pred + 1.96 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None')
plt.title("Gaussian Process Regression")
plt.xlabel("average rating")
plt.ylabel("average point per game")
  
ax2 = fig.add_subplot(122)  
plt.scatter(pass_ratio[category == 0], shot_ratio[category == 0], c='r', marker = 'o')
plt.scatter(pass_ratio[category == 1], shot_ratio[category == 1], c='b', marker = '^')
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
plt.title("Gaussian Process Classification")
plt.xlabel('pass ratio')
plt.ylabel('shot ratio')

plt.show()
