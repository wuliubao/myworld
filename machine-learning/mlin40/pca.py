import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn import decomposition

"""
lib: sklearn
feature: PCA 
"""

# STEP1
#
stats = pd.read_table('dataset/regression.csv')
point = stats.iloc[:,4] / 38
positional_rating = stats.iloc[:,[0,1,2,3]]

# STEP1-BRANCH1-2
#
pca = decomposition.PCA()
pca.fit(positional_rating)

# STEP1-BRANCH1-SHOW
#
print ("The princapal components have variances: \n", pca.explained_variance_)
print ("THe principal componetns have variance ratios: \n", pca.explained_variance_ratio_)

# STEP1-BRANCH1-2
#
pca2 = decomposition.PCA(n_components=2)
pca2.fit(positional_rating)
ratings_new = pca2.transform(positional_rating)

# STEP1-BRANCH2-SHOW
#
plt.scatter(ratings_new[:,0], ratings_new[:,1], c='r', marker='o')
plt.title("PCA with 2 components")
plt.show()
