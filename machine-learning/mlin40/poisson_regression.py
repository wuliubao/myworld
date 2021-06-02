import pandas as pd
import numpy as np
import statsmodels.api as sm

rawstat = pd.read_table('dataset/poisson regression.csv')
offensive = rawstat.iloc[:,[0,1,2,3,4,5]]
goal = rawstat.iloc[:,6]

offensive_add = sm.add_constant(offensive)

# STEP-BRANCH1-2
#
poisson_result = sm.GLM(goal, offensive_add, family=sm.families.Poisson()).fit()

# STEP-BRANCH1-4
print(poisson_result.summary())

# STEP-BRANCH2-2
#
offensive_reduced = offensive.iloc[:,[1,2,5]]
offensive_add = sm.add_constant(offensive_reduced)
poisson_result = sm.GLM(goal, offensive_add, family=sm.families.Poisson()).fit()

# STEP-BRANCH2-4
print(poisson_result.summary())
