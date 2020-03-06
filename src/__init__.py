from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import LinearSVR

import numpy as np
import pandas as pd
import datetime

from eli5.sklearn import PermutationImportance
from eli5 import explain_weights_df


