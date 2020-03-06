from src import *

model_configurations = {
     'random_forest': {
         'estimator': RandomForestRegressor(),
         'parameters': {
             'bootstrap': [True],
             'max_depth': [15, 20, 25, 30],
             'max_features': ['auto'],
             'min_samples_split': [2,3,4],
             'n_estimators': [50, 100, 150],
             'max_samples':[.25,.33,.5],
             'random_state':[42],
             'n_jobs': [-1]
         }
     },
     'ada_boost': {
         'estimator': AdaBoostRegressor(),
         'parameters': {
             'base_estimator': [DecisionTreeRegressor(max_depth=4),
                                DecisionTreeRegressor(max_depth=5),
                                ],
             'n_estimators': [1000],
             'learning_rate': [0.01, 0.1, 1],
             'loss': ['linear', 'square', 'exponential'],
             'random_state':[42],
         }
     }
    ,

    'pls': {
        'estimator': PLSRegression(),
        'parameters': {
            'n_components': [2, 3, 4],
            'max_iter': [200, 500, 1000],
            'scale': [True]

        }
    }
    ,
    'lasso': {
        'estimator': Lasso(),
        'parameters': {
            'normalize': [True, False],
            'max_iter': [200, 500, 1000],
            'alpha': [.1, .5, 1, 100, 500, 1000, 5000, 10000],
            'fit_intercept':[True, False],
            'random_state':[42],

        }
    }
    ,
    'svr': {
        'estimator': LinearSVR(),
        'parameters': {
            'epsilon': [0, .01, .1, .5, 1, 10],
            'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
            'C': [.01, .1, 1, 10, 100, 1000, 10000],
            'fit_intercept':[True, False],
            'dual':[True, False],
            'intercept_scaling':[.01, .1,  1, 10],
            'random_state':[42],

        }
    }
 }