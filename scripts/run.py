from src.preprocessing import *
from src import *
from src.models import RegressorClass

X = process()

#dictionary which is mapping targets and predictive variables
Xvar_dict = {'PM10 ug/m3':['NO2 ug/m3','NO2 ug/m3_lag','PM10 ug/m3_lag'],
           'NO2 ug/m3':['NO2 ug/m3_lag','EC ug/m3','EC ug/m3_lag','OC ug/m3','OC ug/m3_lag', 'PM10 ug/m3', 'PM10 ug/m3_lag'],
           'EC ug/m3':['NO2 ug/m3','NO2 ug/m3_lag','PM10 ug/m3','PM10 ug/m3_lag', 'EC ug/m3_lag'],
           'OC ug/m3':['NO2 ug/m3','NO2 ug/m3_lag','PM10 ug/m3','PM10 ug/m3_lag', 'OC ug/m3_lag']}

#general/all variables
general_variables = ['Precipitation', 'Pressure average', 'Pressure max', 'Relative humidity average',
                     'Relative humidity max', 'Relative humidity max-min', 'Relative humidity min', 'Season',
                     'Temperature average', 'Temperature max', 'Temperature max-min', 'Temperature min', 'Weekday', 'Weekday_lag',
                     'Wind direction', 'Wind speed', 'Year', 'Temperature max_lag', 'Temperature min_lag',
                     'Temperature average_lag', 'Temperature max-min_lag', 'Pressure max_lag', 'Pressure average_lag',
                     'Relative humidity max_lag', 'Relative humidity min_lag', 'Relative humidity average_lag',
                     'Relative humidity max-min_lag', 'Wind speed_lag', 'Precipitation_lag', 'Wind direction_lag']



results_all_experiments = pd.DataFrame()
perm_imp_all_experiments = pd.DataFrame()
for variable in Xvar_dict.keys():  #
    print('\n=========== %s ============== \n' % variable)

    version = 'model'

    # copy the matrix
    X_pre = X.copy()
    # fill media
    X_pre = X_pre.fillna(X_pre.median())

    # choose y, set X for modeling
    y_pre = X_pre[variable]
    X_matrix = X_pre[Xvar_dict[variable] + general_variables]


    # select common indices, i.e. exclude data not present in both X and y
    common_indices = y_pre.index.intersection(X_matrix.index)
    #loc to common indices
    X_matrix = X_matrix.loc[common_indices]
    y = y_pre.loc[common_indices]

    ## ======== X and y are set ========
    X_train, y_train, = X_matrix[X_matrix.index < '2013'], y[y.index < '2013']
    X_test, y_test = X_matrix[X_matrix.index > '2013'], y[y.index > '2013']
    y_test.to_csv('../results/test_%s.csv' %variable.replace('/', '_'))

    results_collection_df = pd.DataFrame([])
    perm_imp_collection_df = pd.DataFrame([])
    #iterates through regressors
    for regressor in ['lasso', 'pls', 'ada_boost', 'random_forest', 'svr']:

        #iterates through feature selection as true false
        for featsel in [False, True]:


            # training, insert regressor to instance
            trainer = RegressorClass(base_est=regressor)
            # inserts feature selector to instance
            trainer.train_model(X_train, y_train, feat_sel=featsel, runs=1)
            rmse_, mae_,r2_score_, bscore_ =  trainer.test_model(X_test, y_test, feat_sel=featsel)


            experiment_results = {
                'rmse':round(rmse_, 2),
                'mae':round(mae_, 2),
                'r2':round(r2_score_, 2),
                'test mean':round(y_test.mean(), 2),
                'cv_res':round(bscore_, 2)}

            #store results as pandas series
            var_coll = pd.Series(experiment_results, name =str(variable) + '|' + str(regressor + '|' + str(featsel)))
            #concat to pandas dataframe per experiment
            results_collection_df = pd.concat([results_collection_df, var_coll.T], axis=1)

            #permutation importnace
            permutation_importance_results = trainer.feat_imp(X_train[trainer.features], y_train)
            permutation_importance_results.index = permutation_importance_results.feature
            permutation_importance_results.drop('feature', axis=1, inplace=True)
            permutation_importance_results.columns = [str(i) + '|' + str(variable) + '|' + str(regressor) + '|' + str(featsel) for i in permutation_importance_results.columns]
            perm_imp_collection_df = pd.concat([perm_imp_collection_df, permutation_importance_results], axis=1)

            #save predictions
            y_predicted = trainer.get_predicted(X_test, feat_sel=featsel)
            y_predicted.name = str(variable).replace('/', '_') + '_' + str(regressor) + '_' + str(featsel)
            y_predicted.to_csv('../results/result_predictions_for_variable_%s.csv' % y_predicted.name)


    #collects results from the predictions
    results_all_experiments = pd.concat([results_all_experiments, results_collection_df], axis=1)
    #collects permutation importances
    perm_imp_all_experiments = pd.concat([perm_imp_all_experiments, perm_imp_collection_df], axis=1)

#store results
results_all_experiments.to_csv('../results/all_results.csv')
perm_imp_all_experiments.to_csv('../results/all_permutation_importances.csv')


