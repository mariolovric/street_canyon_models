from src import *
from src.configs import model_configurations

def rmse(y_true, y_pred):
    '''

    :param y_true:
    :param y_pred:
    :return: RMSE for y true and y predicted
    '''
    return np.sqrt(mean_squared_error(y_true,y_pred))


class RegressorClass:

    def __init__(self, base_est):
        '''

        :param base_est: base estimator can be: 'random_forest' 'ada_boost' 'pls' 'lasso' 'svr'
        '''
        self.grid_search_ = None
        self.parameters = model_configurations[base_est]['parameters']
        self.base_estimator = model_configurations[base_est]['estimator']
        self.base_est = base_est
        self.best_params = None
        self.new_model = None
        self.best_est = None
        self.train_rmse = None
        self.test_rmse = None
        self.best_score = None
        self.cv_results_ = None

    def func_grid_search(self, X_train, y_train):
        '''

        :param X_train: Train dataframe
        :param y_train: Train target series
        :return:
        '''
        cvx = KFold(n_splits=3, random_state=42, shuffle=False)
        self.grid_search_ = GridSearchCV(estimator=self.base_estimator,
                                         param_grid=self.parameters,
                                         cv=cvx)

        self.grid_search_.fit(X_train, y_train)
        self.best_params = self.grid_search_.best_params_
        self.best_est = self.grid_search_.best_estimator_
        self.best_score = self.grid_search_.best_score_
        self.cv_results_ = self.grid_search_.cv_results_

        print('\n best params', self.best_params)
        return self.best_params

    def train_model(self, X_train, y_train, feat_sel = False, runs = 1):
        '''

        :param X_train:
        :param y_train:
        :param feat_sel: Feature selection True or False
        :param runs: number of iterative runs to select featuress
        :return:
        '''
        params = self.func_grid_search(X_train, y_train)
        self.new_model = self.base_estimator.set_params(**params)
        self.new_model.fit(X_train, y_train)

        if self.base_est == 'lasso' or self.base_est == 'pls':
            pass

        self.features =  X_train.columns.tolist()

        if feat_sel == True:

            for run in range(runs):

                if run==0:
                    self.features = self.feat_selector(X_train, y_train)
                else:
                    self.features = self.feat_selector(X_train[self.features], y_train)


                params = self.func_grid_search(X_train[self.features], y_train)
                self.new_model = self.base_estimator.set_params(**params)
                self.new_model.fit(X_train[self.features], y_train)

    def test_model(self, X_test, y_test, feat_sel = False):
        '''

        :param X_test:
        :param y_test:
        :param feat_sel: Feature selection True or False
        :return:
        '''

        if feat_sel == True:
                X_test = X_test[self.features]

        y_pred_test = self.new_model.predict(X_test)
        self.test_rmse = rmse(y_test, y_pred_test)
        self.mae = mean_absolute_error(y_test, y_pred_test)
        self.r2 = r2_score(y_test, y_pred_test)
        return self.test_rmse, self.mae, self.r2, self.best_score

    def get_predicted(self, X_all, feat_sel = False):
        '''

        :param X_all:
        :param feat_sel:
        :return:
        '''

        if feat_sel == True:
                X_all = X_all[self.features]
        y_full_pred = self.new_model.predict(X_all)
        try:
            return pd.Series(y_full_pred, name='predicted', index=X_all.index)
        except:
            return pd.Series(y_full_pred.squeeze(), name='predicted', index=X_all.index)

    def return_val(self):
        results = {'best_cv_score': self.best_score, 'test_score': self.test_rmse}
        return results



    def feat_selector(self, x,y):
        '''

        :param x:
        :param y:
        :return:
        '''
        perm = PermutationImportance(self.new_model, random_state=42).fit(x,y)
        expl_df = explain_weights_df(estimator=perm, feature_names=x.columns.tolist())

        selected = expl_df[expl_df.weight > 0.01]


        self.explain_df = selected
        return selected.feature.tolist()

    def feat_imp(self, x,y):
        '''

        :param x:
        :param y:
        :return:
        '''
        perm = PermutationImportance(self.new_model, random_state=42).fit(x,y)
        expl_df = explain_weights_df(estimator=perm, feature_names=x.columns.tolist())


        return expl_df




