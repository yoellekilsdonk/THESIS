import merfadjusted
import sklearn
import numpy as np
import xgboost as xgb
import math
import pandas as pd
from hyperopt import hp, tpe, Trials, fmin, STATUS_OK
from datetime import datetime

seed = 0
max_evals = 125

"""XGBoost model implementation. Can be tuned by Hyperopt."""

class MeXGBoost:
    def __init__(self, dataset, tune=False, normalization='none', params={}):
        self.dataset = dataset
        self.name = 'MeXGBoost'
        self.normalization = normalization

        norm = ['none', 'minmax', 'quantile', 'standardize']

        if not tune:
            self.params = params

            # Create necessary folds of dependent variable, fixed-effect regressors, random-effect regressors, and
            # clusters
            ytrainfold_finalfit = self.dataset.ytrainfold_finalfit[0]["hb"]  # Train
            ytestfold_finalfit = self.dataset.ytestfold_finalfit[0]["hb"]  # Test
            ytestfoldextra_finalfit = self.dataset.yextratest_finalfold[0]["hb"]  # Validation

            Xtrainfold_finalfit = self.dataset.Xtrainfold_finalfit[0].drop(["id"], axis=1)
            Xtestfold_finalfit = self.dataset.Xtestfold_finalfit[0].drop(["id"], axis=1)
            Xtestfoldextra_finalfit = self.dataset.Xextratest_finalfold[0].drop(["id"], axis=1)

            Ztrainfinalfit = (pd.DataFrame(np.ones(np.size(Xtrainfold_finalfit, 0))))
            Ztestfinalfit = (pd.DataFrame(np.ones(np.size(Xtestfold_finalfit, 0))))
            Ztestextrafinalfit = (pd.DataFrame(np.ones(np.size(Xtestfoldextra_finalfit, 0))))

            clustertrainfinalfit = dataset.Xtrainfold_finalfit[0][['id']].squeeze()
            clustertestfinalfit = dataset.Xtestfold_finalfit[0][['id']].squeeze()
            clustertestextrafinalfit = dataset.Xextratest_finalfold[0][['id']].squeeze()

            # Create mixed-effects XGBoost model
            xgbmodel = xgb.XGBRegressor(n_jobs=-1, random_state=self.dataset.seed, **params, tree_method='auto',
                                        use_label_encoder=False, eval_metric='rmse')
            self.model = merfadjusted.MERF(xgbmodel, max_iterations=5)

            # Fit on training + validation set
            print("Fit final model: {}".format(datetime.now().isoformat(' ', 'seconds')))
            self.model.fit(self.dataset.normalize(Xtrainfold_finalfit, self.normalization), Ztrainfinalfit,
                           clustertrainfinalfit, ytrainfold_finalfit)

            # Predict on test set (containing observations for individuals we have never seen before)
            print("Predict final model: {}".format(datetime.now().isoformat(' ', 'seconds')))
            yhat = self.model.predict(self.dataset.normalize(Xtestfold_finalfit, self.normalization),
                                      Ztestfinalfit, clustertestfinalfit)
            yhat[yhat < 0] = 0
            self.yhat = yhat
            self.rmse = math.sqrt(sklearn.metrics.mean_squared_error(ytestfold_finalfit, self.yhat))

            # Predict the extra test set (containing observations for individuals we have seen before)
            yhat_extra = self.model.predict(self.dataset.normalize(Xtestfoldextra_finalfit, self.normalization),
                                            Ztestextrafinalfit, clustertestextrafinalfit)
            yhat_extra[yhat_extra < 0] = 0  # Cannot make negative predictions
            self.yhat_extra = yhat_extra
            self.rmse_extra = math.sqrt(sklearn.metrics.mean_squared_error(ytestfoldextra_finalfit, self.yhat_extra))

        else:
            self.overview = {}

            def objective(params):
                k = spark_trials.tids[-1] + 1  # Iteration of hyperopt
                self.overview[str(k)] = {}  # Create a dictionary of parameters per hyperopt iteration

                # Retreive parameters
                normalization = params['Normalization']
                params = {
                    'learning_rate': float("{:.3f}".format(params['learning_rate'])),
                    'n_estimators': int(params['n_estimators']),
                    'max_depth': int(params['max_depth']),
                    'reg_lambda': float("{:.3f}".format(params['reg_lambda'])),
                    'colsample_bytree': float("{:.3f}".format(params['colsample_bytree'])),
                    'subsample': float("{:.3f}".format(params['subsample']))
                }

                rmse = []  # Create precision list (can be any other metric)
                for i in range(0, 1):
                    # Create necessary folds of dependent variable, fixed-effect regressors, random-effect regressors,
                    # and clusters
                    ytrainfold = dataset.ytrainfolds[i]["hb"]  # Train
                    yvalidationfold = dataset.yvalidationfolds[i]["hb"]  # Validation

                    Xtrainfold = dataset.Xtrainfolds[i].drop(["id"], axis=1)
                    Xvalidationfold = dataset.Xvalidationfolds[i].drop(["id"], axis=1)

                    Ztrainfold = pd.DataFrame(np.ones(np.size(Xtrainfold, 0)))
                    Zvalidation = (pd.DataFrame(np.ones(np.size(Xvalidationfold, 0))))

                    clustervalidationfold = dataset.Xvalidationfolds[i][['id']].squeeze()
                    clustertrainfold = dataset.Xtrainfolds[i][['id']].squeeze()

                    # Create mixed-effects XGBoost tune model
                    xreg = xgb.XGBRegressor(n_jobs=-1, random_state=self.dataset.seed, **params, tree_method='auto',
                                            use_label_encoder=False, eval_metric='rmse')

                    # Train on test set
                    print("Fit fold {}: {}".format(i, datetime.now().isoformat(' ', 'seconds')))
                    meXGB = merfadjusted.MERF(xreg, max_iterations=5)
                    del xreg
                    meXGB.fit(self.dataset.normalize(Xtrainfold, normalization), Ztrainfold, clustertrainfold,
                              ytrainfold)

                    # Predict on validation set
                    print("Predict fold {}: {}".format(i, datetime.now().isoformat(' ', 'seconds')))
                    yhat = meXGB.predict(self.dataset.normalize(Xvalidationfold, normalization),
                                         Zvalidation, clustervalidationfold)
                    yhat[yhat < 0] = 0  # Cannot make negative predictions
                    del meXGB
                    rmse.append(sklearn.metrics.mean_squared_error(yvalidationfold, yhat))
                    print("Done fold {}: {}".format(i, datetime.now().isoformat(' ', 'seconds')))

                    # Save parameters, rmse, and summary statistics of current hyperopt run
                    self.overview[str(k)][str(i)] = {}
                    self.overview[str(k)][str(i)]['params'] = params
                    self.overview[str(k)][str(i)]['rmse'] = sklearn.metrics.mean_squared_error(yvalidationfold, yhat)
                    self.overview[str(k)][str(i)]['yhat'] = pd.DataFrame(yhat).describe()
                return {'loss': np.mean(rmse), 'status': STATUS_OK}

            # Define search space for hyperopt
            search_space = {
                'type': 'XGBoost',
                'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(1)),
                'n_estimators': hp.quniform('n_estimators', 50, 200, 10),
                'max_depth': hp.quniform('max_depth', 2, 6, 1),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),  # What percentage of features to be used
                # when building tree
                'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),  # L2 regularization term on weights
                'subsample': hp.uniform('subsample', 0.5, 1.0),  # Randomly sample subsample% of training data prior to
                # growing trees
                'Normalization': hp.choice('Normalization', norm)
            }

            # Start hyperopt
            algo = tpe.suggest
            spark_trials = Trials()
            rstate = np.random.default_rng(self.dataset.seed)  # Use any number here but fixed

            best_result = fmin(
                fn=objective,
                space=search_space,
                algo=algo,
                max_evals=max_evals,
                trials=spark_trials,
                rstate=rstate)

            # Save and print best hyperparameter results
            best_result['max_depth'] = int(best_result['max_depth'])
            best_result['n_estimators'] = int(best_result['n_estimators'])
            self.normalization = norm[best_result['Normalization']]
            print(best_result)
            self.params = best_result
            self.params.pop('Normalization')
            self.model = xgb.XGBRegressor(n_jobs=-1, random_state=self.dataset.seed, **best_result, tree_method='auto',
                                          use_label_encoder=False, eval_metric='rmse')
