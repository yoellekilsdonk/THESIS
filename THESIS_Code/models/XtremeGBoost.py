from hyperopt import hp, tpe, Trials, fmin, STATUS_OK
import sklearn
import numpy as np
import xgboost as xgb
import math
import pandas as pd
from datetime import datetime
import time

seed = 0
max_evals = 1  # Number of hyperopt iterations

"""XGBoost model implementation. Can be tuned by Hyperopt."""

class XGBoost:
    def __init__(self, dataset, tune=False, normalization='none', params={}, tweedie=False, p=None):
        self.p = p
        self.tweedie = tweedie
        self.dataset = dataset
        if tweedie:
            self.name = 'XGBoostTweedie'
        else:
            self.name = 'XGBoost'
        self.normalization = normalization

        norm=['none', 'standardize']

        def tweedie_eval(y_true, y_pred):
            def a(y_pred):
                p = self.p
                return np.math.pow(y_pred, (1 - p)) / (1 - p)
            def b(y_pred):
                p = self.p
                return np.math.pow(y_pred, (2 - p)) / (2 - p)  # np.exp(np.log(y_pred) * (2 - p)) / (2 - p)

            part1 = list(map(a, y_pred))
            part2 = list(map(b, y_pred))

            loss = sum(-np.multiply(y_true.values, part1) + part2)
            return loss

        if not tune: # Activated once tuning is done
            self.params = params

            # Create necessary folds of dependent variable and regressors
            ytrainfold_finalfit = self.dataset.ytrainfold_finalfit[0]["hb"]  # Train
            ytestfold_finalfit = self.dataset.ytestfold_finalfit[0]["hb"]  # Test
            ytestfoldextra_finalfit = self.dataset.yextratest_finalfold[0]["hb"]  # Validation

            Xtrainfold_finalfit = self.dataset.Xtrainfold_finalfit[0].drop(["id"], axis=1)
            Xtestfold_finalfit = self.dataset.Xtestfold_finalfit[0].drop(["id"], axis=1)
            Xtestfoldextra_finalfit = self.dataset.Xextratest_finalfold[0].drop(["id"], axis=1)

            # Create XGBoost model
            if self.tweedie == True:
                tweediestring = 'tweedie-nloglik@' + str(self.p)
                self.model = xgb.XGBRegressor(objective='reg:tweedie',
                                              tweedie_variance_power=self.p, n_jobs=-1, random_state=self.dataset.seed, **params, tree_method='auto',
                                              use_label_encoder=False, eval_metric=['rmse', tweediestring])
            else:
                self.model = xgb.XGBRegressor(n_jobs=-1, random_state=self.dataset.seed, **params, tree_method='auto',
                                              use_label_encoder=False, eval_metric='rmse')  # XGBoost model with Root Mean Squared Error loss

            # Fit on training + validation set
            print("Fit final model: {}".format(datetime.now().isoformat(' ', 'seconds')))
            self.model.fit(self.dataset.normalize(Xtrainfold_finalfit, self.normalization), ytrainfold_finalfit)  # Fit model on compelete trainfold

            # Predict on test set (containing observations for individuals we have never seen before)
            print("Predict final model: {}".format(datetime.now().isoformat(' ', 'seconds')))
            yhat = self.model.predict(self.dataset.normalize(Xtestfold_finalfit, self.normalization))
            yhat[yhat < 0] = 0
            self.yhat = yhat
            self.rmse = math.sqrt(sklearn.metrics.mean_squared_error(ytestfold_finalfit, self.yhat))
            self.tweedielist = tweedie_eval(ytestfold_finalfit, self.yhat)

            # Predict the extra test set (containing observations for individuals we have seen before)
            print("Predict extra analysis: {}".format(datetime.now().isoformat(' ', 'seconds')))
            yhat_extra = self.model.predict(self.dataset.normalize(Xtestfoldextra_finalfit, self.normalization))
            yhat_extra[yhat_extra < 0] = 0  # Cannot make negative predictions
            self.yhat_extra = yhat_extra
            self.rmse_extra = math.sqrt(sklearn.metrics.mean_squared_error(ytestfoldextra_finalfit, self.yhat_extra))
            self.tweedielist_extra = tweedie_eval(ytestfoldextra_finalfit, self.yhat_extra)

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
                tweedielist = []
                for i, Xvalidationfolds in enumerate(dataset.Xvalidationfolds):
                    # Create necessary folds of dependent variable and regressors
                    ytrainfold = dataset.ytrainfolds[i]["hb"] # Train
                    yvalidationfold = dataset.yvalidationfolds[i]["hb"] # Validation

                    Xtrainfold = dataset.Xtrainfolds[i].drop(["id"], axis=1)
                    Xvalidationfold = dataset.Xvalidationfolds[i].drop(["id"], axis=1)

                    # Create XGBoost tune model
                    if self.tweedie == True:
                        tweediestring = 'tweedie-nloglik@' + str(self.p)
                        xreg = xgb.XGBRegressor(objective ='reg:tweedie',
                                                tweedie_variance_power=self.p, n_jobs=-1,
                                                random_state=self.dataset.seed, **params, tree_method='auto',
                                                use_label_encoder=False, eval_metric=['rmse',tweediestring])
                    else:
                        xreg = xgb.XGBRegressor(n_jobs=-1, random_state=self.dataset.seed, **params, tree_method='auto',
                                                use_label_encoder=False, eval_metric='rmse')

                    # Train on test set
                    print("Fit fold {}: {}".format(i, datetime.now().isoformat(' ', 'seconds')))
                    xreg.fit(self.dataset.normalize(Xtrainfold, normalization), ytrainfold,
                             eval_set=[(self.dataset.normalize(Xtrainfold, normalization), ytrainfold)],
                             verbose = False)
                    # Predict on validation set
                    print("Predict fold {}: {}".format(i, datetime.now().isoformat(' ', 'seconds')))
                    yhat = xreg.predict(self.dataset.normalize(Xvalidationfold, normalization))
                    yhat[yhat < 0] = 0  # Cannot make negative predictions
                    # del xreg
                    rmse.append(sklearn.metrics.mean_squared_error(yvalidationfold, yhat))
                    print("Done fold {}: {}".format(i, datetime.now().isoformat(' ', 'seconds')))

                    # Save parameters, rmse, and summary statistics of current hyperopt run
                    self.overview[str(k)][str(i)]={}
                    self.overview[str(k)][str(i)]['params'] = params
                    self.overview[str(k)][str(i)]['rmse']= sklearn.metrics.mean_squared_error(yvalidationfold, yhat)
                    self.overview[str(k)][str(i)]['yhat']= pd.DataFrame(yhat).describe()

                    if self.tweedie == True:
                        tweedielist.append(tweedie_eval(yvalidationfold, yhat))
                        self.overview[str(k)][str(i)]['tweedie']= tweedie_eval(yvalidationfold, yhat)
                        losslist = tweedielist
                    else:
                        losslist = rmse
                return {'loss': np.mean(losslist),'status': STATUS_OK} # Cross validation to find average performance
                # across multiple evaluations

            # Define search space for hyperopt
            search_space = {
                'type': 'XGBoost',
                'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(1)),
                'n_estimators': hp.quniform('n_estimators', 50, 200, 10),
                'max_depth': hp.quniform('max_depth', 2, 6, 1),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0), # What percentage of features to be used
                # when building tree
                'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0), # L2 regularization term on weights
                'subsample': hp.uniform('subsample', 0.5, 1.0), # Randomly sample subsample% of training data prior to
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
            best_result['max_depth']=int(best_result['max_depth'])
            best_result['n_estimators']=int(best_result['n_estimators'])
            self.normalization = norm[best_result['Normalization']]
            print(best_result)
            self.params = best_result
            self.params.pop('Normalization')

            if self.tweedie == True:
                self.model = xgb.XGBRegressor(objective ='reg:tweedie',
                                              tweedie_variance_power=self.p, n_jobs=-1,
                                              random_state=self.dataset.seed, **best_result,
                                              tree_method='auto',
                                              use_label_encoder=False, eval_metric='rmse')
            else:
                self.model = xgb.XGBRegressor(n_jobs=-1, random_state=self.dataset.seed, **best_result,
                                              tree_method='auto',
                                              use_label_encoder=False, eval_metric='rmse')
