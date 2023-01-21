# DONE!
import numpy as np
import pandas as pd
import sklearn
import math
import multiprocessing

from hyperopt import hp, tpe, Trials, fmin, STATUS_OK
from keras.models import load_model, Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import keras
from keras.wrappers.scikit_learn import KerasRegressor
from datetime import datetime

seed = 0
max_evals = 125  # Number of hyperopt iterations

"""ANN model implementation. Can be tuned by Hyperopt."""


class KerasNN:
    def __init__(self, dataset, tune=False, normalization='none', params={}):
        self.dataset = dataset
        self.name = 'ANN'
        self.normalization = normalization

        epoch = 100
        patience = 10
        norm = ['none', 'minmax', 'quantile', 'standardize']
        act = ['relu', 'linear', 'sigmoid']
        nlayer = [1, 2]

        if not tune:
            self.params = params

            # Create necessary folds of dependent variable and regressors
            ytrainfold_finalfit = self.dataset.ytrainfold_finalfit[0]["hb"]  # Train
            ytestfold_finalfit = self.dataset.ytestfold_finalfit[0]["hb"]  # Test
            ytestfoldextra_finalfit = self.dataset.yextratest_finalfold[0]["hb"]  # Additional test

            Xtrainfold_finalfit = self.dataset.Xtrainfold_finalfit[0].drop(["id"], axis=1)
            Xtestfold_finalfit = self.dataset.Xtestfold_finalfit[0].drop(["id"], axis=1)
            Xtestfoldextra_finalfit = self.dataset.Xextratest_finalfold[0].drop(["id"], axis=1)

            # Create ANN final model
            def baseline_model_fin():
                model = Sequential()
                model.add(Dense(params['n_neurons'], input_dim=Xtestfold_finalfit.shape[1],
                                activation=params['activation']))
                model.add(Dropout(params['init_dropout']))
                if params['n_layers'] == 2:
                    model.add(Dense(params['n_neurons'], activation=params['activation']))
                    model.add(Dropout(params['mid_dropout']))
                model.add(Dense(1, activation='relu'))

                opt = keras.optimizers.Adam(learning_rate=params['learning_rate'])
                model.compile(optimizer=opt, metrics=keras.metrics.RootMeanSquaredError(),
                              loss=keras.losses.MeanSquaredError())
                return model

            callbacks = [
                EarlyStopping(monitor='loss', patience=patience, mode='auto', verbose=0),
                ModelCheckpoint(
                    'ANN_model_fin', monitor='loss', mode='auto', save_best_only=True, verbose=0)
            ]

            self.model = KerasRegressor(
                build_fn=baseline_model_fin,
                epochs=epoch,
                batch_size=pow(2, params['batch_size']),
                verbose=0,
                callbacks=callbacks)

            # Fit on training + validation set
            print("Fit final model: {}".format(datetime.now().isoformat(' ', 'seconds')))
            self.model.fit(self.dataset.normalize(Xtrainfold_finalfit, self.normalization), ytrainfold_finalfit,
                           verbose=0, use_multiprocessing=True, workers=multiprocessing.cpu_count())
            del self.model
            self.model = load_model('ANN_model_fin')  # Ensures that you use the early-stopping model

            # Predict on test set (containing observations for individuals we have never seen before)
            print("Predict final model: {}".format(datetime.now().isoformat(' ', 'seconds')))
            yhat = self.model.predict(self.dataset.normalize(Xtestfold_finalfit, self.normalization),
                                      verbose=0, use_multiprocessing=True, workers=multiprocessing.cpu_count())
            yhat = yhat.reshape([-1, ])  # Compatibility with other models
            yhat[yhat < 0] = 0  # Cannot make negative predictions
            self.yhat = yhat
            self.rmse = math.sqrt(sklearn.metrics.mean_squared_error(ytestfold_finalfit, self.yhat))

            # Also test the extra test set with observations for individuals we have seen before.
            print("Predict extra analysis: {}".format(datetime.now().isoformat(' ', 'seconds')))
            yhat_extra = self.model.predict(self.dataset.normalize(Xtestfoldextra_finalfit, self.normalization),
                                            verbose=0, use_multiprocessing=True, workers=multiprocessing.cpu_count())
            yhat_extra = yhat_extra.reshape([-1, ])  # Compatibility with other models
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
                    'n_neurons': int(params['n_neurons']),
                    'learning_rate': float("{:.3f}".format(params['learning_rate'])),
                    'batch_size': int(params['batch_size']),
                    'mid_dropout': float("{:.3f}".format(params['mid_dropout'])),
                    'activation': params['activation'],
                    'n_layers': int(params['n_layers']),
                    'init_dropout': float("{:.3f}".format(params['init_dropout']))
                }

                rmse = []  # Create RMSE list (can be any other metric)
                for i, Xvalidationfolds in enumerate(dataset.Xvalidationfolds):
                    # Create necessary folds of dependent variable and regressors
                    ytrainfold = dataset.ytrainfolds[i]["hb"]  # Train
                    yvalidationfold = dataset.yvalidationfolds[i]["hb"]  # Validation

                    Xtrainfold = dataset.Xtrainfolds[i].drop(["id"], axis=1)
                    Xvalidationfold = dataset.Xvalidationfolds[i].drop(["id"], axis=1)

                    # Create ANN tune model
                    def baseline_model():
                        model = Sequential()
                        model.add(Dense(params['n_neurons'], input_dim=Xtrainfold.shape[1],
                                        activation=params['activation']))
                        model.add(Dropout(params['init_dropout']))
                        if params['n_layers'] == 2:
                            model.add(Dense(params['n_neurons'], activation=params['activation']))
                            model.add(Dropout(params['mid_dropout']))
                        model.add(Dense(1, activation='relu'))

                        opt = keras.optimizers.Adam(learning_rate=params['learning_rate'])
                        model.compile(optimizer=opt, metrics=keras.metrics.RootMeanSquaredError(),
                                      loss=keras.losses.MeanSquaredError())
                        return model

                    callbacks = [
                        EarlyStopping(monitor='loss', patience=patience, mode='auto', verbose=0),
                        ModelCheckpoint(
                            'ANN_model_tune',
                            monitor='loss', mode='auto',
                            save_best_only=True,
                            verbose=0)
                    ]

                    nnet = KerasRegressor(
                        build_fn=baseline_model,
                        epochs=epoch,
                        batch_size=pow(2, params['batch_size']),
                        verbose=0,
                        callbacks=callbacks)

                    # Train on test set
                    print("Fit fold {}: {}".format(i, datetime.now().isoformat(' ', 'seconds')))
                    nnet.fit(self.dataset.normalize(Xtrainfold, normalization), ytrainfold, use_multiprocessing=True,
                             workers=multiprocessing.cpu_count())
                    del nnet
                    nnet = load_model('ANN_model_tune')  # Ensures that you use the early-stopping model

                    # Predict on validation set
                    print("Predict fold {}: {}".format(i, datetime.now().isoformat(' ', 'seconds')))
                    ypred = nnet.predict(self.dataset.normalize(Xvalidationfold, normalization), verbose=0,
                                         use_multiprocessing=True, workers=multiprocessing.cpu_count())
                    yhat = ypred.reshape([-1, ])  # Compatibility with other models
                    yhat[yhat < 0] = 0  # Cannot make negative predictions
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
                'type': 'ANN',
                'n_neurons': hp.quniform('n_neurons', 8, 240, 4),  # 8 is 2/3 the input dim of X
                'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(1)),
                'batch_size': hp.uniform('batch_size', 11, 15),
                'init_dropout': hp.uniform('init_dropout', 0.0, 1.0),  # Dropout in first layer
                'mid_dropout': hp.uniform('mid_dropout', 0.0, 1.0),  # Dropout in second layer
                'activation': hp.choice('activation', act),  # ['relu','linear','sigmoid']
                'n_layers': hp.choice('n_layers', nlayer),  # [1,2]
                'Normalization': hp.choice('Normalization', norm)  # ['none', 'minmax', 'quantile', 'standardize']
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
            best_result['n_neurons'] = int(best_result['n_neurons'])
            best_result['batch_size'] = int(best_result['batch_size'])  # 2 ^ this number
            best_result['activation'] = act[best_result['activation']]
            best_result['n_layers'] = nlayer[best_result['n_layers']]
            self.normalization = norm[best_result['Normalization']]
            print(best_result)
            self.params = best_result
            self.params.pop('Normalization')
