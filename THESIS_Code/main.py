import pickle
import os
import time

import DataClass as DC
import utils
from models import XtremeGBoost
from models import ANN
from models import MeANN
from models import MeXtremeGBoost
from datetime import datetime
import warnings
from keras.models import model_from_json

if __name__ == '__main__':
    time1 = time.perf_counter()  # Initiate counter to track time between step 0 and step 1
    print("STEP 0 -- Starting main: {}".format(datetime.now().isoformat(' ', 'seconds')))
    time_total = time.perf_counter()  # Initiate counter to track total time of program

    # Ignore certain set of warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # The following parameters determine what part of the code is ran, with iteration in range(tune, final) = (0,3)
    # we first tune the models, then predict on the test set, and then perform analysis to evaluate the final model
    # Specifically, if iteration = 0 we tune, if iteration = 1 we predict and if iteration = 2 we evaluate
    tune = 0
    final = 1

    path = "V:/UserData/079915/Thesis -- Data"  # Change this path to where you've stored the data
    nr_folds = 4  # Number of folds for k-fold cross validation
    normalize = False  # Change if you'd rather include a normalization scheme from the get go in DataClass
    normalize_scheme = "none"  # Change if you'd rather include a normalization scheme from the get go in DataClass

    currentrun = "MeANN"  # Specify which method to run or set to "all" to run all models
    tweedie = True  # Set to true if you wish to use tweedie loss function as objective (only compatible with XGBoost)
    tweedie_rho = 1.6

    computational_device = False
    path_comp_dev = 'VM/XGBoost--n_est200max_depth6/'

    if (tune != 0):
        warnings.warn('You are trying to run a final model/extra analysis without tuning the model first.'
                      'Ensure that the right parameters are in place, o.w. the code will fail to run.')
    if ((tune > 2) & ((currentrun == "ANN")|(currentrun == "MeANN"))):
        raise Exception("The (Me)ANN model requires that the final model has been ran on the current computational device."
                        "Set tune equal to 1 or 2 and restart main.")
    if (tune>=final):
        raise Exception("'Tune' must always be strictly smaller than 'final', please reset parameters accordingly.")

    print("STEP 1 -- Load and prepare data: {} (+{} min)".format(datetime.now().isoformat(' ', 'seconds'),
                                                                 round((time.perf_counter() - time1) / 30) / 2))
    time1 = time.perf_counter()

    # Write pickle file to save time while programming
    dataset = DC.DataClass(path=path, resample=False, normalization=normalize,normalization_scheme = normalize_scheme, folds=nr_folds)  # Load data set
    filename = 'df'
    outfile = open(filename,'wb')
    pickle.dump(dataset, outfile)
    outfile.close()

    infile = open('df', 'rb')
    dataset = pickle.load(infile)
    infile.close()

    print("STEP 2 -- Tune and fit models: {} (+{} min)".format(datetime.now().isoformat(' ', 'seconds'),
                                                               round((time.perf_counter() - time1) / 30) / 2))
    time1 = time.perf_counter()
    evaluation = {}  # Dictionary containing all performance measures per model.
    optimalparams = {}  # Dictionary containing all optimal parameters per model.
    predictions = {}  # Dictionary containing all predictions per model for every dataset.

    print("This is running {}".format(currentrun))
    if currentrun == "XGBoost" or currentrun == "all":
        for iteration in range(tune, final):
            if iteration == 0:  # Tune
                print("STEP 2.1 -- Start tuning XGBoost: {} (+{} min)".format(datetime.now().isoformat(' ', 'seconds'),
                                                                              round((time.perf_counter() - time1) / 30) / 2))
                time1 = time.perf_counter()

                # If tweedie == True, the tweedie loss function is used as objective in XGBoost and in hyperopt
                # else, RMSE is used
                if tweedie == True:
                    xgb_tune = XtremeGBoost.XGBoost(dataset, tune=True, tweedie=True, p=tweedie_rho)
                else:
                    xgb_tune = XtremeGBoost.XGBoost(dataset, tune=True)
                norm_xgb = xgb_tune.normalization  # Store normalization hyperparameter
                params_xgb = xgb_tune.params  # Store remaining hyperparameters
                print("STEP 2.1 -- Done tuning XGBoost: {} (+{} min)".format(datetime.now().isoformat(' ', 'seconds'),
                                                                             round((time.perf_counter() - time1) / 30) / 2))
                time1 = time.perf_counter()

                filename = xgb_tune.name + 'tune'  # Pickle object for easy access later
                outfile = open(filename, 'wb')
                pickle.dump(xgb_tune, outfile)
                outfile.close()

            elif iteration == 1:  # Predict
                print("STEP 2.2 -- Start final model XGBoost: {} (+{} min)".format(datetime.now().isoformat(' ', 'seconds'),
                                                                                   round((time.perf_counter() - time1) / 30) / 2))
                time1 = time.perf_counter()

                if tweedie == True:
                    xgb_final = XtremeGBoost.XGBoost(dataset, tune=False, normalization=norm_xgb, params=params_xgb,
                                                     tweedie=True, p=tweedie_rho)
                else:
                    xgb_final = XtremeGBoost.XGBoost(dataset, tune=False, normalization=norm_xgb, params=params_xgb)

                evaluation[xgb_final.name] = {}
                evaluation[xgb_final.name]['rmse'] = xgb_final.rmse  # RMSE of final model

                if tweedie == True:
                    evaluation[xgb_final.name]['tweedie'] = xgb_final.tweedielist  # Tweedie loss of final model

                optimalparams[xgb_final.name] = xgb_final.params  # Dictionary of optimal parameters final model
                predictions[xgb_final.name] = xgb_final.yhat  # Predictions in final model
                print("STEP 2.2 -- Done final model XGBoost: {} (+{} min)".format(datetime.now().isoformat(' ', 'seconds'),
                                                                                  round((time.perf_counter() - time1) / 30) / 2))
                time1 = time.perf_counter()

                filename = xgb_final.name + 'final'
                outfile = open(filename, 'wb')
                pickle.dump(xgb_final, outfile)
                outfile.close()

            elif iteration == 2:
                print("STEP 2.3 -- Start custom tunes model XGBoost: {} (+{} min)".format(datetime.now().isoformat(' ', 'seconds'),
                                                                                          round((time.perf_counter() - time1) / 30) / 2))
                if computational_device == True:
                    pass
                else:  # This is only relevant if you run iteration == 2 on a different computer
                    time1 = time.perf_counter()
                    filename = path_comp_dev + xgb_final.name + 'final'
                    infile = open(filename, 'rb')
                    xgb_final = pickle.load(infile)
                    infile.close()

                # Denote what model to analyze, this object holds the model, prediction errors, and df of id,
                # true y, and y pred
                analysis = utils.analyze(model=xgb_final)

                # Write dataframe to csv for visualisation in R, it is possible to add suffix if desired
                analysis.writecsv(path=path,suffix='')
                pcc = analysis.perc_correctlyspec(threshold=47)  # Determine classifcation accuracy based on threshold
                pdev = analysis.perc_deviation(threshold=5)  # Percentage of points within threshold units of true Hb

    if currentrun == "MeXGBoost" or currentrun == "all":
        for iteration in range(tune, final):
            if iteration == 0:
                print("STEP 2.1 -- Start tuning MeXGB: {} (+{} min)".format(datetime.now().isoformat(' ', 'seconds'),
                                                                            round((time.perf_counter() - time1) / 30) / 2))
                time1 = time.perf_counter()
                mexgb_tune = MeXtremeGBoost.MeXGBoost(dataset, tune=True)
                norm_mexgb = mexgb_tune.normalization
                params_mexgb = mexgb_tune.params
                print("STEP 2.1 -- Done tuning MeXGB: {} (+{} min)".format(datetime.now().isoformat(' ', 'seconds'),
                                                                           round((time.perf_counter() - time1) / 30) / 2))
                time1 = time.perf_counter()

                filename = mexgb_tune + 'tune'
                outfile = open(filename, 'wb')
                pickle.dump(mexgb_tune, outfile)
                outfile.close()

            elif iteration == 1:
                print(
                    "STEP 2.2 -- Start final model MeXGB: {} (+{} min)".format(datetime.now().isoformat(' ', 'seconds'),
                                                                               round((time.perf_counter() - time1) / 30) / 2))
                time1 = time.perf_counter()
                mexgb_final = MeXtremeGBoost.MeXGBoost(dataset, tune=False, normalization=norm_mexgb,
                                                       params=params_mexgb)
                evaluation[mexgb_final.name] = mexgb_final.rmse
                optimalparams[mexgb_final.name] = mexgb_final.params
                predictions[mexgb_final.name] = mexgb_final.yhat
                print(
                    "STEP 2.2 -- Done final model MeXGB: {} (+{} min)".format(datetime.now().isoformat(' ', 'seconds'),
                                                                              round((time.perf_counter() - time1) / 30) / 2))
                time1 = time.perf_counter()

                filename = mexgb_final.name + 'final'
                outfile = open(filename, 'wb')
                pickle.dump(mexgb_final, outfile)
                outfile.close()

            elif iteration == 2:  # This is only relevant if you run iteration == 1 on a different computer
                print("STEP 2.3 -- Start custom tunes model MeXGBoost: {} (+{} min)".format(datetime.now().isoformat(' ', 'seconds'),
                                                                                            round((time.perf_counter() - time1) / 30) / 2))
                time1 = time.perf_counter()

                if computational_device == True:
                    pass
                else:  # This is only relevant if you run iteration == 2 on a different computer
                    filename = path_comp_dev + mexgb_final.name + 'final'
                    infile = open(filename, 'rb')
                    mexgb_final = pickle.load(infile)
                    infile.close()

                # Denote what model to analyze, this object holds the model, prediction errors, and df of id,
                # true y, and y pred
                analysis = utils.analyze(model=mexgb_final)
                analysis.writecsv(path=path,
                                  suffix='')  # Write dataframe to csv for visualisation in R, it is possible to add suffix if desired
                pcc = analysis.perc_correctlyspec(threshold=47)  # Determine classifcation accuracy based on threshold
                pdev = analysis.perc_deviation(threshold=5)  # Percentage of points within threshold units of true Hb

    if currentrun == "ANN" or currentrun == "all":
        for iteration in range(tune, final):
            if iteration == 0:
                print("STEP 2.1 -- Start tuning ANN: {} (+{} min)".format(datetime.now().isoformat(' ', 'seconds'),
                                                                          round((time.perf_counter() - time1) / 30) / 2))
                time1 = time.perf_counter()
                ann_tune = ANN.KerasNN(dataset, tune=True)
                norm_ann = ann_tune.normalization
                params_ann = ann_tune.params
                print("STEP 2.1 -- Done tuning ANN: {} (+{} min)".format(datetime.now().isoformat(' ', 'seconds'),
                                                                         round((time.perf_counter() - time1) / 30) / 2))
                time1 = time.perf_counter()

                filename = 'ANNTune'
                outfile = open(filename, 'wb')
                pickle.dump(ann_tune, outfile)
                outfile.close()

            elif iteration == 1:
                print("STEP 2.2 -- Start final model ANN: {} (+{} min)".format(datetime.now().isoformat(' ', 'seconds'),
                                                                               round((time.perf_counter() - time1) / 30) / 2))
                time1 = time.perf_counter()
                #norm_ann = 'standardize'
                #params_ann = {'Normalization': 3, 'activation': 'sigmoid', 'batch_size': 13,
                #              'init_dropout': 0.08056170515563518, 'learning_rate': 0.0048004463852016266,
                #              'mid_dropout': 0.6604972479807039, 'n_layers': 2, 'n_neurons': 192} #opt params from VMs
                ann_final = ANN.KerasNN(dataset, tune=False, normalization=norm_ann, params=params_ann)
                evaluation[ann_final.name] = ann_final.rmse
                optimalparams[ann_final.name] = ann_final.params
                predictions[ann_final.name] = ann_final.yhat
                print("STEP 2.2 -- Done final model ANN: {} (+{} min)".format(datetime.now().isoformat(' ', 'seconds'),
                                                                              round((time.perf_counter() - time1) / 30) / 2))
                time1 = time.perf_counter()

                filename = 'ANNFinal'
                outfile = open(filename, 'wb')
                pickle.dump(ann_final, outfile)
                outfile.close()

                # Load keras model (this does not load the self object
                model_json = ann_final.model.to_json()
                with open("model.json", "w") as json_file:
                    json_file.write(model_json)

                ann_final.model.save_weights("model.h5")

                file = open('model.json', 'r')
                loaded = file.read()
                file.close()

                loaded_model = model_from_json(loaded)
                loaded_model.load_weights("model.h5")

            elif iteration == 2:  # This is only relevant if you run iteration == 1 on a different computer
                print("STEP 2.3 -- Start custom tunes model ANN: {} (+{} min)".format(
                    datetime.now().isoformat(' ', 'seconds'),
                    round((time.perf_counter() - time1) / 30) / 2))
                time1 = time.perf_counter()

                infile = open('VM/ANNFinal',
                              'rb')  # This opens the ANNFinal, which I had to run on a different computer
                ann_custom = pickle.load(infile)
                infile.close()

                analysis = utils.analyze(
                    model=ann_custom)  # Denote what model to analyze, this object holds the model, prediction errors, and df of id, true y, and y pred
                analysis.writecsv(path=path,
                                  suffix='')  # Write dataframe to csv for visualisation in R, it is possible to add suffix if desired
                pcc = analysis.perc_correctlyspec(threshold=47)  # Determine classifcation accuracy based on threshold
                pdev = analysis.perc_deviation(
                    threshold=5)  # Percentage of points within bounds of threshold units of Hb

    if currentrun == "MeANN" or currentrun == "all":
        for iteration in range(tune, final):
            if iteration == 0:
                print("STEP 2.1 -- Start tuning MeANN: {} (+{} min)".format(datetime.now().isoformat(' ', 'seconds'),
                                                                            round((
                                                                                              time.perf_counter() - time1) / 30) / 2))
                time1 = time.perf_counter()
                meann_tune = MeANN.MeKerasNN(dataset, tune=True)
                norm_meann = meann_tune.normalization
                params_meann = meann_tune.params
                print("STEP 2.1 -- Done tuning MeANN: {} (+{} min)".format(datetime.now().isoformat(' ', 'seconds'),
                                                                           round(
                                                                               (time.perf_counter() - time1) / 30) / 2))
                time1 = time.perf_counter()

                filename = 'MEANNTune'
                outfile = open(filename, 'wb')
                pickle.dump(meann_tune, outfile)
                outfile.close()
            elif iteration == 1:
                print(
                    "STEP 2.2 -- Start final model MeANN: {} (+{} min)".format(datetime.now().isoformat(' ', 'seconds'),
                                                                               round((
                                                                                                 time.perf_counter() - time1) / 30) / 2))
                time1 = time.perf_counter()
                meann_final = MeANN.MeKerasNN(dataset, tune=False, normalization=norm_meann, params=params_meann)
                evaluation[meann_final.name] = meann_final.rmse
                optimalparams[meann_final.name] = meann_final.params
                predictions[meann_final.name] = meann_final.yhat
                print(
                    "STEP 2.2 -- Done final model MeANN: {} (+{} min)".format(datetime.now().isoformat(' ', 'seconds'),
                                                                              round((
                                                                                                time.perf_counter() - time1) / 30) / 2))
                time1 = time.perf_counter()

                filename = 'MEANNFinal'
                outfile = open(filename, 'wb')
                pickle.dump(meann_final, outfile)
                outfile.close()
            elif iteration == 2:  # This is only relevant if you run iteration == 1 on a different computer
                print("STEP 2.3 -- Start custom tunes model MeANN: {} (+{} min)".format(
                    datetime.now().isoformat(' ', 'seconds'),
                    round((time.perf_counter() - time1) / 30) / 2))
                time1 = time.perf_counter()

                infile = open('VM/MeANNFinal',
                              'rb')  # This opens the ANNFinal, which I had to run on a different computer
                meann_custom = pickle.load(infile)
                infile.close()

                analysis = utils.analyze(
                    model=meann_custom)  # Denote what model to analyze, this object holds the model, prediction errors, and df of id, true y, and y pred
                analysis.writecsv(path=path,
                                  suffix='')  # Write dataframe to csv for visualisation in R, it is possible to add suffix if desired
                pcc = analysis.perc_correctlyspec(threshold=47)  # Determine classifcation accuracy based on threshold
                pdev = analysis.perc_deviation(
                    threshold=5)  # Percentage of points within bounds of threshold units of Hb

    if currentrun not in ["XGBoost", "MeXGBoost", "ANN", "MeANN", "all"]:
        print("Current run method {} not implemented.".format(currentrun))
        raise NotImplementedError

    else:
        print("Process successfully finished: {}".format(datetime.now().isoformat(' ', 'seconds'),
                                                         round((time.perf_counter() - time1) / 30) / 2))
        print("Total runtime: {} minutes".format(round((time.perf_counter() - time_total) / 30) / 2))
