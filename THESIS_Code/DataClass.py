import pandas as pd
import numpy as np
import copy
import sklearn
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, QuantileTransformer
"""Class of a dataset. It has the option to be resampled, and it automatically generates the (resampled) folds
that are used for cross-validation. This is done to save computation time per model while evaluating performance."""

path = "V:/UserData/079915/Thesis -- Data"

class DataClass:
    def __init__(self, path, normalization=False, normalization_scheme="none", folds=4, seed=0):
        y, X_gaus, X_bin = self.load_data(path) # Load data
        self.seed = seed
        self.folds = folds

        if normalization: # If you wish to normalize the data, instead of feeding this as hyperparameter, set normalization to one of [minmax, standardize, robust, quantile].
            X_gaus = self.normalize(X_gaus, normalization_scheme)

        X = X_bin.join(X_gaus, lsuffix="", rsuffix="2") # Join binary and non-binary covariates to create X.
        X = X.drop(["id2"], axis=1) # Delete unnecessary id.

        # Create variables necessary later on
        maxage = X.groupby('id')['participation_age'].transform(max)  # Calculate oldest age per id
        counter = X.groupby('id')['id'].transform('count')  # Calculate number of rounds id participates in
        X = pd.concat([X, counter.rename('counter'), maxage.rename('maxage')], axis=1)

        # Not all ages are correctly specified in the dot set, so we delete the wrong observations
        equal_age = X[X["maxage"] == X["participation_age"]]
        doubles_count = equal_age.groupby("id")['id'].transform('count') # Total obs per id
        wrong_age = pd.concat([equal_age, doubles_count.rename("double")], axis=1)
        doubles = np.unique(wrong_age[wrong_age["double"] == 2]["id"])
        X = X[~X.id.isin(doubles)]  # Delete id's from X
        y = y[~y.id.isin(doubles)]  # Delete id's from y

        X_og = copy.deepcopy(X)
        y_og = copy.deepcopy(y)

        # Since this research uses panel data, we cannot split the data set as one would normally do, as we do not want observations from people in the train set also in the test set.
        # Thus we modify the X and y to only include one observation per individual (it doesn't matter which one because we add back all observations in the end).
        X_indiv = X.groupby("id").max().reset_index()
        y_indiv = y.groupby("id").max().reset_index()

        # Create test set for X and Y, and a non-test set for X and Y with Shuffle = False because ordering of observations is irrelevant
        self.skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.seed)  # Folds used for k-cross validation.
        X_nontest_indiv, X_test_indiv, y_nontest_indiv, y_test_indiv = sklearn.model_selection.train_test_split(X_indiv, y_indiv, test_size = 0.3, shuffle=False) #0.3

        # Create list of test folds, with all observations per included id reinstated
        self.Xtestfold_finalfit = [X[X.id.isin(X_test_indiv["id"])]] # Test samples of X (= testsize*100% of the data), used for final testing
        self.ytestfold_finalfit = [y[y.id.isin(y_test_indiv["id"])]]

        self.Xtrainfold_finalfit = [X[X.id.isin(X_nontest_indiv["id"])]] # Test samples of X (= (1-testsize)*100% of the data), used for hyperparameter tuning and final training
        self.ytrainfold_finalfit = [y[y.id.isin(y_nontest_indiv["id"])]]

        ## Create the extra test set
        # First, create a copy of non test set (x and y)
        xremain = pd.concat([X_nontest_indiv, y_nontest_indiv], axis=1)
        xremain.columns = np.append(X_nontest_indiv.columns.values, ['y_id','y_hb'])  # Rename appended values

        # Create 3 subsets with 3000 ids which participate 4 rounds, 3 rounds, and 2 rounds
        rounds4 = xremain.loc[xremain['counter'] == 4].sample(n=3334)  # Extract 3000 ids with 4 rounds
        xremain = xremain[~xremain.id.isin(rounds4["id"])]  # Remove above ids from nontest set (will be added back in later)
        # without the last obs)

        rounds3 = xremain.loc[xremain['counter'] == 3].sample(n=3333)
        xremain = xremain[~xremain.id.isin(rounds3["id"])]

        rounds2 = xremain.loc[xremain['counter'] == 2].sample(n=3333)
        xremain = xremain[~xremain.id.isin(rounds2["id"])]

        extra_test = pd.concat([rounds2,rounds3,rounds4])  # Vertically append all three data sets

        # Create df consisting of all observations for each id in extra_test (should be of length 3000*(2+3+4))
        xextra = X[X.id.isin(extra_test["id"])]
        yextra = y[y.id.isin(extra_test["id"])]

        # Create fold consisting of only last observation for id in xextra
        # (Last observation := obs where maximum age == participation age)
        self.Xextratest_finalfold = [xextra[xextra["maxage"] == xextra["participation_age"]]]
        self.yextratest_finalfold = [yextra[yextra.index.isin(self.Xextratest_finalfold[0].index)]]  # Alternative: yextra.loc[self.Xextratest_finalfold.index]

        # Create fold consisting of only first observations for id in xextra
        # (First observation := all obs in xextra excluding xextratest_finalfold)
        xremain2 = xextra[xextra["maxage"] != xextra["participation_age"]]  # First observations of the 9k individuals
        yremain2 = yextra[~yextra.index.isin(self.Xextratest_finalfold[0].index)]

        # Now create a replacement for X and y which contains all observations in X_nontest
        # except for the 9000 consisting in xextratest_finalfold through first making df of all observations for every
        # id not in extra_test (and thus *is* in xremain)
        X_star_prec = X[X.id.isin(xremain["id"])]
        y_star_prec = y[y.id.isin(xremain["id"])]

        # Then add the first observations for the ids in extra_test (i.e., xremain2)
        X_star = pd.concat([X_star_prec, xremain2])
        y_star = pd.concat([y_star_prec, yremain2])

        X = X_star  # For making the following folds, we now use the X_star and y_star
        y = y_star

        # Create list of non-test folds (i.e., training and validation folds).
        self.Xtrainfolds = [] # Training samples of X (= (nr_folds-1)/nr_folds*(1-testsize)*100% of the data)
        self.ytrainfolds = []
        self.Xvalidationfolds = [] # Validation samples of X (= 1/nr_folds*(1-testsize)*100% of the data per fold)
        self.yvalidationfolds = []

        # Stratified K-fold only takes binary variable to stratify on, so convert dependent variable to 1 when hb > 0, and 0 o.w.
        stratify = copy.deepcopy(y_nontest_indiv) # Must make deep copy, o.w. you edit y_nontest_indiv
        stratify["hb"] = np.where(stratify["hb"] > 0, 1, stratify["hb"])

        i = -1
        for train_index, test_index in self.skf.split(X_nontest_indiv, stratify["hb"]): # Stratify is only used to produce indices, it is NOT used as input anywhere else
            i += 1
            #print("fold:", i, "ids:", train_index, "num of ids:", len(train_index))
            #print("fold:", i, "ids:", test_index, "num of ids:", len(test_index))
            fold_X_train, fold_X_val = X_nontest_indiv.iloc[train_index], X_nontest_indiv.iloc[test_index]
            fold_y_train, fold_y_val = y_nontest_indiv.iloc[train_index], y_nontest_indiv.iloc[test_index]

            self.Xtrainfolds.append(X[X.id.isin(fold_X_train["id"])]) # List of nr_folds dataframes for training
            self.ytrainfolds.append(y[y.id.isin(fold_y_train["id"])])
            self.Xvalidationfolds.append(X[X.id.isin(fold_X_val["id"])]) # List of nr_folds dataframes for validation
            self.yvalidationfolds.append(y[y.id.isin(fold_y_val["id"])])

    # This function loads the dataset from a pre-specified path.
    def load_data(self, path):
        path += "/final_imputed_dataset.csv"
        hb_dataset = pd.read_csv(path)
        df = hb_dataset[["id", "participation_age","hb", "hb_stage_cat", "sex", "FIT", "hb_previous", "hb_max", "hb_min","birthyear"]] # Only include relevant observations
        y = df[["id","hb"]] # Define dependent variable
        X_gaus = df.drop(["hb","hb_stage_cat","FIT"],axis=1) # Define continuous or binary covariates
        X_cat = df.drop(["hb","participation_age","sex","hb_previous","hb_max","hb_min","birthyear"],axis=1) # Define categorical covariates (our data set does not include nominal variables)
        X_bin = pd.get_dummies(X_cat, columns = ["hb_stage_cat","FIT"], drop_first=True) # Perform one-hot-encoding on dummies
        return y, X_gaus, X_bin

    def normalize(self, X, normalization):
        X_norm = X[["participation_age","hb_previous","hb_max","hb_min","birthyear"]] # Only perform normalization on continuous covariates (so X_gaus above, excluding binary covariate "sex" and "id")

        if normalization == 'minmax': # Scales and translates each feature between zero (minimum) and one (maximum).
            X_norm = pd.DataFrame(MinMaxScaler().fit_transform(X_norm), columns=X_norm.columns).set_axis(X.index, axis='index')
            X = X.assign(**{"participation_age":X_norm["participation_age"],
                                                "hb_previous":X_norm["hb_previous"],
                                                "hb_max":X_norm["hb_max"],
                                                "hb_min":X_norm["hb_min"],
                                                "birthyear":X_norm["birthyear"]})
            return X
        elif normalization == 'standardize': # Standardize features by removing the mean and scaling to unit variance.
            X_norm = pd.DataFrame(StandardScaler().fit_transform(X_norm), columns=X_norm.columns).set_axis(X.index, axis='index')
            X = X.assign(**{"participation_age": X_norm["participation_age"],
                            "hb_previous": X_norm["hb_previous"],
                            "hb_max": X_norm["hb_max"],
                            "hb_min": X_norm["hb_min"],
                            "birthyear": X_norm["birthyear"]})
            return X
        elif normalization=='robust': # Removes the median and scales the data according to the quantile range (defaults to IQR: the range between the 1st quartile (25th quantile) and the 3rd quartile (75th quantile)).
            X_norm = pd.DataFrame(RobustScaler().fit_transform(X_norm), columns=X_norm.columns).set_axis(X.index, axis='index')
            X = X.assign(**{"participation_age": X_norm["participation_age"],
                            "hb_previous": X_norm["hb_previous"],
                            "hb_max": X_norm["hb_max"],
                            "hb_min": X_norm["hb_min"],
                            "birthyear": X_norm["birthyear"]})
            return X
        elif normalization=='quantile': # This method transforms the features to follow a uniform or a normal distribution (also robust).
            X_norm = pd.DataFrame(QuantileTransformer().fit_transform(X_norm), columns=X_norm.columns).set_axis(X.index, axis='index')
            X = X.assign(**{"participation_age": X_norm["participation_age"],
                            "hb_previous": X_norm["hb_previous"],
                            "hb_max": X_norm["hb_max"],
                            "hb_min": X_norm["hb_min"],
                            "birthyear": X_norm["birthyear"]})
            return X
        elif normalization=='none':
            return X
        else:
            return print('Error:', normalization, 'not in normalization methods')