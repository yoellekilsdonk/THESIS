"""
Mixed Effects Random Forest model.
"""
import logging
import sys
import time
import copy
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from datetime import datetime
from tqdm import tqdm

logger = logging.getLogger(__name__)

def Estep(inputE):
    # inputE = [(y_by_cluster, Z_by_cluster, b_hat_df, y_star, group_indices, i) for i in Z_by_cluster.keys()]
    y_by_cluster = inputE[0]
    Z_by_cluster = inputE[1]
    b_hat_df = inputE[2]
    y_star = inputE[3]
    group_indices = inputE[4]
    i = inputE[5]

    y_i = y_by_cluster[i] # Obtain dependent variable per id
    Z_i = Z_by_cluster[i] # Obtain Z matrix per id
    b_hat_i = b_hat_df.loc[i] # Obtain B matrix per id

    y_star[group_indices[i]] = y_i - Z_i.dot(b_hat_i) # Calculate y_star per id
    return y_star

def Mstep(inputM):
    # inputM = [(y_by_cluster, Z_by_cluster, n_by_cluster, I_by_cluster, group_indices, f_hat, sigma_hat, D_hat, b_hat_df, i) for i in Z_by_cluster.keys()]
    y_by_cluster = inputM[0]
    Z_by_cluster = inputM[1]
    n_by_cluster = inputM[2]
    I_by_cluster = inputM[3]
    group_indices = inputM[4]
    f_hat = inputM[5]
    sigma2_hat2 = inputM[6]
    D_hat2 = inputM[7]
    b_hat_df = inputM[8]
    i = inputM[9]

    y_i = y_by_cluster[i]
    Z_i = Z_by_cluster[i]
    n_i = n_by_cluster[i] # Obtain number of observations per id
    I_i = I_by_cluster[i] # Obtain I matrix per id

    f_hat_i = f_hat[group_indices[i]] # Obtain predicted y_star values per id
    V_hat_i = Z_i.dot(D_hat2).dot(Z_i.T) + sigma2_hat2 * I_i # Calculate V_i

    V_hat_inv_i = np.linalg.pinv(V_hat_i) # Take inverse of V_i
    b_hat_i = D_hat2.dot(Z_i.T).dot(V_hat_inv_i).dot(y_i - f_hat_i) # Update B_i

    # Compute the total error for this cluster
    eps_hat_i = y_i - f_hat_i - Z_i.dot(b_hat_i)

    b_hat_df.loc[i, :] = b_hat_i

    sigma_hat_sum = eps_hat_i.T.dot(eps_hat_i) + sigma2_hat2 * (
            n_i - sigma2_hat2 * np.trace(V_hat_inv_i)) # Calculate sigma_i
    D_hat_sum = np.outer(b_hat_i, b_hat_i) + (
            D_hat2 - D_hat2.dot(Z_i.T).dot(V_hat_inv_i).dot(Z_i).dot(D_hat2) # Calculate D_i
    )
    return sigma_hat_sum, D_hat_sum, b_hat_df

def GLLstep(inputGLL):
    # inputGLL = [(i, y_by_cluster, Z_by_cluster, b_hat_df, I_by_cluster, group_indices, f_hat, sigma_hat, D_hat) for i in Z_by_cluster.keys()]
    i = inputGLL[0]
    y_by_cluster = inputGLL[1]
    Z_by_cluster = inputGLL[2]
    b_hat_df = inputGLL[3]
    I_by_cluster = inputGLL[4]
    group_indices = inputGLL[5]
    f_hat = inputGLL[6]
    sigma2_hat2 = inputGLL[7]
    D_hat = inputGLL[8]

    y_i = y_by_cluster[i]
    Z_i = Z_by_cluster[i]
    I_i = I_by_cluster[i]

    # Slice f_hat and get b_hat
    f_hat_i = f_hat[group_indices[i]]
    R_hat_i = sigma2_hat2 * I_i
    b_hat_i = b_hat_df.loc[i]

    # Numerically stable way of computing log(det(A))
    _, logdet_D_hat = np.linalg.slogdet(D_hat)
    _, logdet_R_hat_i = np.linalg.slogdet(R_hat_i)

    # Calculate gll
    gll = (
            (y_i - f_hat_i - Z_i.dot(b_hat_i))
            .T.dot(np.linalg.pinv(R_hat_i))
            .dot(y_i - f_hat_i - Z_i.dot(b_hat_i))
            + b_hat_i.T.dot(np.linalg.pinv(D_hat)).dot(b_hat_i)
            + logdet_D_hat
            + logdet_R_hat_i
    )
    return gll

class MERF(object):
    """
    This is the core class to instantiate, train, and predict using a mixed effects random forest model.
    It roughly adheres to the sklearn estimator API.
    Note that the user must pass in an already instantiated fixed_effects_model that adheres to the
    sklearn regression estimator API, i.e. must have a fit() and predict() method defined.

    It assumes a data model of the form:

    .. math::

        y = f(X) + b_i Z + e

    * y is the target variable. The current code only supports regression for now, e.g. continuously varying scalar value
    * X is the fixed effect features. Assume p dimensional
    * f(.) is the nonlinear fixed effects mode, e.g. random forest
    * Z is the random effect features. Assume q dimensional.
    * e is iid noise ~N(0, sigma_e²)
    * i is the cluster index. Assume k clusters in the training.
    * bi is the random effect coefficients. They are different per cluster i but are assumed to be drawn from the same distribution ~N(0, Sigma_b) where Sigma_b is learned from the data.


    Args:
        fixed_effects_model (sklearn.base.RegressorMixin): instantiated model class
        gll_early_stop_threshold (float): early stopping threshold on GLL improvement
        max_iterations (int): maximum number of EM iterations to train
    """

    def __init__(
        self,
        fixed_effects_model=None,
        gll_early_stop_threshold=None,
        max_iterations=20,
    ):
        self.gll_early_stop_threshold = gll_early_stop_threshold
        self.max_iterations = max_iterations

        self.cluster_counts = None
        # Note fixed_effects_model must already be instantiated when passed in.
        self.fe_model = fixed_effects_model
        self.trained_fe_model = None
        self.trained_b = None

        self.b_hat_history = []
        self.sigma2_hat_history = []
        self.D_hat_history = []
        self.gll_history = []
        self.val_loss_history = []

    def predict(self, X: np.ndarray, Z: np.ndarray, clusters: pd.Series):
        """
        Predict using trained MERF.  For known clusters the trained random effect correction is applied.
        For unknown clusters the pure fixed effect (RF) estimate is used.

        Args:
            X (np.ndarray): fixed effect covariates
            Z (np.ndarray): random effect covariates
            clusters (pd.Series): cluster assignments for samples

        Returns:
            np.ndarray: the predictions y_hat
        """
        print("start: {}".format(datetime.now().isoformat(' ', 'seconds')))
        if type(clusters) != pd.Series:
            raise TypeError("clusters must be a pandas Series.")

        if self.trained_fe_model is None:
            raise NotFittedError(
                "This MERF instance is not fitted yet. Call 'fit' with appropriate arguments before "
                "using this method"
            )

        Z = np.array(Z)  # cast Z to numpy array (required if it's a dataframe, otw, the matrix mults later fail)

        # Apply fixed effects model to all
        y_hat = self.trained_fe_model.predict(X)

        # Check for each individual in the current data set if it previously occured in the clusters used for fitting the models
        known_cluster = []
        for i in clusters.value_counts().index:  # for id in current cluster
            if i in self.cluster_counts.index:  # if id in fitting cluster
                known_cluster.append(i)  # append known cluster ID (for which we have RE estimates)

        # We only enter this cycle for individuals for which we have known random effects
        for cluster_id in known_cluster:
            indices_i = clusters == cluster_id

            # The following should never happen because we only iterate over the known clusters
            assert len(indices_i) != 0

            # If cluster does exist, apply the correction.
            b_i = self.trained_b.loc[cluster_id]
            Z_i = Z[indices_i]
            y_hat[indices_i] += Z_i.dot(b_i)
        return y_hat

    def fit(
        self,
        X: np.ndarray,
        Z: np.ndarray,
        clusters: pd.Series,
        y: np.ndarray,
        X_val: np.ndarray = None,
        Z_val: np.ndarray = None,
        clusters_val: pd.Series = None,
        y_val: np.ndarray = None,
    ):
        """
        Fit MERF using Expectation-Maximization algorithm.

        Args:
            X (np.ndarray): fixed effect covariates
            Z (np.ndarray): random effect covariates
            clusters (pd.Series): cluster assignments for samples
            y (np.ndarray): response/target variable

        Returns:
            MERF: fitted model
        """

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Input Checks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if type(clusters) != pd.Series:
            raise TypeError("clusters must be a pandas Series.")

        assert len(Z) == len(X) # Length of random effects regressor matrix should equal that of fixed effects regressors
        assert len(y) == len(X) # Length of dependent variable should equal that of fixed effects regressors
        assert len(clusters) == len(X) # Lenght of cluster indicator should equal that of fixed effects regressors

        if X_val is None: # Always the case in our application
            assert Z_val is None
            assert clusters_val is None
            assert y_val is None
        else:
            assert len(Z_val) == len(X_val)
            assert len(clusters_val) == len(X_val)
            assert len(y_val) == len(X_val)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        logger.info("Initialising variables to fit MERF: {}".format(datetime.now().isoformat(' ', 'seconds')))
        n_clusters = len(set(clusters)) # Faster way to calculate unique occurences (= number of clusters) in clusters
        n_obs = len(y) # Number of observations
        q = Z.shape[1]  # Random effects dimension (=1 in our case)
        # Z = np.array(Z)  # cast Z to numpy array (required if it's a dataframe, otw, the matrix mults later fail)
        # to be fair this code doesnt use Z, because I fix it to Z=1 for length of y during creation of dictionaries
        self.cluster_counts = clusters.value_counts() # Create a series where cluster_id is the index and n_i (num obs per cluster) is the value

        logger.info("Creating dictionaries per cluster: {}".format(datetime.now().isoformat(' ', 'seconds')))
        pd.options.mode.chained_assignment = None
        y["id"] = copy.deepcopy(clusters)
        pd.options.mode.chained_assignment = 'warn'
        del clusters # No longer necessary beyond this point, delete to save memory

        group = y.groupby("id")
        y_by_cluster = {int(k): value.astype('float64') for k, value in group} # dictionary of (n_i,) Series of dependent variable value per id
        Z_by_cluster = {int(k): np.ones((value.count(),1)) for k, value in group} # dictionry of [1 x n_i] vector of ones per id
        n_by_cluster = {int(k): value.count() for k, value in group} # dictionary of n_i per id
        I_by_cluster = {int(k): np.eye(value.count()) for k, value in group} # dictionary of [n_i x n_i] identity matrices per id
        group_indices = {int(k): value for k,value in group.indices.items()} # dictionary of (n_i,) array with indices of cluster per id
        del group, y # Delete to save memory

        b_hat_df = pd.DataFrame(np.zeros((n_clusters, q)), index=self.cluster_counts.index).sort_index(axis=0) # Initialise B as df of [n_clusters x q] zeros

        sigma_hat = 1 # Initialise variance epsilon
        D_hat = np.eye(q) # Initialise variance B

        # vectors to hold history
        self.b_hat_history.append(b_hat_df)
        self.sigma2_hat_history.append(sigma_hat)
        self.D_hat_history.append(D_hat)

        early_stop_flag = False # Not implemented

        logger.info("Starting EM-algorithm: {}".format(datetime.now().isoformat(' ', 'seconds')))
        for iteration in tqdm(range(0,self.max_iterations), ascii= False, desc="EM-algorithm MeXGBoost"):
            if early_stop_flag == True:
                raise NotImplementedError()

            # E-step:
            y_star = np.zeros(n_obs) # Initialise y_star as vector of zeros of length len(y)
            inputE = [(y_by_cluster, Z_by_cluster, b_hat_df, y_star, group_indices, i) for i in
                      Z_by_cluster.keys()] # List of [n_cluster x 6] used as input for map function
            y_star = list(map(Estep, inputE))[-1] # Select last element of map object (doesn't matter which one though they're all the same)
            del inputE

            # check that still one dimensional
            assert len(y_star.shape) == 1

            # Do the fixed effects regression with all the fixed effects features
            self.fe_model.fit(X, y_star)
            del y_star
            f_hat = self.fe_model.predict(X) # UPDATE F_HAT

            # M-step:
            inputM = [(y_by_cluster, Z_by_cluster, n_by_cluster, I_by_cluster, group_indices, f_hat, sigma_hat,
                       D_hat, b_hat_df, i) for i in Z_by_cluster.keys()]
            sigmaD = list(zip(*map(Mstep, inputM))) # Returns list containing sigma_hat and D_hat
            del inputM
            sigma_hat = (1.0 / n_obs) * (sum(sigmaD[0])) # UPDATE SIGMA
            D_hat = (1.0 / n_clusters) * (sum(sigmaD[1])) # UPDATE D
            b_hat_df = sigmaD[2][-1] # UPDATE B

            assert len(D_hat) == q == b_hat_df.shape[1]
            assert b_hat_df.shape[0] == n_clusters

            # Store off history so that we can see the evolution of the EM algorithm
            self.b_hat_history.append(b_hat_df.copy())
            self.sigma2_hat_history.append(sigma_hat)
            self.D_hat_history.append(D_hat)

            # Calculate GLL:
            inputGLL = [(i, y_by_cluster, Z_by_cluster, b_hat_df, I_by_cluster, group_indices,
                         f_hat, sigma_hat, D_hat) for i in Z_by_cluster.keys()]
            gll = sum(list(map(GLLstep, inputGLL)))
            del inputGLL

            logger.info("Training GLL is {} at iteration {}: {}".format(gll, iteration, datetime.now().isoformat(' ', 'seconds')))
            self.gll_history.append(gll)

            # Save off the most updated fixed effects model and random effects coefficents
            self.trained_fe_model = self.fe_model
            self.trained_b = b_hat_df

            # # If you do want to use Early Stopping, you must adjust the following code:
            # if self.gll_early_stop_threshold is not None and len(self.gll_history) > 1:
            #     curr_threshold = np.abs((gll - self.gll_history[-2]) / self.gll_history[-2])
            #     logger.debug("stop threshold = {}".format(curr_threshold))
            #
            #     if curr_threshold < self.gll_early_stop_threshold:
            #         logger.info("Gll {} less than threshold {}, stopping early ...".format(gll, curr_threshold))
            #         early_stop_flag = True
            #
            # # Compute Validation Loss
            # if X_val is not None:
            #     print("error: you've passed X_val, not done coding")
            #     yhat_val = self.predict(X_val, Z_val, clusters_val)
            #     val_loss = np.square(np.subtract(y_val, yhat_val)).mean()
            #     logger.info(f"Validation MSE Loss is {val_loss} at iteration {iteration}.")
            #     self.val_loss_history.append(val_loss)
        logger.info("Finished fitting MERF: {}".format(datetime.now().isoformat(' ', 'seconds')))
        return self

    def score(self, X, Z, clusters, y):
        raise NotImplementedError()

    def get_bhat_history_df(self):
        """
        This function does a complicated reshape and re-indexing operation to get the
        list of dataframes for the b_hat_history into a multi-indexed dataframe.  This
        dataframe is easier to work with in plotting utilities and other downstream
        analyses than the list of dataframes b_hat_history.

        Args:
            b_hat_history (list): list of dataframes of bhat at every iteration

        Returns:
            pd.DataFrame: multi-index dataframe with outer index as iteration, inner index as cluster
        """
        # Step 1 - vertical stack all the arrays at each iteration into a single numpy array
        b_array = np.vstack(self.b_hat_history)

        # Step 2 - Create the multi-index. Note the outer index is iteration. The inner index is cluster.
        iterations = range(len(self.b_hat_history))
        clusters = self.b_hat_history[0].index
        mi = pd.MultiIndex.from_product([iterations, clusters], names=("iteration", "cluster"))

        # Step 3 - Create the multi-indexed dataframe
        b_hat_history_df = pd.DataFrame(b_array, index=mi)
        return b_hat_history_df