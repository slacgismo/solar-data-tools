from anomalydetector.multidata_handler import MultiDataHandler
from anomalydetector.utils import (
                    full_signal,
                    reconstruct_signal,
                    form_xy,
                    reshape_for_cheb,
                    make_cheb_basis,
                    optimal_weighted_regression,
                    fit_dask)
from spcqe.quantiles import SmoothPeriodicQuantiles

import numpy as np
import matplotlib.pyplot as plt
import warnings
import joblib
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score


class QLinear:
    def __init__(self,
                 sites,
                 ndil = None,
                 target = None
                 ):
        """
        This class contains the object QLinear which is the superposition of a quantile estimation, linear model and 
        binary classification. The object contains the 3 models and the intermediate results for the train set and test set.
        :param sites: the list of the name of the sites
        :type sites: str list

        :param ndil: the number of steps by days, if the dilution of the MultidataHandler hasn't been done the fit/predict 
        start by diluating the data.
        :type ndil: int, optional

        :param target: the name of the target, if the failure scenario generation hasn't been made the fit function start by 
        generating a failure scenarion with default parameters.
        :type target: str, optional
        """
        self.sites = sites
        self.ndil = ndil
        self.target = target
        self._initialize_attributes()

    def _initialize_attributes(self):
        self.spqs = None
        self.start_train = None
        self.quantile_train = None
        self.quantile_test = None
        self.quantile_failure = None
        self.quantile_failure_test = None
        self.residuals_train = None
        self.residuals_test = None
        self.residuals_failure = None
        self.residuals_failure_test = None
        self.linear_coeff = None
        self.preprocess = None
        self.residual_model = None
        self.nlag = None
        self.num_basis = None

    def fit_quantiles(self,multidata,param = {}):
        """
        This function fits the first part of the model and compute the intermediate results for the training set
        :param multidata: This object contains all information about the sites, if the data haven't been diluated yet 
        and self.ndil is given the function start by diluating the data. 
        :type multidata: MultiDataHandler
        :param param: Dictionnary of every possible parameters for the learning step: weight_quantiles, quantiles, 
        solver_quantiles, num_harmonics,client.
        :type param: dict
        """

        if multidata.dil_mat is not None and self.ndil is not None and multidata.ndil() != self.ndil:
            raise ValueError("ndil between the argument and the model is different")
        elif multidata.dil_mat is None and self.ndil is not None :
            multidata.dilate(ndil = self.ndil)
        elif multidata.dil_mat is not None and self.ndil is None :
            self.ndil = multidata.ndil()
        elif multidata.dil_mat is None and self.ndil is None :
            raise ValueError('No ndil given')
        
        if multidata.target is not None and self.target is not None and multidata.target != self.target:
            raise ValueError("target between the argument and the model is different")
        elif multidata.target is None and self.target is not None :
            multidata.generate_failure(self.target)
            warnings.warn(
                "No failure scenario detected — calling multidata.generate_failure automatically with default parameters.",
                category=UserWarning
            )
        elif multidata.target is not None and self.target is None :
            self.target = multidata.target
        elif multidata.target is None and self.target is None :
            raise ValueError('No target given')
        
        _default_qs = np.array([0.02, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.98])
        self.start_train = multidata.common_days[0]
        weight = param.get("weight_quantiles", 5)
        quantiles = param.get("quantiles", _default_qs)
        solver = param.get("solver_quantiles", "clarabel")
        num_harmonic = param.get("num_harmonics", [30,3])
        client = param.get("client", None)

        dict_quantile = {}
        for site in self.sites :
            dict_quantile[site] = SmoothPeriodicQuantiles(
            num_harmonics=num_harmonic,
            periods=[self.ndil, 365.24225*self.ndil],
            standing_wave=[True, False],
            trend=False,
            quantiles=quantiles,
            weight=weight,
            problem='sequential',
            solver=solver,
            extrapolate='solar'
            )
        self.spqs = dict_quantile

        if client is not None :
            full_array_dict = {}
            for key in multidata.keys :
                full_array,_ = full_signal(multidata.common_days,self.start_train,multidata.dil_mat[key],self.ndil)
                full_array_dict[key] = full_array
            futures_spqs = {key : client.scatter(self.spqs[key],hash=False) for key in self.spqs}
            tasks = [fit_dask(futures_spqs[key],full_array_dict[key]) for key in multidata.keys]
            futures = client.compute(tasks)
            client.gather(futures)
            for key in multidata.keys :
                self.spqs[key] = futures_spqs[key].result()


        self.quantile_train = {}
        for key in multidata.dil_mat :
            full_array,_ = full_signal(multidata.common_days,self.start_train,multidata.dil_mat[key],self.ndil)
            self.spqs[key].fit(full_array)
            self.quantile_train[key] = reconstruct_signal(self.spqs[key].transform(full_array),multidata.common_days,self.ndil)
        full_failure,_ = full_signal(multidata.common_days,self.start_train,multidata.failure_mat,self.ndil)
        self.quantile_failure = reconstruct_signal(self.spqs[self.target].transform(full_failure),multidata.common_days,self.ndil)

    def fit_linear(self,param = {}):
        """
        This function fit the linear regression part. It creates some lagged features and learn a linear regression where the coefficients
        between each step in a day is driven by a chebyvev basis.
        :param param: Dictionnary of every possible parameters for the learning step: nlag, num_basis, weight_linear
        lambda_range, num_split.
        :type param: dict
        """
        X = np.array([self.quantile_train[key] for key in self.quantile_train if key != self.target])
        y = self.quantile_train[self.target]
        y_failure = self.quantile_failure
        #Numerical stability
        X = np.clip(X,-4,4)
        y = np.clip(y,-4,4)
        y_failure = np.clip(y_failure,-4,4)
        X[np.isnan(X)] = 0
        y[np.isnan(y)] = 0
        y_failure[np.isnan(y_failure)] = 0

        nlag = param.get("nlag", 3)
        num_basis = param.get("num_basis", 8)
        n_features = (len(self.sites)-1) * (2*nlag+1) + 1
        _default_weight = np.tile(np.arange(1, num_basis+1)**2,n_features)
        _default_lbda = np.logspace(0, 3, num=10)
        weights = param.get("weight_linear", _default_weight)
        lbda_values = param.get("lambda_range", _default_lbda)
        n_split = param.get("num_split",5)
        self.nlag = nlag
        self.num_basis = num_basis

        X,y,y_failure = form_xy(X,y,y_failure,nlag)
        X_flat,y_flat = reshape_for_cheb(X,y,num_basis)
        _,y_failure_flat = reshape_for_cheb(X,y_failure,num_basis)
        self.lambda_optimal,self.linear_coeff = optimal_weighted_regression(X_flat,y_flat,weights,lbda_values,n_splits=n_split)
        self.residuals_train = (y_flat-X_flat@self.linear_coeff).reshape(y.shape)
        self.residuals_failure = (y_failure_flat-X_flat@self.linear_coeff).reshape(y_failure.shape)

    def fit_residuals(self,param = {}):
        """
        Finaly this function takes the residuals computed in the prevous step and classify the failure and no-failure case.
        :param param: Dictionnary of every possible parameters for the learning step: model_residuals, train_size.
        :type param: dict
        """
        _default_model = SVC(kernel='rbf')
        model = param.get("model_residuals", _default_model)
        train_size = param.get("train_size", 0.8)
        X = np.concat([self.residuals_train,self.residuals_failure])
        y = np.concat([[0]*self.residuals_train.shape[0],[1]*self.residuals_failure.shape[0]])
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1-train_size, shuffle=True,random_state=0)

        self.preprocess = StandardScaler()
        self.preprocess.fit(X_train)
        X_train = self.preprocess.transform(X_train)
        X_val = self.preprocess.transform(X_val)

        model.fit(X_train,y_train)
        self.residual_model = model

        return {"accuracy_train" : accuracy_score(y_train,model.predict(X_train)),
                "f1_score_train" : f1_score(y_train,model.predict(X_train)),
                "accuracy_val" : accuracy_score(y_val,model.predict(X_val)),
                "f1_score_val" : f1_score(y_val,model.predict(X_val))
                }

    def fit(self,mutlidata,param = {}):
        self.fit_quantiles(mutlidata,param)
        self.fit_linear(param)
        return self.fit_residuals(param)

    def predict_quantiles(self,multidata):

        """
        This function computes the transform data and load the intermediate result for the test set.
        :param multidata: This object contains all information about the sites, if the data haven't been diluated yet 
        and self.ndil is given the function start by diluating the data. 
        :type multidata: MultiDataHandler
        :param param: Dictionnary of every possible parameters for the learning step: weight_quantiles, quantiles, 
        solver_quantiles, num_harmonics.
        :type param: dict
        """

        if multidata.dil_mat is not None and self.ndil is not None and multidata.ndil() != self.ndil:
            raise ValueError("ndil between the argument and the model is different")
        elif multidata.dil_mat is None and self.ndil is not None :
            multidata.dilate(ndil = self.ndil)
        elif multidata.dil_mat is not None and self.ndil is None :
            self.ndil = multidata.ndil()
        elif multidata.dil_mat is None and self.ndil is None :
            raise ValueError('No ndil given')
        
        if multidata.target is not None and self.target is not None and multidata.target != self.target:
            raise ValueError("target between the argument and the model is different")
        elif multidata.target is None and self.target is None :
            raise ValueError('No target given')
        
        for key in multidata.dil_mat :
            full_array,start_index = full_signal(multidata.common_days,self.start_train,multidata.dil_mat[key],self.ndil)
            time_index = np.arange(start_index, len(full_array) + start_index)
            self.quantile_test[key] = reconstruct_signal(self.spqs[key].transform(full_array,y = time_index),multidata.common_days,self.ndil)

    def predict_linear(self):
        """
        This function compute the linear regression part. It creates the lagged features and compute the residuals for the test set.
        :param param: Dictionnary of every possible parameters for the learning step: nlag, num_basis, weight_linear
        lambda_range, num_split.
        :type param: dict
        """
        X = np.array([self.quantile_test[key] for key in self.quantile_train if key != self.target])
        y = self.quantile_test[self.target]
        #Numerical stability
        X = np.clip(X,-4,4)
        y = np.clip(y,-4,4)
        X[np.isnan(X)] = 0
        y[np.isnan(y)] = 0

        X,y,_ = form_xy(X,y,y,self.nlag)
        X_flat,y_flat = reshape_for_cheb(X,y,self.num_basis)
        self.residuals_test = (y_flat-X_flat@self.linear_coeff).reshape(y.shape)

    def predict_residuals(self):
        """
        Finaly this function takes the residuals computed in the prevous step and predict the different case.
        :param param: Dictionnary of every possible parameters for the learning step: model_residuals, train_size.
        :type param: dict
        """
        X = np.concat([self.residuals_train,self.residuals_failure])
        X = self.preprocess.transform(X)
        y = self.residual_model.predict(X)
        return y

    def predict(self,multidata):
        self.predict_quantiles(multidata)
        self.predict_linear()
        return self.predict_residuals()

    def test_quantiles(self,multidata):
        """
        This function compute the first part of the model and compute the intermediate results for the test set and failure scenario
        :param multidata: This object contains all information about the sites, if the data haven't been diluated yet 
        and self.ndil is given the function start by diluating the data. 
        :type multidata: MultiDataHandler
        """

        if multidata.dil_mat is not None and self.ndil is not None and multidata.ndil() != self.ndil:
            raise ValueError("ndil between the argument and the model is different")
        elif multidata.dil_mat is None and self.ndil is not None :
            multidata.dilate(ndil = self.ndil)
        elif multidata.dil_mat is not None and self.ndil is None :
            self.ndil = multidata.ndil()
        elif multidata.dil_mat is None and self.ndil is None :
            raise ValueError('No ndil given')
        
        if multidata.target is not None and self.target is not None and multidata.target != self.target:
            raise ValueError("target between the argument and the model is different")
        elif multidata.target is None and self.target is not None :
            multidata.generate_failure(self.target)
            warnings.warn(
                "No failure scenario detected — calling multidata.generate_failure automatically with default parameters.",
                category=UserWarning
            )
        elif multidata.target is not None and self.target is None :
            self.target = multidata.target
        elif multidata.target is None and self.target is None :
            raise ValueError('No target given')
        
        self.quantile_test = {}
        for key in multidata.dil_mat :
            full_array,start_index = full_signal(multidata.common_days,self.start_train,multidata.dil_mat[key],self.ndil)
            time_index = np.arange(start_index, len(full_array) + start_index)
            self.quantile_test[key] = reconstruct_signal(self.spqs[key].transform(full_array,y = time_index),multidata.common_days,self.ndil)
        full_failure,start_index = full_signal(multidata.common_days,self.start_train,multidata.failure_mat,self.ndil)
        time_index = np.arange(start_index, len(full_array) + start_index)
        self.quantile_failure_test = reconstruct_signal(self.spqs[self.target].transform(full_failure,y = time_index),multidata.common_days,self.ndil)

    def test_linear(self):
        """
        This function computes the linear regression part. 
        It creates the lagged features and compute the residuals for the failure/nofailure scenrario.
        """
        X = np.array([self.quantile_test[key] for key in self.quantile_test if key != self.target])
        y = self.quantile_test[self.target]
        y_failure = self.quantile_failure_test
        #Numerical stability
        X = np.clip(X,-4,4)
        y = np.clip(y,-4,4)
        y_failure = np.clip(y_failure,-4,4)
        X[np.isnan(X)] = 0
        y[np.isnan(y)] = 0
        y_failure[np.isnan(y_failure)] = 0

        X,y,y_failure = form_xy(X,y,y_failure,self.nlag)
        X_flat,y_flat = reshape_for_cheb(X,y,self.num_basis)
        _,y_failure_flat = reshape_for_cheb(X,y_failure,self.num_basis)
        self.residuals_test = (y_flat-X_flat@self.linear_coeff).reshape(y.shape)
        self.residuals_failure_test = (y_failure_flat-X_flat@self.linear_coeff).reshape(y_failure.shape)

    def test_resiudals(self):
        """
        Finaly this function takes the residuals computed in the prevous step and classify the failure and no-failure case.
        """
        X = np.concat([self.residuals_test,self.residuals_failure_test])
        y = np.concat([[0]*self.residuals_test.shape[0],[1]*self.residuals_failure_test.shape[0]])

        X = self.preprocess.transform(X)

        return {"accuracy_test" : accuracy_score(y,self.residual_model.predict(X)),
                "f1_score_test" : f1_score(y,self.residual_model.predict(X))}

    def test(self,multidata):
        self.test_quantiles(multidata)
        self.test_linear()
        return self.test_resiudals()

    def display(self,idx, multidata = None):
        if multidata is None :
            _, ax = plt.subplots(figsize=(10, 4))

            residual_nofail = self.residuals_train[idx,:]
            residual_fail = self.residuals_failure[idx,:]
            x = np.arange(residual_fail.shape[0])
            
            ax.plot(x, residual_fail, label="Residuals with outage", color="blue")
            ax.plot(x, residual_nofail, label="Residuals no outage",
                    color="blue", linestyle="dotted", linewidth=1)
            ax.set_ylabel("Power")
            lines1, labels1 = ax.get_legend_handles_labels()
            ax.legend(lines1, labels1, loc="upper right")
            ax.grid(True)
            plt.tight_layout()
            plt.show()
        
        else :
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.3})

            residual_nofail = self.residuals_train[idx, :]
            residual_fail = self.residuals_failure[idx, :]
            x = np.arange(self.nlag, self.ndil - self.nlag-1)

            ax2.plot(x, residual_fail, label="Residuals with outage", color="blue")
            ax2.plot(x, residual_nofail, label="Residuals no outage", color="blue", linestyle="dotted", linewidth=1)

            ax2.set_ylabel("Power")
            ax2.legend(loc="upper right")

            multidata.display(idx,site = self.target,ax = ax1)
            ax2.grid(True)
            plt.tight_layout()
            plt.show()

    def coef_linear(self):
        nval = self.residuals_train.shape[1]
        c = make_cheb_basis(np.arange(nval), self.num_basis, (0, nval))
        theta_reshaped = self.linear_coeff.reshape(self.linear_coeff.shape[0]//self.num_basis, self.num_basis)
        coeffs = np.dot(c, theta_reshaped.T)
        return coeffs


def save(file, obj): 
    """
    Save a QLinear object to a file using joblib with gzip compression.
    
    Parameters:
    -----------
    file : str or file-like object
        The file path where the object should be saved.
    obj : QLinear object
        The object to be saved.
    """
    # Ensure the directory exists
    dir_name = os.path.dirname(file)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    # Save the object with joblib using gzip compression
    joblib.dump(obj, file, compress=('gzip', 3))

def load(file) -> QLinear:
    """
    Load a QLinear object from a file using joblib.
    
    Parameters:
    -----------
    file : str or file-like object
        The file path from which the object should be loaded.
    """
    obj = joblib.load(file)
    return obj



        

        
        