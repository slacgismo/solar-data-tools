from solardatatools import DataHandler

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from dask import delayed


def divide_df(data_frame,datetime_col = None):
    """
    This function divide a dataframe with a single column timeIndex and multiple columns for the values
    into dataframes with a uniuqe column timeIndex and unique column values.
    """
    if datetime_col is not None:
        if datetime_col not in data_frame.columns:
            raise ValueError(f"'{datetime_col}' is not a column in the DataFrame.")
        df = data_frame.set_index(pd.to_datetime(data_frame[datetime_col]))
        df = df.drop(columns=[datetime_col])
    else:
        if not isinstance(data_frame.index, pd.DatetimeIndex):
            raise ValueError("Data frame must have a DatetimeIndex or the user must set the datetime_col kwarg.")
        df = data_frame

    result = {col: df[[col]].copy() for col in df.columns}
    return result

def common_days(dh_dict):
    """
    This function takes a dictionary of DataHandlers and returns a dictionary of boolean NumPy arrays,
    where each value indicates whether the corresponding day is a common good day.
    """
    common_good_days = None  
    for dh in dh_dict.values():
        good_days = set(dh.day_index[dh.daily_flags.no_errors])
        if common_good_days is None:
            common_good_days = good_days
        else:
            common_good_days &= good_days 
    result = {}
    for name, dh in dh_dict.items():
        result[name] = np.isin(
            dh.day_index.astype('datetime64[ns]'),
            np.array(list(common_good_days)).astype('datetime64[ns]')
        )
    return result

@delayed
def fit_dask(spq,x):
    spq.fit(x)

def full_signal(day_list,start_day,signal,ndil):
    """
    This function construct the full signal with np.nan when the days was labelled bad. Moreover, it returns the start of time index
    for SmoothPeriodicQuantiles.transform when apply with different days.
    """
    assert signal.shape[1] == len(day_list), "Data length or days length is incorrect"
    if not isinstance(start_day, datetime):
        start_day = pd.to_datetime(start_day)
    day_list = pd.to_datetime(day_list)
    day_num = [(d.date() - day_list[0].date()).days for d in day_list]
    max_day = day_num[-1]+1
    full_result = np.full((max_day * ndil,), np.nan)
    for d, day in enumerate(day_num):
        full_result[day * ndil:(day + 1) * ndil] = signal[:, d]
    return full_result,ndil*(day_list[0].date()-start_day.date()).days

def reconstruct_signal(full_signal,day_list,ndil):
    """
    This function is the inverse of the previous one.s
    """
    day_list = pd.to_datetime(day_list)
    day_num = [(d.date() - day_list[0].date()).days for d in day_list]
    matrix_res = np.zeros((ndil,len(day_list)))
    for d, day in enumerate(day_num):
        matrix_res[:,d] = full_signal[day * ndil:(day + 1) * ndil]
    return matrix_res

def form_xy(X,Y,Y_failure, nlags):
    """
    This function construct from X and Y a matrix of features with lagged features, it takes nlag before and nlag after
    """
    nsites, ndil, ndays = X.shape
    nvals_per_day = ndil - 2*nlags - 1 # last value each day is not relevant
    nfeatures = nsites * (2*nlags+1) + 1
    x = np.zeros((ndays, nvals_per_day, nfeatures))
    y = np.zeros((ndays, nvals_per_day))
    y_failure = np.zeros((ndays, nvals_per_day))
    x[:, :, 0] = 1 # bias term
    for i in range(ndays):
        for j in range(nvals_per_day):
            idx = 1
            for siteidx in range(nsites):
                for k in range(-nlags, nlags+1):
                    x[i, j, idx] = X[siteidx,j+k+nlags,i]
                    idx += 1
            y[i,j] = Y[j+nlags,i]
            y_failure[i,j] = Y_failure[j+nlags,i]
    return x,y,y_failure

def make_cheb_basis(xs, num_basis, domain):
    xs = np.atleast_1d(xs)
    basis_matrix = np.zeros((len(xs), num_basis))
    chebyshev = np.polynomial.chebyshev.Chebyshev(np.ones(num_basis))
    for _ix in range(num_basis):
        basis_generator = chebyshev.basis(
            _ix, window=(-1, 1), domain=domain
        )
        basis_matrix[:, _ix] = basis_generator(xs)
    return basis_matrix

def reshape_for_cheb(x_mat, y_mat, num_basis):
    """
    This function take X and Y and return the flatten vector of X@B where B is the chebychev basis. 
    """
    ndays, nvals_per_day, nfeatures = x_mat.shape
    c = make_cheb_basis(np.arange(nvals_per_day), num_basis, (0, nvals_per_day))
    c_reshaped = c.reshape(1, nvals_per_day, 1, num_basis)
    x_mat_reshaped = x_mat.reshape(ndays, nvals_per_day, nfeatures, 1)
    x_flat = (c_reshaped * x_mat_reshaped).reshape(ndays * nvals_per_day, nfeatures * num_basis)
    y_flat = y_mat.ravel()
    return x_flat, y_flat

def weighted_ridge_regression(X, y, weights, lbda):
    # Solve the ridge regression problem with different weights
    lambda_weights = lbda * np.sqrt(weights)
    _, n_features = X.shape
    sqrt_lambda = np.diag(lambda_weights)
    X_aug = np.vstack([X, sqrt_lambda])
    y_aug = np.hstack([y, np.zeros(n_features)])
    theta, _, _, _ = np.linalg.lstsq(X_aug, y_aug, rcond=None)
    return theta

def optimal_weighted_regression(X, y, weights, lambda_values, n_splits=5, random_state=0):
    """
    Select the best lambda with K-fold CV for the righe regression
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    mse_values = np.zeros(len(lambda_values))

    for i, lbda in enumerate(lambda_values):
        mses = []
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            theta = weighted_ridge_regression(X_train, y_train, weights, lbda)
            y_pred = X_val @ theta
            mses.append(mean_squared_error(y_val, y_pred))
        
        mse_values[i] = np.mean(mses)

    optimal_lambda = lambda_values[np.argmin(mse_values)]
    optimal_theta = weighted_ridge_regression(X, y, weights, optimal_lambda)

    return optimal_lambda, optimal_theta








