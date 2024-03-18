""" PVPRO Post Processing Module

This module contains a class that takes in the output dataframe of PVPRO and
contains methods to process the dataset, perform signal decompositions to model
degradation trends, analyze how well the models fit, and visualize the trends.

"""

import numpy as np

import pandas as pd
import cvxpy as cp

import matplotlib.pyplot as plt

from collections import defaultdict
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from time import time
from solardatatools.utilities import progress


class PVPROPostProcessor:
    """This is a class to process a dataset, perform signal decomposition to
    model degradation trends, and analyze and visualize the resulting trends.

    :param file_name: Name of the data file to be imported (must be .csv)
    :type file_name: str
    :param period: How many data points in the file make up a full year of data
    (for instance, in 5-day interval data, the period is 73)
    :type period: int
    :param index_col: column in the input data to be used as the index
    :type index_col: int, optional
    :param dates: a list of integer column indices to be parsed as dates
    :type dates: list, optional
    :param df_prep: A T/F switch to determine whether the dataframe preparation
    steps should be performed
    :type df_prep: bool, optional
    :param include: Input to include a key term in selected column names
    :type include: str, optional
    :param exclude: Input to exclude a key term from selected column names
    :type exclude: str, optional
    :param verbose: A T/F switch to show percentage of data points on the
    boundaries
    :type verbose: bool, optional
    :param bp: A T/F switch to choose whether to look for data points on the
    boundaries
    "type bp: bool, optional
    """

    def __init__(
        self,
        file_name,
        period,
        index_col=0,
        dates=None,
        df_prep=True,
        include=None,
        exclude=None,
        verbose=False,
        bp=True,
    ):
        """Imports dats,  assigns index column and date columns, performs
        optional dataframe preparation steps, does preprocessing on the data
        frame, creates a 'preprocessed' data frame to be used in future
        analysis, and initializes other attributes of the class.
        """

        self.param_dict = {
            "photocurrent": (0.01, 10, 0.01),
            "saturation_current": (0, 1 * 10 ** (-6), 5 * 10 ** (-12)),
            "resistance_series": (0.1, 2, 0.05),
            "resistance_shunt": (100, 500, 1),
            "i_sc": (-np.inf, np.inf, 1),
            "v_oc": (-np.inf, np.inf, 1),
            "i_mp": (-np.inf, np.inf, 1),
            "v_mp": (-np.inf, np.inf, 1),
            "p_mp": (-np.inf, np.inf, 1),
        }

        if dates is None:
            dates = [0]
        else:
            dates = dates

        def default_val():
            return None

        self.df = pd.read_csv(file_name, index_col=index_col, parse_dates=dates)
        self.period = int(period)

        # dataframe preparation (column selection, time index correction)
        if df_prep is True:
            self.data_setup(include=include, exclude=exclude)
            self.df_ds = self.df_ds
        else:
            self.df_ds = self.df

        # processing steps
        if bp == True:
            self.boundary_points(verbose=verbose)
            self.boundary_to_nan()
        df_scaled = self.scale_max_1()
        df_p = self.ln_df()

        # attributes
        self.df_p = df_p
        self.df_error = None
        self.df_error_avg = None
        self.scaled_data = defaultdict(default_val)
        self.descaled_data = defaultdict(default_val)
        self.processed_result = defaultdict(default_val)
        self.df_x1 = None
        self.df_x2 = None
        self.df_x3 = None
        self.df_x4 = None
        self.df_x5 = None
        self.df_cs = None

    ###########################################################################
    # Processing functions
    ###########################################################################

    def data_setup(self, include=None, exclude=None):
        """Adjusts time index so that there are equal intervals and isolates
        columns of interest. Creates dataframe self.df_ds with the selected
        columns and adjusted time index.

        :param include: Input to include a key term in selected column names
        :type include: str, optional
        :param exclude: Input to exclude a key term from selected column names
        :type exclude: str, optional
        """

        # adjusting the time series
        time_delta, count = mode(np.diff(self.df.index), keepdims=True)
        freq = int(time_delta[0] / np.timedelta64(1, "s"))
        new_index = pd.date_range(
            start=self.df.index[0], end=self.df.index[-1], freq="{}s".format(freq)
        )
        self.df = self.df.reindex(index=new_index)

        # column selection
        rule1 = lambda x: (
            "photocurrent" in x
            or "saturation" in x
            or "resistance" in x
            or "i_sc" in x
            or "v_oc" in x
            or "mp" in x
        )

        if include is not None:
            rule2 = lambda x: include in x
            if exclude is not None:
                rule3 = lambda x: exclude not in x
                cols = [
                    c for c in self.df.columns if rule1(c) and rule2(c) and rule3(c)
                ]
            else:
                cols = [c for c in self.df.columns if rule1(c) and rule2(c)]
        else:
            if exclude is not None:
                rule3 = lambda x: exclude not in x
                cols = [c for c in self.df.columns if rule1(c) and rule3(c)]
            else:
                cols = [c for c in self.df.columns if rule1(c)]

        df_ds = self.df.loc[:, cols]
        df_ds = df_ds.reindex(index=new_index)

        self.df_ds = df_ds

    def boundary_points(self, verbose=False):
        """Determines indices of points on the boundary to a tolerance
        determined in param_dict. Creates a list of indices where data points
        are on the boundary in any of the system parameters.

        :param verbose: A T/F switch to show percentage of data points on the
        boundaries
        :type verbose: bool, optional
        """

        bounds = self.param_dict
        indices = []
        df = self.df_ds
        cond = "".join(df.columns)
        bounded_params = [k for k in bounds.keys() if k in cond]

        for name in bounded_params:
            label_selection = [c for c in df.columns if name in c]
            label = label_selection[0]

            if bounds[name][0] == -np.inf:
                lbpoints = []
            else:
                lbpoints = np.arange(len(df[label]))[
                    df[label] <= bounds[name][0] + bounds[name][2]
                ]

            if bounds[name][1] == np.inf:
                ubpoints = []
            else:
                ubpoints = np.arange(len(df[label]))[
                    df[label] >= bounds[name][1] - bounds[name][2]
                ]

            indices = np.concatenate((indices, lbpoints, ubpoints))

        indices = np.unique(indices).astype(int)
        self.bound_indices = indices

        if verbose is True:
            print("Percent on boundaries:", (100 * (len(indices) / len(df.index))), "%")
        else:
            pass

    def boundary_to_nan(self):
        """Makes all points in the dataframe at boundary point indices be nan."""

        self.df_ds.iloc[self.bound_indices, :] = np.nan

    def scale_max_1(self):
        """Scales a dataframe to have max value 1.

        :return: self.scaler, which saves all of the values involved in scaling
        the data frame
        :rtype: array
        """

        scaler = MaxAbsScaler()
        df_scaled = scaler.fit_transform(self.df_ds.to_numpy())
        df_scaled = pd.DataFrame(
            df_scaled, columns=self.df_ds.columns, index=self.df_ds.index
        )

        self.scaler = scaler
        return df_scaled

    def ln_df(self):
        """Takes the natural log of the scaled dataframe and makes all inf
        values nan.

        :return: df_l, a scaled to max 1 data frame in log space
        "rtype: Pandas DataFrame
        """

        df_scaled = self.scale_max_1()
        cond = df_scaled > 0
        df_l = np.log(df_scaled[cond])

        return df_l

    def view_minmax(self, df):
        """Prints the minimum and maximum values for each column in the
        dataframe.
        """

        for label, values in df.items():
            print(label)
            print(min(df[label]))
            print(max(df[label]))

    ###########################################################################
    # Signal decomposition
    ###########################################################################

    def optimize(
        self,
        label,
        lambda4,
        lambda5,
        model,
        lambda2=0.001,
        verbose=False,
        known=None,
        solver="Default",
    ):
        """Runs an optimization problem to perform a 5-component signal
        decomposition using cvxpy on one parameter of the PV system. Creates
        two data frames of the resulting components and a composed signal of
        the noiseless components. One data frame is in the scaled log space and
        the other is in the original space. These resulting data frames can be
        accessed in the self.scaled_data and self.descaled_data dictionaries
        using the key (label + '_' + model).

        :param label: Column name that indicates which system parameter is
        being optimized.
        :type label: str
        :param lambda4: Weight which determines the strength of smoothing on
        the periodic component
        :type lambda4: float
        :param lambda5: Weight which determines the strength of smoothing on
        the degradation component
        :type lambda5: float
        :param model: Names the model to use for the degradation component, can
        be 'linear', 'monotonic',
        'smooth_monotonic', or 'piecewise_linear'
        :type model: str
        :param lambda2: Weight on the Laplacian noise term
        :type lambda2: float, optional
        :param verbose: T/F switch to determine whether cvxpy prints a verbose
        output of the solve
        :type verbose: bool, optional
        :param known: Option to input a mask on the data inputted into the
        solver
        :type known: bool mask, optional
        :param solver: Indicates which solver cvxpy should call to perform the
        optimization problem
        :type solver: str, optional
        """

        acceptable_models = [
            "linear",
            "monotonic",
            "smooth_monotonic",
            "piecewise_linear",
        ]

        if model not in acceptable_models:
            print("check model entry")

        # initializing data and characteristic values
        data = self.df_p[label]
        y = self.df_p[label].values
        T = len(y)
        p = self.period

        # applying mask if applicable
        if known is None:
            known = ~np.isnan(y)
        else:
            known = np.logical_and(known, ~np.isnan(y))

        if "series" in label:
            decreasing = False
        else:
            decreasing = True

        # components
        x1 = cp.Variable(T)
        x2 = cp.Variable(T)
        x3 = cp.Variable(T)
        x4 = cp.Variable(T)
        x5 = cp.Variable(T)

        # weights
        lambda_2 = cp.Parameter(value=lambda2, nonneg=True)
        lambda_4 = cp.Parameter(value=lambda4, nonneg=True)
        lambda_5 = cp.Parameter(value=lambda5, nonneg=True)

        # initial cost function to be minimized and initial constraints
        cost = (
            (1 / T) * cp.sum_squares(x1)
            + lambda_2 * cp.norm1(x2)
            + lambda_4 * cp.sum_squares(cp.diff(x4, k=2))
        )

        constraints = [
            y[known] == (x1 + x2 + x3 + x4 + x5)[known],
            cp.diff(x3, k=1) == 0,
            cp.sum(x4[:p]) == 0,
            x4[p:] == x4[:-p],
            x5[0] == 0,
        ]

        # additional costs and conditions for all the model types
        if model == "linear":
            constraints.append(cp.diff(x5, k=2) == 0)
        else:
            if decreasing == True:
                constraints.append(cp.diff(x5, k=1) <= 0)
            else:
                constraints.append(cp.diff(x5, k=1) >= 0)

            if model == "smooth_monotonic":
                cost += lambda_5 * cp.sum_squares(cp.diff(x5, k=2))

            elif model == "piecewise_linear":
                cost += lambda_5 * cp.norm1(cp.diff(x5, k=2))

        # setting up the problem
        obj = cp.Minimize(cost)
        prob = cp.Problem(obj, constraints)
        if solver == "Default":
            solver = "OSQP"

        if solver == "OSQP":
            prob.solve(
                solver=solver,
                verbose=verbose,
                eps_prim_inf=1 * 10 ** (-6),
                eps_dual_inf=1 * 10 ** (-6),
                eps_rel=1 * 10 ** (-6),
                eps_abs=1 * 10 ** (-6),
            )
        else:
            prob.solve(solver=solver, verbose=verbose)

        # resulting components
        df_components = pd.DataFrame(
            index=self.df_p.index,
            data={
                "x1": x1.value,
                "x2": x2.value,
                "x3": x3.value,
                "x4": x4.value,
                "x5": x5.value,
                "composed_signal": (x3.value + x4.value + x5.value),
            },
        )

        self.scaled_data[label + "_" + model] = df_components

        # exponentiating the components and undoing the scaling
        x1 = np.exp(df_components["x1"].values)
        x2 = np.exp(df_components["x2"].values)
        x3 = np.exp(df_components["x3"].values)
        x4 = np.exp(df_components["x4"].values)
        x5 = np.exp(df_components["x5"].values)

        ind = self.df_ds.columns.get_loc(label)

        max_val = self.scaler.scale_[ind]
        x3 = x3 * max_val

        df_descaled = pd.DataFrame(
            index=self.df_ds.index,
            data={
                "x1": x1,
                "x2": x2,
                "x3": x3,
                "x4": x4,
                "x5": x5,
                "composed_signal": (x3 * x4 * x5),
            },
        )

        self.descaled_data[label + "_" + model] = df_descaled
        self.processed_result[label + "_" + model] = df_descaled["x5"]

    def analyze(
        self,
        label,
        lambda2=0.001,
        lambda4=0.1,
        lambda5=1,
        model="smooth_monotonic",
        verbose=False,
        known=None,
        solver="Default",
    ):
        """Performs optimize() with default values. All parameters and outputs
        are the same as those in optimize().

        :param label: Column name that indicates which system parameter is
        being optimized.
        :type label: str
        :param lambda2: Weight on the Laplacian noise term
        :type lambda2: float, optional
        :param lambda4: Weight which determines the strength of smoothing on
        the periodic component
        :type lambda4: float
        :param lambda5: Weight which determines the strength of smoothing on
        the degradation component
        :type lambda5: float
        :param model: Names the model to use for the degradation component, can
        be 'linear', 'monotonic',
        'smooth_monotonic', or 'piecewise_linear'
        :type model: str
        :param verbose: T/F switch to determine whether cvxpy prints a verbose
        output of the solve
        :type verbose: bool, optional
        :param known: Option to input a mask on the data inputted into the
        solver
        :type known: bool mask, optional
        :param solver: Indicates which solver cvxpy should call to perform the
        optimization problem
        :type solver: str, optional
        """

        self.optimize(
            label,
            lambda4,
            lambda5,
            model,
            lambda2=lambda2,
            verbose=verbose,
            known=known,
            solver=solver,
        )

    def retreive_result(self, label, model="smooth_monotonic"):
        if self.processed_result[label + "_" + model] is None:
            opt = input("No data entry, would you like to run optimization? (y/n)\n")

            if opt == "y":
                self.analyze(label, model=model)
            elif opt == "n":
                pass
            else:
                pass
        result = self.processed_result[label + "_" + model]
        return result

    def sd_result_dfs(
        self,
        lambda2=0.001,
        lambda4=0.1,
        lambda5=1,
        model="smooth_monotonic",
        known=None,
        solver="Default",
    ):
        """Creates six new data frames containing the six signal decomposition
        components produced by performing optimize() with the indicated inputs
        over all system parameters. One data frame holds one component for all
        system parameters.

        :param lambda2: Weight on the Laplacian noise term
        :type lambda2: float, optional
        :param lambda4: Weight which determines the strength of smoothing on
        the periodic component
        :type lambda4: float
        :param lambda5: Weight which determines the strength of smoothing on
        the degradation component
        :type lambda5: float
        :param model: Names the model to use for the degradation component, can
        be 'linear', 'monotonic',
        'smooth_monotonic', or 'piecewise_linear'
        :type model: str
        :param known: Option to input a mask on the data inputted into the
        solver
        :type known: bool mask, optional
        :param solver: Indicates which solver cvxpy should call to perform the
        optimization problem
        :type solver: str, optional
        """

        # initializing data frames
        df_x1 = pd.DataFrame()
        df_x2 = pd.DataFrame()
        df_x3 = pd.DataFrame()
        df_x4 = pd.DataFrame()
        df_x5 = pd.DataFrame()
        df_cs = pd.DataFrame()

        # solving for all parameters and recording components in their
        # respective data frames
        for column in self.df_p.columns:
            self.optimize(
                column,
                lambda4,
                lambda5,
                model,
                lambda2=lambda2,
                known=known,
                solver=solver,
            )

            x1_data = pd.Series(
                data=self.descaled_data[column + "_" + model]["x1"], name=column
            )
            x2_data = pd.Series(
                data=self.descaled_data[column + "_" + model]["x2"], name=column
            )
            x3_data = pd.Series(
                data=self.descaled_data[column + "_" + model]["x3"], name=column
            )
            x4_data = pd.Series(
                data=self.descaled_data[column + "_" + model]["x4"], name=column
            )
            x5_data = pd.Series(
                data=self.descaled_data[column + "_" + model]["x5"], name=column
            )
            cs_data = pd.Series(
                data=self.descaled_data[column + "_" + model]["composed_signal"],
                name=column,
            )

            df_x1 = pd.concat([df_x1, x1_data], axis=1)
            df_x2 = pd.concat([df_x2, x2_data], axis=1)
            df_x3 = pd.concat([df_x3, x3_data], axis=1)
            df_x4 = pd.concat([df_x4, x4_data], axis=1)
            df_x5 = pd.concat([df_x5, x5_data], axis=1)
            df_cs = pd.concat([df_cs, cs_data], axis=1)

        self.df_x1 = df_x1
        self.df_x2 = df_x2
        self.df_x3 = df_x3
        self.df_x4 = df_x4
        self.df_x5 = df_x5
        self.df_cs = df_cs

    ###########################################################################
    # Plotting functions
    ###########################################################################

    def plot_sd_space(self, label, model="smooth_monotonic"):
        """Plots the SD of one system parameter in the scaled log space.

        :param label: Column name that indicates which system parameter is
        being optimized
        :type label: str
        :param model: Names the model to use for the degradation component, can
        be 'linear', 'monotonic',
        'smooth_monotonic', or 'piecewise_linear'
        :type model: str, optional
        :return: Plot of the SD in scaled log space
        :rtype: figure
        """

        if model == "smooth_monotonic":
            model_title = "Smooth Monotonic"
        if model == "piecewise_linear":
            model_title = "Piecewise Linear"
        if model == "linear":
            model_title = "Linear"
        if model == "monotonic":
            model_title = "Monotonic"

        if self.scaled_data[label + "_" + model] is None:
            opt = input("No data entry, would you like to run optimization? (y/n)\n")

            if opt == "y":
                self.analyze(label, model=model)
                self.plot_sd_space(label, model=model)

            if opt == "n":
                pass

        else:
            components = self.scaled_data[label + "_" + model]

            titles = [
                "Residual",
                "Laplacian Noise",
                "Bias",
                "Periodic",
                model_title,
                "Composed Signal",
            ]

            plt.figure(figsize=(6, 20))
            counter = 0

            for col, values in components.items():
                plt.subplot(611 + counter)
                if (
                    titles[counter] == "Residual"
                    or titles[counter] == "Laplacian Noise"
                ):
                    plt.scatter(self.df_p.index, components[col])
                elif titles[counter] == "Composed Signal":
                    plt.scatter(self.df_p.index, self.df_p[label], c="orange")
                    plt.plot(self.df_p.index, components[col])
                    plt.xlabel("Time")
                elif titles[counter] == "Bias":
                    plt.plot(self.df_p.index, components[col])
                    plt.ylim(
                        (
                            components[col][0] - 0.1 * components[col][0],
                            components[col][0] + 0.1 * components[col][0],
                        )
                    )
                else:
                    plt.plot(self.df_p.index, components[col])

                plt.title(titles[counter])
                plt.xticks(rotation=45)

                counter += 1

            plt.tight_layout()
            return plt.gcf()

    def plot_original_space(self, label, model="smooth_monotonic"):
        """Plots the SD of one system parameter in the original space.

        :param label: Column name that indicates which system parameter is
        being optimized
        :type label: str
        :param model: Names the model to use for the degradation component, can
        be 'linear', 'monotonic',
        'smooth_monotonic', or 'piecewise_linear'
        :type model: str, optional
        :return: Plot of the SD in original space
        :rtype: figure
        """

        if model == "smooth_monotonic":
            model_title = "Smooth Monotonic"
        if model == "piecewise_linear":
            model_title = "Piecewise Linear"
        if model == "linear":
            model_title = "Linear"
        if model == "monotonic":
            model_title = "Monotonic"

        if self.descaled_data[label + "_" + model] is None:
            opt = input("No data entry, would you like to run optimization? (y/n)\n")

            if opt == "y":
                self.analyze(label, model=model)
                self.plot_sd_space(label, model=model)

            if opt == "n":
                pass

        else:
            data = self.df_ds[label]
            components = self.descaled_data[label + "_" + model]

            titles = [
                "Residual",
                "Laplacian Noise",
                "Bias",
                "Periodic",
                model_title,
                "Composed Signal",
            ]

            plt.figure(figsize=(6, 20))
            counter = 0

            for col, values in components.items():
                plt.subplot(611 + counter)
                if col == "x1" or col == "x2":
                    plt.scatter(self.df_ds.index, components[col])
                elif col == "composed_signal":
                    plt.scatter(self.df_ds.index, self.df_ds[label], c="orange")
                    plt.plot(self.df_ds.index, components[col])
                    plt.xlabel("Time")
                elif col == "x3":
                    plt.plot(self.df_ds.index, components[col])
                    plt.ylim(
                        (
                            components[col][0] - 0.1 * components[col][0],
                            components[col][0] + 0.1 * components[col][0],
                        )
                    )
                else:
                    plt.plot(self.df_ds.index, components[col])

                plt.title(titles[counter])
                plt.xticks(rotation=45)

                counter += 1

            plt.tight_layout()
            return plt.gcf()

    ###########################################################################
    # Error calculations
    ###########################################################################

    def error_analysis(
        self, lambda4, lambda5, num_runs, lambda2=0.001, solver="Default"
    ):
        """Calculates the holdout error, looping over system parameters,
        models, cost function weights, and
        number of repetitions. Creates a data frame with error results for each
        iteration and another which averages
        over all runs for each unique set of inputs.

        :param lambda4: A list of periodic component weight values to loop over
        :type lambda4: list
        :param lambda5: A list of degradation component weight values to loop
        over
        :type lambda5: list
        :param num_runs: Number of runs to perform; more runs yields more
        generalizable results at the cost of time
        :type num_runs: int
        :param lambda2: Laplacian noise weight term value
        :type lambda2: float, optional
        :param solver: Indicates which solver cvxpy should call to perform the
        optimization problem
        :type solver: str, optional
        """

        ti = time()
        period = self.period
        lambda_2 = lambda2
        lambda_4_values = np.atleast_1d(lambda4)
        lambda_5_values = np.atleast_1d(lambda5)
        runs = np.arange(num_runs)
        models = ["linear", "monotonic", "smooth_monotonic", "piecewise_linear"]
        cols = [
            "system_parameter",
            "run_number",
            "degradation_model",
            "lambda_2_val",
            "lambda_4_val",
            "lambda_5_val",
            "mean_sq_error",
        ]
        num_rows = (
            (len(lambda_5_values) + 1)
            * len(lambda_4_values)
            * 2
            * len(runs)
            * len(self.df_p.columns)
        )
        df_error = pd.DataFrame(columns=cols, index=np.arange(num_rows))

        counter = 0
        progress(counter, num_rows)

        for label, values in self.df_p.items():
            data = self.df_p[label]
            known = ~np.isnan(data)
            T = len(data)
            indices = np.arange(T)
            indices = indices[known]

            for r in runs:
                train_inds, test_inds = train_test_split(indices, test_size=0.1)
                train = np.zeros(T, dtype=bool)
                test = np.zeros(T, dtype=bool)
                train[train_inds] = True
                test[test_inds] = True

                test_data = data[test]

                for m_type in models:
                    for l4_val in lambda_4_values:
                        if m_type == "smooth_monotonic" or m_type == "piecewise_linear":
                            l5_iter = lambda_5_values

                        else:
                            l5_iter = [1]

                        for l5_val in l5_iter:
                            if solver == "Default":
                                self.optimize(
                                    label,
                                    l4_val,
                                    l5_val,
                                    m_type,
                                    lambda2=lambda_2,
                                    solver=solver,
                                    known=train,
                                )
                            else:
                                self.optimize(
                                    label,
                                    l4_val,
                                    l5_val,
                                    m_type,
                                    lambda2=lambda_2,
                                    solver=solver,
                                    known=train,
                                )

                            x_vals = self.scaled_data[label + "_" + m_type]
                            composed_sig = x_vals["composed_signal"].values

                            mse = (
                                1
                                / len(test_data)
                                * sum((composed_sig[test] - data.values[test]) ** 2)
                            )

                            row = [label, r, m_type, lambda_2, l4_val, l5_val, mse]
                            df_error.loc[counter] = row
                            counter += 1

                            t_progress = time()
                            msg = "{:.2f} minutes".format((t_progress - ti) / 60)
                            progress(counter, num_rows, status=msg)

        self.df_error = df_error

        # averaging over runs for each unique set of inputs
        grouped = self.df_error.groupby(
            [
                "system_parameter",
                "degradation_model",
                "lambda_2_val",
                "lambda_4_val",
                "lambda_5_val",
            ]
        )
        df_error_avg = grouped.mean().reset_index().drop(["run_number"], axis=1)

        self.df_error_avg = df_error_avg
