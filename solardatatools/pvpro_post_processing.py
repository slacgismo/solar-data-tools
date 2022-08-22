""" PVPRO Post Processing Module

This module contains a class that takes in the output dataframe of PVPRO and contains
methods to process the dataset, perform signal decompositions to model degradation trends,
analyze how well the models fit, and visualize the trends. 

"""

import numpy as np

import pandas as pd
import cvxpy as cp

import matplotlib.pyplot as plt

from collections import defaultdict
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from time import time
from solardatatools.utilities import progress


class PVPROPostProcessor():
    
    def __init__(self, file_name, period, index_col=0, dates=None, verbose=False, bp=True, 
                 bounds='Default', est=False):
        # imports data and assigns index column and date columns - the optional dates entry
        # should be a list of column indices [i, j, k]
        # the verbose keyword argument indicates whether the % of points on the boundaries 
        # should be printed
        # the bounds keyword arguments allows the user to input custom boundary values; input
        # is a dictionary
        # est should only be True when the input dataframe contains 'ref' and 'ref_est'
        # columns
        
        if dates is None:
            dates = [0]
        else:
            dates = dates
        
        # attributes
        self.df = pd.read_csv(file_name, index_col=index_col, parse_dates=dates)
        self.period = period
        self.df_p = None
        self.df_ds = None
        self.bound_indices = None
        self.df_scaled = None
        self.scaler = None
        self.df_l = None
        self.df_error = None
        self.df_error_avg = None
        
        if bounds is 'Default':
            self.PVPROSystem = {'photocurrent_ref':[0.01, 10, 0.01], 
                                'saturation_current_ref':[0, 1*10**(-6), 5*10**(-12)], 
                                'resistance_series_ref':[0.1, 2, 0.05], 
                                'resistance_shunt_ref':[100, 500, 1]}
        else:
            self.PVPROSystem = bounds
        
        def default_val():
            return 'no_entry'
        
        self.ScaledData = defaultdict(default_val)
        self.DescaledData = defaultdict(default_val)
        
        # combines all of the preprocessing steps
        self.data_setup(est=est)
        if bp==True:
            self.boundary_points(verbose=verbose)
            self.boundary_to_nan()
        else:
            pass
        self.scale_max_1()
        self.ln_df()
        df_p = self.df_l
        
        self.df_p = df_p
    
    ############################################################################################################
    # Processing functions
    ############################################################################################################
    
    def data_setup(self, est=False):
        # adjusts time index so that there are equal intervals
        time_delta, count = mode(np.diff(self.df.index))
        freq = int(time_delta[0] / np.timedelta64(1, 's'))
        new_index = pd.date_range(start=self.df.index[0], end=self.df.index[-1], freq='{}s'.format(freq))
        self.df = self.df.reindex(index=new_index)
        
        # isolates columns of interest
        if est is True:
            cols = [c for c in self.df.columns if 'start' not in c and 'end' not in c and 'mean' not in c and 
                    'alpha' not in c and 'beta' not in c and 'nNsVth' not in c and 'diode' not in c and 'year' 
                    not in c and 'Unnamed' not in c and 'cells' not in c and 'est' in c]
        else:
            cols = [c for c in self.df.columns if 'start' not in c and 'end' not in c and 'mean' not in c and 
                    'alpha' not in c and 'beta' not in c and 'nNsVth' not in c and 'diode' not in c and 'year' 
                    not in c and 'Unnamed' not in c and 'cells' not in c]
        
        df_ds = self.df.loc[:, cols]
        df_ds = df_ds.reindex(index=new_index)
        
        self.df_ds = df_ds
    
    def boundary_points(self, verbose=False):
        # determines indices of points on the boundary to a tolerance (set in init)
        bounds = self.PVPROSystem
        indices = []
        df = self.df
        bounded_params = bounds.keys()
        
        for name in bounded_params:
            bpoints = np.arange(len(df[name]))[np.logical_or(df[name] >= bounds[name][1] - bounds[name][2], 
                                                              df[name] <= bounds[name][0] + bounds[name][2])]
            indices = np.concatenate((indices, bpoints))
        
        indices = np.unique(indices).astype(int)
        self.bound_indices = indices
        
        if verbose is True:
            print('Percent on boundaries:', (100*(len(indices)/len(df.index))), '%')
        else:
            pass
    
    def boundary_to_nan(self):
        # makes all points in the dataframe at boundary point
        # indices be nan
        self.df_ds.iloc[self.bound_indices, :] = np.nan
    
    def scale_max_1(self):
        # scales a dataframe to have max value 1, min value 0
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(self.df_ds.to_numpy())
        df_scaled = pd.DataFrame(df_scaled, columns=self.df_ds.columns, index=self.df_ds.index)
        
        self.df_scaled = df_scaled
        self.scaler = scaler
    
    def ln_df(self):
        # takes the natural log of the scaled dataframe and 
        # makes all inf values nan
        cond = self.df_scaled > 0
        df_l = np.log(self.df_scaled[cond])
        
        self.df_l = df_l
    
    def view_minmax(self, df):
        # prints the minimum and maximum values for each column 
        # in the dataframe
        for label, values in df.items():
            print(label)
            print(min(df[label]))
            print(max(df[label]))
    
    ############################################################################################################
    # Signal decomposition
    ############################################################################################################
    
    def optimize(self, label, lambda4, lambda5, model, lambda2=0.001, 
                 verbose=False, known=None, solver='Default'):    
        # runs an optimization problem to perform a 5-component signal decomposition
        # on one parameter of the PV system
        # lambda4 indicates the strength of smoothing on the periodic component
        # and lambda5 controls the weight of smoothing on the degradation component
        
        data = self.df_p[label]
        y = self.df_p[label].values
        T = len(y)
        p = self.period
        
        if known is None:
            known = ~np.isnan(y)
        else:
            known = np.logical_and(known, ~np.isnan(y))
        
        if 'series' in label:
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
        
        cost = (1/T)*cp.sum_squares(x1) + lambda_2*cp.norm1(x2) + lambda_4*cp.sum_squares(cp.diff(x4, k=2))
        
        constraints = [y[known] == (x1 + x2 + x3 + x4 + x5)[known],
                           cp.diff(x3, k=1) == 0,
                           cp.sum(x4[:p]) == 0,
                           x4[p:] == x4[:-p],
                           x5[0] == 0]
        
        # additional costs and conditions for all the model types
        if model == 'linear':        
            constraints.append(cp.diff(x5, k=2) == 0)
        
        elif model == 'monotonic':        
            if decreasing == True:
                constraints.append(cp.diff(x5, k=1) <= 0)
            else:
                constraints.append(cp.diff(x5, k=1) >= 0)

        elif model == 'smooth_monotonic':
            cost += lambda_5*cp.sum_squares(cp.diff(x5, k=2))

            if decreasing == True:
                constraints.append(cp.diff(x5, k=1) <= 0)
            else:
                constraints.append(cp.diff(x5, k=1) >= 0)

        elif model == 'piecewise_linear':
            cost += lambda_5*cp.norm1(cp.diff(x5, k=2))

            if decreasing == True:
                constraints.append(cp.diff(x5, k=1) <= 0)
            else:
                constraints.append(cp.diff(x5, k=1) >= 0)
        
        else:
            print('No model', model)

        obj = cp.Minimize(cost)
        prob = cp.Problem(obj, constraints)

        if solver is 'Default':
            if verbose is True:
                prob.solve(verbose=True)
            else:
                prob.solve()
        else:
            if verbose is True:
                prob.solve(solver=solver, verbose=True)
            else:
                prob.solve(solver=solver)
        
        df_components = pd.DataFrame(index=self.df_p.index, data={'x1':x1.value, 'x2':x2.value, 'x3':x3.value,
                                                                  'x4':x4.value, 'x5':x5.value, 
                                                                  'composed_signal':(x3.value + x4.value + x5.value)})
        
        def default_val():
            return 'no_entry'
        
        self.ScaledData = defaultdict(default_val)
        self.DescaledData = defaultdict(default_val)
        
        self.ScaledData[label + '_' + model] = df_components
        
        # exponentiating the components and undoing the scaling
        x1 = np.exp(df_components['x1'].values)
        x2 = np.exp(df_components['x2'].values)
        x3 = np.exp(df_components['x3'].values)
        x4 = np.exp(df_components['x4'].values)
        x5 = np.exp(df_components['x5'].values)
        
        ind = self.df_ds.columns.get_loc(label)
        
        max_val = self.scaler.data_max_[ind]
        min_val = self.scaler.data_min_[ind]
        coefficient = (max_val - min_val)
        x3 = x3*coefficient
        
        df_descaled = pd.DataFrame(index=self.df_ds.index, data={'x1':x1, 'x2':x2, 'x3':x3, 'x4':x4, 'x5':x5, 
                                                            'composed_signal':(x3*x4*x5 + min_val)})
        
        self.DescaledData[label + '_' + model] = df_descaled
        
    ############################################################################################################
    # Plotting functions
    ############################################################################################################
    
    def plot_df(self, df):
        # plots columns of up to a 9 column dataframe
        plt.figure(figsize=(7,len(df.columns)*2))
        sp_counter = len(df.columns)*100 + 11
        
        for label, values in df.items():
            plt.subplot(sp_counter)
            plt.scatter(df.index, df[label])
            plt.title(label)
            plt.xticks(rotation=45)
            sp_counter += 1
        
        plt.tight_layout()
        plt.show()
    
    def plot_sd_space(self, label, model, model_title=None):
        # plots the SD of one system parameter jn the scaled log space
        if self.ScaledData[label + '_' + model] is 'no_entry':
            opt = input('No data entry, would you like to run optimization? (y/n)\n')
            
            if opt == 'y':
                l4 = input('Enter your periodic weight value:\n')
                l5 = input('Enter your degradation weight value:\n')
                
                print('Running optimize()')
                self.optimize(label, float(l4), float(l5), model)
                self.plot_sd_space(label, model, model_title=model_title)
                
            if opt == 'n':
                pass
        else:        
            components = self.ScaledData[label + '_' + model]
            if model_title is None:
                model_title = model
            else:
                pass

            titles = ['Residual', 'Laplacian Noise', 'Bias', 'Periodic', model_title, 'Composed Signal']

            plt.figure(figsize=(6,20))
            counter = 0

            for col, values in components.items():
                plt.subplot(611 + counter)
                if titles[counter] == 'Residual' or titles[counter] == 'Laplacian Noise':
                    plt.scatter(self.df_p.index, components[col])
                elif titles[counter] == 'Composed Signal':
                    plt.scatter(self.df_p.index, self.df_p[label], c='orange')
                    plt.plot(self.df_p.index, components[col])
                    plt.xlabel('Time')
                elif titles[counter] == 'Bias':
                    plt.plot(self.df_p.index, components[col])
                    plt.ylim((components[col][0] - 0.1*components[col][0], 
                              components[col][0] + 0.1*components[col][0]))
                else:
                    plt.plot(self.df_p.index, components[col])

                plt.title(titles[counter])
                plt.xticks(rotation=45)

                counter += 1

            plt.tight_layout()
            plt.show()
    
    def plot_original_space(self, label, model, model_title=None):
        if self.DescaledData[label + '_' + model] is 'no_entry':
            opt = input('No data entry, would you like to run optimization? (y/n)\n')
            
            if opt == 'y':
                l4 = input('Enter your periodic weight value:\n')
                l5 = input('Enter your degradation weight value:\n')
                
                print('Running optimize()')
                self.optimize(label, float(l4), float(l5), model)
                self.plot_original_space(label, model, model_title=model_title)
                
            if opt == 'n':
                pass
        else:        
            data = self.df_ds[label]
            components = self.DescaledData[label + '_' + model]
            if model_title is None:
                model_title = model
            else:
                pass

            titles = ['Residual', 'Laplacian Noise', 'Bias', 'Periodic', model_title, 'Composed Signal']

            plt.figure(figsize=(6,20))
            counter = 0

            for col, values in components.items():
                plt.subplot(611 + counter)
                if col == 'x1' or col == 'x2':
                    plt.scatter(self.df_ds.index, components[col])
                elif col == 'composed_signal':
                    plt.scatter(self.df_ds.index, self.df_ds[label], c='orange')
                    plt.plot(self.df_ds.index, components[col])
                    plt.xlabel('Time')
                elif col == 'x3':
                    plt.plot(self.df_ds.index, components[col])
                    plt.ylim((components[col][0] - 0.1*components[col][0], 
                              components[col][0] + 0.1*components[col][0]))
                else:
                    plt.plot(self.df_ds.index, components[col])

                plt.title(titles[counter])
                plt.xticks(rotation=45)

                counter += 1

            plt.tight_layout()
            plt.show()
    
    ############################################################################################################
    # Error calculations 
    ############################################################################################################
    
    def error_analysis(self, lambda4, lambda5, num_runs, lambda2=0.001, solver='Default'):
        # calculates the holdout error, looping over system parameters, models, cost function weights,
        # and number of repetitions
        # lambda4 and lambda5 inputs are arrays
        
        ti = time()
        period = self.period
        lambda_2 = lambda2
        lambda_4_values = lambda4
        lambda_5_values = lambda5
        runs = np.arange(num_runs)
        models = ['linear', 'monotonic', 'smooth_monotonic', 'piecewise_linear']
        cols = ['system_parameter', 'run_number', 'degradation_model', 'lambda_2_val', 
                'lambda_4_val', 'lambda_5_val', 'mean_sq_error']
        num_rows = (len(lambda_5_values) + 1)*len(lambda_4_values)*2*len(runs)*len(self.df_p.columns)
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
                        if m_type == 'smooth_monotonic' or m_type == 'piecewise_linear':
                            l5_iter = lambda_5_values   
                            
                        else:
                            l5_iter = [1]
                            
                        for l5_val in l5_iter:
                            if solver is 'Default':
                                self.optimize(label, l4_val, l5_val, m_type, lambda2=lambda_2, 
                                              solver=solver, known=train)
                            else:
                                self.optimize(label, l4_val, l5_val, m_type, lambda2=lambda_2, 
                                              solver=solver, known=train)
                                
                            x_vals = self.ScaledData[label + '_' + m_type]
                            composed_sig = x_vals['composed_signal'].values
                            
                            mse = 1/len(test_data)*sum((composed_sig[test] - data.values[test])**2)

                            row = [label, r, m_type, lambda_2, l4_val, l5_val, mse]
                            df_error.loc[counter] = row
                            counter += 1

                            t_progress = time()
                            msg = "{:.2f} minutes".format((t_progress-ti)/60)
                            progress(counter, num_rows, status=msg)
                            
        self.df_error = df_error
        
        ti = time()
        cols = ['system_parameter', 'degradation_model', 'lambda_2_val', 'lambda_4_val', 
                'lambda_5_val', 'mean_sq_error']
        num_rows_avg = len(lambda_4_values)*(len(lambda_5_values) + 1)*2*len(self.df_p.columns)
        df_error_avg = pd.DataFrame(columns=cols, index=np.arange(num_rows_avg))
        counter = 0
        progress(counter, num_rows_avg)
        
        for label, values in self.df_p.items():
            for m_type in models:
                for l4_val in lambda_4_values:
                    if m_type == 'smooth_monotonic' or m_type == 'piecewise_linear':
                        l5_iter = lambda_5_values     
                    else:
                        l5_iter = [1]
                        
                    for l5_val in l5_iter:
                        ar1 = df_error[df_error['system_parameter']==label]
                        ar2 = ar1[ar1['degradation_model']==m_type]
                        ar3 = ar2[ar2['lambda_4_val']==l4_val]
                        all_runs = ar3[ar3['lambda_5_val']==l5_val]
                        avg_mse = all_runs['mean_sq_error'].mean()
                        
                        row = [label, m_type, lambda_2, l4_val, l5_val, avg_mse]
                        df_error_avg.loc[counter] = row
                        counter += 1
                        
                        t_progress = time()
                        msg = "{:.2f} minutes".format((t_progress-ti)/60)
                        progress(counter, num_rows_avg, status=msg)
        
        self.df_error_avg = df_error_avg