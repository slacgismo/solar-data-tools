
from solardatatools import DataHandler
from solardatatools.dataio import load_redshift_data
from solardatatools.algorithms import Dilation

from anomalydetector.utils import divide_df,common_days

from scipy.stats import uniform
from tqdm import tqdm
import numpy as np
import pandas as pd


class MultiDataHandler:
    def __init__(
            self,
            data_frames = None,
            datetime_col = None,
        ):

        """
        Class to handle a set of different timeseries and preprocess them. 
        This class work with a list of data frames, allowing the pipeline from the main class DataHandler to run 
        for each timeseries, aligned the dataframe between each others and dilate them.

        :param data_frame: The list of Pandas Dataframe or unique Dataframe containing the values for 
                            all the different sites.
        :type data_frame: pandas.DataFrame, optional 
        :param datetime_col: The name of the column containing datetime information,
                             used to set the DataFrame's index to a DatetimeIndex.
                             Required if the DataFrame index is not already a DatetimeIndex.
        :type datetime_col: str, optional
        """

        if data_frames is not None :
            if isinstance(data_frames,list) :
                dict_dataframes = {}
                for data_frame in data_frames :
                    dict_dataframes.update(divide_df(data_frame,datetime_col))
            else :
                dict_dataframes = divide_df(data_frames, datetime_col)
        else :
            dict_dataframes = {}
        self.raw_df = dict_dataframes

        dict_handler = {key : DataHandler(dict_dataframes[key]) for key in dict_dataframes}
        self.raw_handler = dict_handler
        self.keys = list(self.raw_handler.keys())
        self._initialize_attributes()

    def _initialize_attributes(self):
        self.filled_mat = None
        self.good_days = None
        self.common_days = None
        self.dil_mat = None

        self.random_mat = None
        self.failure_mat = None
        self.target = None

        
    def align(self):
        """
        This method aims to align the different datasets to produce a final set of common 
        good days across all dataframes. We first run the pipeline for all datahandlers 
        and use these results to align the data.

        :return: None
        :rtype: NoneType
        """
        if self.good_days is None :
            for key in tqdm(self.keys, desc="Running pipelines"):
                self.raw_handler[key].run_pipeline(verbose=False)
            self.good_days = common_days(self.raw_handler)
            self.common_days = self.raw_handler[self.keys[0]].day_index[self.good_days[self.keys[0]]]
        if self.filled_mat is None :   
            dict_mat = {}
            for key in tqdm(self.keys,desc="Aligning datasets"):
                dict_mat[key] = self.raw_handler[key].filled_data_matrix[:,self.good_days[key]]
            self.filled_mat = dict_mat


    def dilate(self,ndil = None):
        """
        This method dilates the datasets and construct a dictionary of dilated matrices
        :param ndil: number of step by day after the dilution
        :type ndil: int, optional 
        """
        if ndil is None :
            self.dil_mat = self.filled_mat
        else :
            dict_mat_dil = {}
            for key in self.raw_handler :
                dil = Dilation(self.raw_handler[key],**{'nvals_dil': ndil, 'matrix': 'filled'})
                dil_vector = dil.signal_dil[1:]
                dil_matrix = dil_vector.reshape(dil_vector.shape[0]//ndil,ndil).T
                dict_mat_dil[key] = dil_matrix[:,self.good_days[key]]
            self.dil_mat = dict_mat_dil
        if self.failure_mat is not None and self.target is not None and self.random_mat is not None :
            self.generate_failure(self.target)


    def generate_failure(self,
                         target,
                         loss_distribution = None,
                         proportion_totalday = 0,
                         seed = 42
                         ):
        """
        This method generate a dataset where target is affected by a random loss with a 
        determined distribution for the length of the failure and the intensity.

        :param target: The string of the target timeserie we generate a failure scenario for this timeserie.
        :type target: str
        :param loss_distribution: The distribution of the loss amplitude the distribution should be between 0 and 1 
                                but is clip anyway. The default distribution is the uniform distribution.
        :type loss_distribution: scipy.stats distribution, optional
        :param proportion_totalday: The proportion of days with a complete failure, ie during the whole day.
        :type proportion_totalday: int, optional
        """
        np.random.seed(seed)
        if loss_distribution is None :
            loss_distribution = uniform(loc=0, scale=1)
        if self.dil_mat is None :
            self.dilate()
        ndil = self.dil_mat[target].shape[0]
        N = self.dil_mat[target].shape[1]

        duration = np.random.randint(1,ndil,N)
        start = np.random.randint(0,ndil-duration)
        end = start+duration
        losses = np.clip(loss_distribution.rvs(size=N),a_min=0,a_max=1)
        full_day = np.random.choice([False, True], size=N, p=[1 - proportion_totalday, proportion_totalday])

        start[full_day] = 0
        end[full_day] = ndil

        mask = np.zeros(self.dil_mat[target].shape)
        for j in range(mask.shape[1]):
            mask[start[j]:end[j], j] = losses[j]
        self.random_mat = mask
        self.failure_mat = self.dil_mat[target]*(1-mask)
        self.target = target

        
        



        
        





        
        









