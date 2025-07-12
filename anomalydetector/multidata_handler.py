
from solardatatools import DataHandler
from solardatatools.dataio import load_redshift_data
from solardatatools.algorithms import Dilation
from solardatatools.algorithms.dilation import dilate_signal
from anomalydetector.utils import divide_df,common_days
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
        self.filled_mat = None
        self.good_days = None
        


    def aligned(self):
        """
        This method aims to align the different datasets to produce a final set of common 
        good days across all dataframes. We first run the pipeline for all datahandlers 
        and use these results to align the data.

        :return: None
        :rtype: NoneType
        """
        if self.good_days is None :
            for key in self.raw_handler :
                self.raw_handler[key].run_pipeline(verbose=False)
            self.good_days = common_days(self.raw_handler)  
        if self.filled_mat is None :   
            dict_mat = {}
            for key in self.raw_handler:
                dict_mat[key] = self.raw_handler[key].filled_data_matrix[:,self.good_days[key]]
            self.filled_mat = dict_mat
        





        
        









