
from solardatatools import DataHandler
from solardatatools.dataio import load_redshift_data
from solardatatools.algorithms import Dilation
from solardatatools.algorithms.dilation import dilate_signal
from anomalydetector.utils import divide_df
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

        :param datetime_col: The name of the column containing datetime information,
                             used to set the DataFrame's index to a DatetimeIndex.
                             Required if the DataFrame index is not already a DatetimeIndex.
        
        """

        if data_frames is not None :
            if isinstance(data_frames,list) :
                dict_dataframes = {}
                for data_frame in data_frames :
                    dict_dataframes.update(divide_df(data_frame,datetime_col))
            else :
                dict_dataframes = divide_df(data_frames, datetime_col)
        self.raw_df = dict_dataframes

        dict_handler = {key : DataHandler(dict_dataframes[key]) for key in dict_dataframes}
        for key in dict_handler :
            dict_handler[key].run_pipeline(verbose=False)
        self.raw_handler = dict_handler


    def aligned_data(self):









