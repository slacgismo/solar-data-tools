
from solardatatools import DataHandler
from solardatatools.algorithms import Dilation

from anomalydetector.utils import divide_df,common_days

from scipy.stats import uniform
from tqdm import tqdm
import numpy as np
from datetime import datetime
import bisect
import matplotlib.pyplot as plt
import warnings



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
        :type data_frame: pandas.DataFrame/list, optional 
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
        #Keep track of the clean data
        self.filled_mat = None
        self.good_days = None
        self.common_days = None
        #Keep track of the dilated data 
        self.dil_mat = None
        #Keep track of the failure scenario data
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
            warnings.warn(
                "No failure scenario detected — calling multidata.generate_failure automatically with default parameters.",
                category=UserWarning
            )

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

    def ndil(self):
        if self.dil_mat is None :
            return None
        else :
            key = list(self.dil_mat.keys())[0]
            return self.dil_mat[key].shape[0]
        
    def display(self, idx, site=None, ax=None):
        created_figure = False
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
            created_figure = True

        if isinstance(idx, datetime):
            idx = bisect.bisect_left(self.common_days, idx)

        if site is None or site == self.target:
            ax2 = ax.twinx()
            x = np.arange(self.ndil())

            ax.plot(x, self.failure_mat[:, idx], label="Power with outage", color="black")
            ax.plot(x, self.dil_mat[self.target][:, idx], label="Power no outage",
                    color="black", linestyle="dotted", linewidth=1)
            ax2.plot(x, self.random_mat[:, idx], label="Loss fraction",
                    color="red", linestyle="dotted")

            ax.set_ylabel("Power")
            ax2.set_ylabel("Loss")
            ax2.set_ylim(0, 1)

            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
            ax.grid(True)
        
        else :
            x = np.arange(self.ndil())
            ax.plot(x, self.dil_mat[site][:, idx], label="Power no outage",
                    color="black", linestyle="dotted", linewidth=1)
            ax.set_ylabel("Power")
            lines1, labels1 = ax.get_legend_handles_labels()
            ax.legend(lines1, labels1, loc="upper right")
            ax.grid(True)

        ax.set_xlabel("Time step")
        ax.set_title(f"Display for day {self.common_days[idx].date()} ({site if site else self.target})")

        if created_figure:
            plt.tight_layout()
            plt.show()

def train_test_split(multidata : MultiDataHandler,test_size = None, train_size = None, shuffle = True):

    if test_size is None and train_size is None:
        raise ValueError("Missing one argument of size")
    elif test_size is None :
        test_size = 1-train_size
    elif train_size is None :
        train_size = 1-test_size
    elif train_size+test_size != 1 :
        raise ValueError("Incompatible arguments given")
    
    X1 = MultiDataHandler()
    X2 = MultiDataHandler()
    X1.raw_handler = multidata.raw_handler
    X2.raw_handler = multidata.raw_handler
    X1.keys = multidata.keys
    X2.keys = multidata.keys
    if multidata.common_days is None or multidata.filled_mat is None:
        multidata.align()
    n = len(multidata.common_days)
    split = int(train_size * n)
    if shuffle :
        indices = np.random.permutation(n)
        train_indices = indices[:split]
        test_indices = indices[split:]
    else :
        train_indices = list(range(split))       
        test_indices = list(range(split, n)) 
    X1.common_days = multidata.common_days[train_indices]
    X2.common_days = multidata.common_days[test_indices]
    X1.filled_mat = {}
    X2.filled_mat = {}
    X1.good_days = {}
    X2.good_days = {}
    for key in multidata.keys :
        X1.good_days[key] = multidata.good_days[key][train_indices]
        X2.good_days[key] = multidata.good_days[key][test_indices]
        X1.filled_mat[key] = multidata.filled_mat[key][:,train_indices]
        X2.filled_mat[key] = multidata.filled_mat[key][:,test_indices]
    if multidata.dil_mat is not None :
        X1.dil_mat = {}
        X2.dil_mat = {}
        for key in multidata.keys :
            X1.dil_mat[key] = multidata.dil_mat[key][:,train_indices]
            X2.dil_mat[key] = multidata.dil_mat[key][:,test_indices]
    if multidata.random_mat is not None :
        X1.random_mat = multidata.random_mat[:,train_indices]
        X2.random_mat = multidata.random_mat[:,test_indices]
        X1.failure_mat = multidata.failure_mat[:,train_indices]
        X2.failure_mat = multidata.failure_mat[:,test_indices]
        X1.target = multidata.target
        X2.target = multidata.target
    return X1,X2





        
        



        
        





        
        









