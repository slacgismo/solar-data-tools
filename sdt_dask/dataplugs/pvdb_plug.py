import os
import pandas as pd
from solardatatools.dataio import load_redshift_data
from solardatatools.time_axis_manipulation import make_time_series
from sdt_dask.dataplugs.dataplug import DataPlug

class PVDBPlug(DataPlug):
    """
    Dataplug class for retrieving data from the PVDB (Redshift) database.
    """
    def __init__(self, power_col="meas_val_f"):
        self.api_key = os.environ.get('REDSHIFT_API_KEY')
        self.power_col = power_col

    def _pull_data(self, siteid, sensor):
        """
        Pull data from the PVDB database.

        :param siteid: Site ID for the data to be retrieved
        :param sensor: Sensor Index for the data to be retrieved (staring from 0)
        """
        query = {
            'siteid': siteid,
            'api_key': self.api_key,
            'sensor': sensor
        }
        
        self.df = load_redshift_data(**query)

    def _clean_data(self):
        """
        Clean the data and convert the index to a datetime object by calling
        the make_time_series function from the solardatatools package
        """
        self.df, _ = make_time_series(self.df)

    def get_data(self, keys: tuple[str, int]) -> pd.DataFrame:
        """
        This is the main function that the Dask tool will interact with.
        Users should keep the args and returns as defined here when writing
        their custom dataplugs.

        :param keys: Tuple containing the required inputs: a unique set of
            historical power generation measurements, which should be a 
            siteid and a sensor id 
        :return: Returns a pandas DataFrame with a timestamp column and
            a power column
        """
        self._pull_data(*keys)
        self._clean_data()

        return self.df
    

    
