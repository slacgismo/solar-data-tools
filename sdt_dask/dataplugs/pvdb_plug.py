import pandas as pd

from solardatatools.dataio import load_redshift_data
from sdt_dask.dataplugs.dataplug import DataPlug

class PVDBPlug(DataPlug):
    """Dataplug class for retrieving data from the PVDB (Redshift) database.
    """
    def __init__(self, api_key="DEMO_KEY", power_col="ac_power"):
        self.api_key = api_key
        self.power_col = power_col

    def _pull_data(self, key):
        
        self.df = load_redshift_data(siteid=key, api_key=self.api_key, sensor=0)

    def _clean_data(self):
        """Clean the retrieved data.
        """
        self.df = self.df[[self.power_col]]

    def get_data(self, keys: tuple[int]) -> pd.DataFrame:
        """This is the main function that the Dask tool will interact with.
        Users should keep the args and returns as defined here when writing
        their custom dataplugs.

        :param keys: Tuple containing the required inputs: a unique set of
            historical power generation measurements
        :return: Returns a pandas DataFrame with a timestamp column and
            a power column
        """
        self._pull_data(*keys)
        self._clean_data()

        return self.df
    

    
