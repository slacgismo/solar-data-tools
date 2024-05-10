"""Class for dataplugs to be used with the SDT Dask tool."""
import pandas as pd
from solardatatools.dataio import get_pvdaq_data
from sdt_dask.dataplugs.dataplug import DataPlug


class PVDAQPlug(DataPlug):
    """
    Dataplug class for retrieving data from the PVDAQ DB.
    Note that the DEMO_KEY has a rate limit of 30/h, 50/d per IP address.
    """
    def __init__(self, api_key="DEMO_KEY", power_col="ac_power"):
        self.api_key = api_key
        self.power_col = power_col

    def _pull_data(self, key, year):
        """
        Pull the data from the PVDAQ database using the get_pvdaq_data function
        from the solardatatools package.
        """
        self.df = get_pvdaq_data(sysid=key, year=year, api_key=self.api_key)

    def _clean_data(self):
        # pick out one power col
        self.df = self.df[['ac_power']]

    def get_data(self, keys: tuple[int, int]) -> pd.DataFrame:
        """This is the main function that the Dask tool will interact with.
        Users should keep the args and returns as defined here when writing
        their custom dataplugs.

        :param keys: Tuple containing the required inputs: a unique set of
            historical power generation measurements, and the year to query
        :return: Returns a pandas DataFrame with a timestamp column and
            a power column
        """
        # In this case the process to get the data is simple since it's all
        # done in the get_pvdaq_data function, but in some cases it could be
        # more complex
        self._pull_data(*keys)
        self._clean_data()

        return self.df


