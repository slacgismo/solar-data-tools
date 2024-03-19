import pandas as pd
from solardatatools.dataio import load_redshift_data
from sdt_dask.dataplugs.dataplug import DataPlug

class PVDBPlug(DataPlug):
    """Dataplug class for retrieving data from the PVDB (Redshift) database.
    """
    def __init__(self, power_col="meas_val_f"):
        self.api_key = os.environ.get('REDSHIFT_API_KEY')
        self.power_col = power_col

    def _pull_data(self, siteid, sensor):
        """Pull data from the PVDB database.
        """
        query = {
            'siteid': siteid,
            'api_key': self.api_key,
            'sensor': sensor
        }
        
        self.df = load_redshift_data(**query)

    def _clean_data(self):
        """Clean the retrieved data and set 'ts' column as the index."""
        self.df['ts'] = pd.to_datetime(self.df['ts'])
        self.df.set_index('ts', inplace=True)
        self.df = self.df[[self.power_col]]

    def get_data(self, keys: tuple[str, int]) -> pd.DataFrame:
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
    

    
