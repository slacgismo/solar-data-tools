"""Class for dataplugs to be used with the SDT Dask tool."""
import pandas as pd
class DataPlug:
    """
    Dataplug class for retrieving data from some source. It's recommended
    that user-created dataplug inherit from this class to ensure compatibility.

    The initialization argument for each class will be different depending on
    the source. The main requirement is to keep the ``DataPlug.get_data`` method,
    and make sure the args and returns as defined here.
    """
    def __init__(self):
        pass

    def get_data(self, keys: tuple) -> pd.DataFrame:
        """
        This is the main function that the Dask tool will interact with.
        Users should keep the args and returns as defined here when writing
        their custom dataplugs.

        :param keys: Tuple containing the required inputs for each plug, w
            which will at least include a unique set of historical power
         generation measurements
        :return: Returns a pandas DataFrame with a timestamp column and
            a power column
        """
        pass
