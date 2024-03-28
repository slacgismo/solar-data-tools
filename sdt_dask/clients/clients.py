"""
Class for clients to be used with SDT Dask Tool
"""
from dask.distributed import Client
class Clients:
    """
    Clients class for configuring dask client on local and cloud services.
    It's recommended that the user-created client inherit from this class to 
    ensure compatibility.

    The initialization argument for each class will be different depending on
    the client service. The main requiremnet is to keep the 
    ``Clients.init_client`` method, and make sure the args and returns are as
    defined here.
    """
    def __init__(self):
        pass


    def init_client(self, **kwargs) -> Client:
        """
        This is the main function that the Dask Tool will create the clusters
        and the clients. Users should keep the args and returns as defined here
        when writing their custom clients.

        :return: Returns a initialized dask client with the user designed 
            configuration
        """
        pass
