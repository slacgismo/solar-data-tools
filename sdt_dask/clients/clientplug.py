"""
Class for clients to be used with SDT Dask Tool (Runner)
"""
from dask.distributed import Client
class ClientPlug:
    """Clients class for configuring dask client on local and cloud services.
    It's recommended that the user-created client inherit from this class to 
    ensure compatibility.

    The initialization argument for each class will be different depending on
    the client service. The main requirement is to keep the
    ``Clients.init_client`` method, and make sure the args and returns are as
    defined here.

    Used in:
        std_dask/clients/local_client.py
        std_dask/clients/aws/fargate_client.py
        std_dask/clients/azure/azure_client.py
    """
    def __init__(self, **kwargs):
        """
        This is the client plug class initialization method where the arguments
        for the cluster configurations are provided as keyword arguments. This
        is then passed to the ``ClientPlug.init_client`` method to initialize
        the Cluster and Client.
        """
        pass


    def init_client(self) -> Client:
        """
        This is the main function that the Dask Tool will create the clusters
        and the clients. Users should keep the args and returns as defined here
        when writing their custom clients.

        :return: Returns a initialized dask client with the user designed 
            configuration
        :rtype: `dask.distributed.Client` object
        """
        pass
