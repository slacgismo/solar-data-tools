"""
Class for the Azure client plug to be used with the SDT Dask Tool (Runner)
"""
import os
from dask.distributed import Client
from dask_cloudprovider.azure import AzureVMCluster
from sdt_dask.clients.clientplug import ClientPlug

class AzureClient(ClientPlug):
    """Azure Client Class for configuring dask client on Azure VM Cluster.
    The Class takes in parameters to set up the AzureVMCluster and the
    Dask Client is initialized using the AzureVMCluster

    Used in:
        sdt_dask/examples/rev_far_base_dask.py
        sdt_dask/examples/rev_far_pvdb_dask.py

    :param workers: The number of workers to initialize the AzureVMCluster,
        defaults to 5
    :type workers: int
    :param threads: The number of threads used by each worker, defaults to 2
    :type threads: int
    :param memory: The amount of memory to be used by each worker, the largest
        Dataframe size observed is 5.66 GiB, defaults to 15.36, for more
        information on Azure CPU and memory ranges visit
        https://azure.microsoft.com/en-us/pricing/details/virtual-machines/series/
    :type memory: float
    :param kwargs: Keyword arguments for the Dask AzureVMCluster, for more
        information on the LocalCluster arguments see
        https://cloudprovider.dask.org/en/latest/azure.html
    :type kwargs: dict
    """
    def __init__(self, workers: int = 5, threads: int = 2, memory: float = 15.63, **kwargs):
            self.workers = workers
            self.threads = threads
            self.memory = memory
            self.kwargs = kwargs
            self.client = None
            self.cluster = None

    def init_client(self) -> Client:
        """Initializes the Dask Client and the AzureCluster with the defined
        configuration settings.

        :return: Returns an initialized dask client with the user designed
            configuration
        :rtype: `dask.distributed.Client` object
        """
        print(f"Initializing Azure Cluster with {self.workers} workers, "
              f"{self.threads} threads and {self.memory}MiB per worker...")

        self.cluster = AzureVMCluster(n_workers=self.workers,
                                      worker_options={
                                          "nthreads": self.threads,
                                          "memory_limit": f"{self.memory:.2f}GiB"
                                      }, **self.kwargs)

        print("Initialized Azure VM Cluster")
        print("Initializing Dask Client ...")

        self.client = Client(self.cluster)

        print(f"Dask Dashboard: {self.client.dashboard_link}")

        return self.client