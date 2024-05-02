"""
Class for the fargate client plug to be used with the SDT Dask Tool (Runner)
"""
import dask.config
from sdt_dask.clients.clientplug import ClientPlug
from dask_cloudprovider.aws import FargateCluster
from dask.distributed import Client

class FargateClient(ClientPlug):
    """Fargate Client Class for configuring dask client on ECS Cluster using
    Fargate. The Class takes in parameters to set up the FargateCluster and the
    Dask Client is initialized using the FargateCluster

    Used in:
        sdt_dask/examples/rev_far_base_dask.py
        sdt_dask/examples/rev_far_pvdb_dask.py

    :param workers: The number of workers to initialize the FargateCluster,
        defaults to 2
    :type workers: int
    :param threads: The number of threads used by each worker, defaults to 2
    :type threads: int
    :param memory: The amount of memory to be used by each worker, default CPU
        is 4vCPU and the memory can be specified between 8 and 30, the largest
        Dataframe size observed is 5.66 GiB, defaults to 16 GB, for more
        information on ECS CPU and memory ranges visit
        https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task-cpu-memory-error.html
    :type memory: int
    :param kwargs: Keyword arguments for the Dask FargateCluster, for more
        information on the FargateCluster arguments see
        https://cloudprovider.dask.org/en/latest/aws.html
    :type kwargs: dict
    """
    def __init__(self, workers: int = 2, threads: int = 2, memory: int = 16, **kwargs):
        self.workers = workers
        self.threads = threads
        self.memory = memory * 1024
        self.kwargs = kwargs
        self.dask_config = dask.config
        self.client = None
        self.cluster = None

    def init_client(self) -> Client:
        """Initializes the Dask Client and the FargateCluster with the defined
        configuration settings.

        :return: Returns an initialized dask client with the user designed
            configuration
        :rtype: `dask.distributed.Client` object
        """
        print(f"Initializing Fargate Cluster with {self.workers} workers, "
              f"{self.threads} threads and {self.memory}MiB per worker...")

        self.cluster = FargateCluster(n_workers=self.workers,
                                      worker_nthreads=self.threads,
                                      worker_mem=self.memory,
                                      **self.kwargs)

        print("Initialized Fargate Cluster")
        print("Initializing Dask Client ...")

        self.client = Client(self.cluster)

        self.client.get_versions(check=False)

        print(f"Dask Dashboard Link: {self.client.dashboard_link}")

        return self.client
