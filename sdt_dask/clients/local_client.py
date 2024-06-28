"""
Class for the local client plug to be used with the SDT Dask Tool (Runner)
"""
import os, platform, psutil, dask.config
from dask.distributed import Client, LocalCluster
from sdt_dask.clients.clientplug import ClientPlug


class LocalClient(ClientPlug):
    """Local Client Class for configuring dask client on local machine.
    The Class takes in parameters to set up the LocalCluster and the Dask
    Client is initialized using the LocalCluster

    Used in:
        sdt_dask/examples/rev_loc_base_dask.py

    :param workers: The number of workers to initialize the LocalCluster,
        defaults to 2
    :type workers: int
    :param threads: The number of threads used by each worker, defaults to 2
    :type threads: int
    :param memory: The amount of memory to be used by each worker, the
        largest Dataframe size observed is 5.66 GiB, defaults to 6.0
    :type memory: float
    :param kwargs: Keyword arguments for the Dask LocalCluster, for more
        information on the LocalCluster arguments see
        https://distributed.dask.org/en/stable/api.html#distributed.LocalCluster
    :type kwargs: dict
    """
    def __init__(self, workers: int = 2, threads: int = 2, memory: float = 6.0, **kwargs):
        self.workers = workers
        self.threads = threads
        self.memory = memory
        self.kwargs = kwargs
        # Gets the dask configurations to view and make changes to
        # type: dict
        self.dask_config = dask.config
        self.client = None
        self.cluster = None

    def _get_sys_var(self):
        """Obtains the system variables to check available resources on the
        local system with the LocalCluster configurations.

        Gets the number of cores and available memory on the local machine
        """
        # type: int
        self.cpu_count = os.cpu_count()
        # memory is stored in GB
        # type: float
        self.sys_memory = (psutil.virtual_memory().total / (1024.**3))

    def _config_init(self):
        """Checks for dask settings saved in dask's config YAML file, if there
        is no YAML file in the directory it sets the memory management variables
        for the dask worker which include memory spill, target and worker pause,
        terminate variables
        """
        tmp_dir = dask.config.get('temporary_directory')
        if not tmp_dir:
            self.dask_config.set({'distributed.worker.memory.spill': True})
            self.dask_config.set({'distributed.worker.memory.pause': True})
            self.dask_config.set({'distributed.worker.memory.target': 0.95})
            self.dask_config.set({'distributed.worker.memory.terminate': False})


    def _check(self):
        """Checks if the workers, threads and memory are within the resource
        limits of the local machine. If the total worker memory exceeds the
        resource limits and activates the memory spill in the dask worker.
        """
        self._get_sys_var()
        if self.workers * self.threads > self.cpu_count:
            raise Exception(f"workers and threads exceed local resources, {self.cpu_count} cores present")
        if self.workers * self.memory > self.sys_memory:
            self.dask_config.set({'distributed.worker.memory.spill': True})
            print(f"Memory per worker exceeds system memory ({self.memory} GB), activating memory spill\n")

    def init_client(self) -> Client:
        """Initializes the Dask Client and the LocalCluster with the defined
        configuration settings.

        :return: Returns an initialized dask client with the user designed
            configuration
        :rtype: `dask.distributed.Client` object
        """
        self._config_init()
        self._check()

        self.cluster = LocalCluster(n_workers=self.workers,
                                    threads_per_worker=self.threads,
                                    memory_limit=f"{self.memory:.2f}GiB",
                                    **self.kwargs)
        self.client = Client(self.cluster)

        print(f"\nDask Dashboard Link: {self.client.dashboard_link}\n")

        return self.client
