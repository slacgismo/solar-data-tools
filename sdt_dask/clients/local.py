"""
Local Client for Dask Distributed Computing
============================================

This module provides a class for initializing a Dask client optimized for local execution.
It retrieves system information and configures the client based on resource availability.

Classes:
--------
Local
    Manages the creation of a Dask client with local configuration.

Functions:
----------
None
"""
try:
    import os, platform, psutil
    from dask.distributed import Client
    from sdt_dask.clients.clients import Clients

except ModuleNotFoundError as error:
    # Could work on installing modules from the code
    # Or just raise an error like so
    packages = "\tos\n\tplatfrom\n\tpsutil\n\tdask.distributed"
    raise ModuleNotFoundError(f"{error}\n[!] Check or reinstall the following packages\n{packages}")

finally:
    """
    Initializes a Dask client for local execution with resource-aware configuration.
    """       
    class Local(Clients):
        """
        Initializes class attributes.
        """
        def __init__(self):
            pass
        
        """
        Retrieves system information for client configuration.

        Attributes:
        -----------
        self.system: str
            The operating system name (e.g., "windows", "linux").
        self.cpu_count: int
            The number of CPU cores available on the system.
        self.memory: int
            The total system memory in GB.
        """
        def _get_variables(self):
            self.system = platform.system().lower()
            self.cpu_count = os.cpu_count()
            self.memory = int((psutil.virtual_memory().total / (1024.**3)))

        """
        Checks if the specified worker configuration is compatible with system resources.

        Raises:
        -------
        Exception:
            If the configuration exceeds available resources.
        """
        def _check(self):
            if self.workers * self.threads_per_worker > self.cpu_count:
                raise Exception(f"workers and threads exceed local resources, {self.cpu_count} cores present")
            elif self.memory_per_worker < 5:
                raise Exception(f"memory per worker too small, minimum memory size per worker 5 GB")
            
        """
        Initializes a Dask client with local configuration.

        Args:
        -----
        n_workers: int, optional
            The number of Dask workers to create (default: 2).
        threads_per_worker: int, optional
            The number of threads to use per worker (default: 2).
        memory_per_worker: int, optional
            The memory limit for each worker in GB (default: 5).
        verbose: bool, optional
            If True, prints system and client configuration information.

        Returns:
        --------
        Client:
            The initialized Dask client object.
        """
        def init_client(self, n_workers: int = 2, threads_per_worker: int = 2, memory_per_worker: int = 5, verbose: bool = False) -> Client:
            self._get_variables()


            self.workers = n_workers
            self.threads_per_worker = threads_per_worker
            self.memory_per_worker = memory_per_worker
            memory_spill_fraction = False

            self._check()

            if self.workers * self.memory_per_worker > self.memory:
                print(f"[!] memory per worker exceeds system memory ({self.memory} GB), activating memory spill fraction\n")
                memory_spill_fraction = 0.8
            
            if self.system == "windows":
                self.client = Client(processes=False, 
                    memory_spill_fraction=memory_spill_fraction, 
                    memory_pause_fraction=False, 
                    memory_target_fraction=0.8,  # 0.8
                    n_workers=self.workers, 
                    threads_per_worker=self.threads_per_worker, 
                    memory_limit=f"{self.memory_per_worker:.2f}GiB"
                )
            else:
                self.client = Client(memory_limit=f"{self.memory}GB")
            
            if verbose:
                print(f"[i] System: {self.system}")
                print(f"[i] CPU Count: {self.cpu_count}")
                print(f"[i] Memory: {self.memory}")
                print(f"[i] Workers: {self.workers}")
                print(f"[i] Threads per Worker: {self.threads_per_worker}")
                print(f"[i] Memory per Worker: {self.memory_per_worker}")

            print(f"\n[>] Dask Dashboard: {self.client.dashboard_link}\n")
            
            return self.client
