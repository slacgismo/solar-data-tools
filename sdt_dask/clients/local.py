"""
TODO: Change documentation to sphinx format
"""
try:
    import os, platform, psutil, dask.config
    from dask.distributed import Client
    from sdt_dask.clients.clients import Clients

except ModuleNotFoundError as error:
    # Could work on installing modules from the code
    # Or just raise an error like so
    packages = "\tos\n\tplatfrom\n\tpsutil\n\tdask.distributed"
    raise ModuleNotFoundError(f"{error}\n[!] Check or reinstall the following packages\n{packages}")

finally:

    class Local(Clients):
        def __init__(self, n_workers: int = 2, threads_per_worker: int = 2, memory_per_worker: int = 5, verbose: bool = False):
            self.verbose = verbose
            self.n_workers = n_workers
            self.threads_per_worker = threads_per_worker
            self.memory_per_worker = memory_per_worker
            self.dask_config = dask.config

        def _get_sys_var(self):
            self.system = platform.system().lower()
            self.cpu_count = os.cpu_count()
            self.memory = int((psutil.virtual_memory().total / (1024.**3)))

        def _config_init(self):
            tmp_dir = dask.config.get('temporary_directory')
            if not tmp_dir:
                self.dask_config.set({'distributed.worker.memory.spill': False})
                self.dask_config.set({'distributed.worker.memory.pause': False})
                self.dask_config.set({'distributed.worker.memory.target': 0.8})

        def _check(self):
            self._get_sys_var()
            # workers and threads need to be less than cpu core count
            # memory per worker >= 5 GB but total memory use should be less than the system memory available
            if self.n_workers * self.threads_per_worker > self.cpu_count:
                raise Exception(f"workers and threads exceed local resources, {self.cpu_count} cores present")
            if self.n_workers * self.memory_per_worker > self.memory:
                self.dask_config.set({'distributed.worker.memory.spill': True})
                print(f"[!] memory per worker exceeds system memory ({self.memory} GB), activating memory spill fraction\n")

        def init_client(self) -> Client:
            self._config_init()
            self._check()
            
            if self.system == "windows":
                self.client = Client(processes=False,
                    n_workers=self.n_workers,
                    threads_per_worker=self.threads_per_worker, 
                    memory_limit=f"{self.memory_per_worker:.2f}GiB"
                )
            else:
                self.client = Client(processes=True,
                    n_workers=self.n_workers,
                    threads_per_worker=self.threads_per_worker,
                    memory_limit=f"{self.memory_per_worker:.2f}GiB"
                )
            
            if self.verbose:
                print(f"[i] System: {self.system}")
                print(f"[i] CPU Count: {self.cpu_count}")
                print(f"[i] System Memory: {self.memory}")
                print(f"[i] Workers: {self.n_workers}")
                print(f"[i] Threads per Worker: {self.threads_per_worker}")
                print(f"[i] Memory per Worker: {self.memory_per_worker}")
                print(f"[i] Dask worker config:")
                for key, value in self.dask_config.get('distributed.worker').items():
                    print(f"{key} : {value}")

            print(f"\n[>] Dask Dashboard: {self.client.dashboard_link}\n")
            
            return self.client
