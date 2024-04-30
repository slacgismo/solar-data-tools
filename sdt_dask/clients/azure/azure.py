try:
    import os
    from dask.distributed import Client
    from dask_cloudprovider.azure import AzureVMCluster
    from sdt_dask.clients.clients import Clients

except ModuleNotFoundError as error:
    packages = "\tos\n\tdask.distributed\n\tdask_cloudprovider.azure"
    msg = f"{error}\n[!] Check or reinstall the following packages\n{packages}"
    raise ModuleNotFoundError(msg)

finally:
    class Azure(Clients):
        def __init__(self,
                    resource_group: str = "",
                    vnet: str = "",
                    security_group: str = "",
                    region_name: str = "westus2",
                    vm_size: str = "Standard_D4s_v3",
                    public_ingress: bool = True,
                    disk_size: int = 100,
                    n_workers: int = 5,
                    worker_options: dict = {
                        "nthreads": 2,
                        "memory_limit": "15.63GiB"
                    },
                    docker_image: str = ""
                    ):
                self.resource_group = resource_group
                self.vnet = vnet
                self.security_group = security_group
                self.region_name = region_name
                self.vm_size = vm_size
                self.public_ingress = public_ingress
                self.disk_size = disk_size
                self.n_workers = n_workers
                self.worker_options = worker_options
                self.docker_image = docker_image

        def _check_versions(self):
            data = self.client.get_versions(check=True)
            scheduler_pkgs = data['scheduler']['packages']
            client_pkgs = data['client']['packages']

            for (c_pkg, c_ver), (s_pkg, s_ver) in zip(scheduler_pkgs.items(), client_pkgs.items()):
                if c_ver != s_ver:
                    msg = 'Please update the client version to match the scheduler version'
                    raise EnvironmentError(f"{c_pkg} version mismatch:\n\tScheduler: {s_ver} vs Client: {c_ver}\n{msg}")

        def init_client(self) -> Client:
            try:
                print("[i] Initializing Azure VM Cluster ...")

                #output the worker options
                print(f"[i] Worker Options: {self.worker_options}")

                cluster = AzureVMCluster(
                    resource_group=self.resource_group,
                    vnet=self.vnet,
                    security_group=self.security_group,
                    location=self.region_name,
                    vm_size=self.vm_size,
                    public_ingress=self.public_ingress,
                    disk_size=self.disk_size,
                    n_workers=self.n_workers,
                    worker_options=self.worker_options,

                    docker_image=self.docker_image,
                    env_vars={
                    "AWS_ACCESS_KEY_ID": os.getenv('AWS_ACCESS_KEY_ID'),
                    "AWS_SECRET_ACCESS_KEY": os.getenv('AWS_SECRET_ACCESS_KEY'),
                    "AWS_REGION": os.getenv('AWS_DEFAULT_REGION')
                    }
                )

                print("[i] Initialized Azure VM Cluster")
                print("[i] Initializing Dask Client ...")

                self.client = Client(cluster)

                # self._check_versions()

                print(f"[>] Dask Dashboard: {self.client.dashboard_link}")

                return self.client
            except Exception as e:
                raise Exception(e)

