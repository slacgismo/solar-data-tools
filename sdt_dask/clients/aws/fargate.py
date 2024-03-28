# TODO: Change all documentation to sphinx
try:
    # Import checks
    import os
    from sdt_dask.clients.clients import Clients
    from dask_cloudprovider.aws import FargateCluster
    from dask.distributed import Client

# Raises exception if modules aren't installed in the environment
except ModuleNotFoundError as error:
    packages = "\tos\n\tdask.distributed\n\tdask_cloudprovider"
    msg = f"{error}\n[!] Check or reinstall the following packages\n{packages}"
    raise ModuleNotFoundError(msg)

finally:
    class Fargate(Clients):
        """
        Fargate Class for Dask on AWS Fargate

        This class simplifies the process of setting up a Fargate cluster and
        connecting a Dask client to it, enabling distributed execution
        using AWS Fargate.

        Requires:
        - dask
        - dask_cloudprovider

        **Important:** Ensure you have appropriate IAM permissions to manage
        AWS Fargate resources.
        """
        def __init__(self):
            pass
        def _check_versions(self):
            data = self.client.get_versions(check=True)
            scheduler_pkgs = data['scheduler']['packages']
            client_pkgs = data['client']['packages']

            for (c_pkg, c_ver), (s_pkg, s_ver) in zip(scheduler_pkgs.items(), client_pkgs.items()):
                if c_ver != s_ver:
                    msg = 'Please Update the client version to match the Scheduler version'
                    raise EnvironmentError(f"{c_pkg} version Mismatch:\n\tScheduler: {s_ver} vs Client: {c_ver}\n{msg}")

        def init_client(self, 
                       image: str = "", 
                       tags: dict = {}, # optional
                       vpc: str = "",
                       region_name: str = "",
                       environment: dict = {}, 
                       n_workers: int = 10, 
                       threads_per_worker: int = 2
                    ) -> Client:
            """
            Initializes a Dask Client instance that leverages AWS Fargate for distributed execution.

            Args:
                image (str, required): Docker image to use for the Fargate tasks. Defaults to "".
                tags (dict, optional): Dictionary of tags to associate with the Fargate cluster. Defaults to an empty dictionary.
                vpc (str, required): VPC ID to launch the Fargate cluster in. Defaults to "".
                region_name (str, required): AWS region to launch the Fargate cluster in. Defaults to "".
                environment (dict, required): Environment variables to set for the Fargate tasks. Defaults to an empty dictionary.
                n_workers (int, optional): Number of worker nodes in the Fargate cluster. Defaults to 10.
                threads_per_worker (int, optional): Number of threads per worker in the Fargate cluster. Defaults to 2.

            Returns:
                Client: The initialized Dask client object connected to the Fargate cluster.
            """
            print("[i] Initilializing Fargate Cluster ...")

            cluster = FargateCluster(
                tags = tags,
                image = image,
                vpc = vpc,
                region_name = region_name,
                environment = environment,
                n_workers = n_workers,
                worker_nthreads = threads_per_worker
            )

            print("[i] Initialized Fargate Cluster")
            print("[i] Initilializing Dask Client ...")

            self.client = Client(cluster)

            self._check_versions()

            print(f"[>] Dask Dashboard: {self.client.dashboard_link}")

            return self.client
            