
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
    class Fargate(Clients):
        def __init__(self):
            pass
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
        def init_client(self, 
                       image: str = "", 
                       tags: dict = {}, # optional
                       vpc: str = "",
                       region_name: str = "",
                       environment: dict = {}, 
                       n_workers: int = 10, 
                       threads_per_worker: int = 2
                    ) -> Client:

            print("[i] Initilializing Fargate Cluster ...")

            cluster = FargateCluster(
                tags = tags,
                image = image,
                vpc = vpc,
                region_name = region_name,
                environment = {},
                n_workers = n_workers,
                worker_nthreads = threads_per_worker
            )

            print("[i] Initilializing Dask Client ...")

            client = Client(cluster)

            print(f"[>] Dask Dashboard: {client.dashboard_link}")

            return client
            
