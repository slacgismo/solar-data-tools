"""Note: this module is not yet implemented.
 This can be written as an example of how to create a Dask client (given some resources)
 that will integrate seamlessly with our SDT Dask tool.
 We should determine if this module/class would be helpful/is needed.
 """

from dask.distributed import Client
from dask_cloudprovider.aws import FargateCluster
"""
Optional:
    tags:  only use this if your organization enforces tag policies
Required:
    image:   should be a dockerhub public image. Please customize your image if needed
"""


def _init_fargate_cluster(**kwargs) -> FargateCluster:
    cluster = FargateCluster(kwargs)
    return cluster


def _init_dask_client(cluster: FargateCluster) -> Client:
    client = Client(cluster)
    return client


def get_fargate_cluster(
        tags={},
        image="",
        vpc="",
        region_name="",
        environment={},
        n_workers=10
) -> Client:

    cluster = _init_fargate_cluster(
        tags={},
        image="",
        vpc="",
        region_name="",
        environment={},
        n_workers=10
    )

    return _init_dask_client(cluster)
