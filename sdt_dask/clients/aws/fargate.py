import os
from dask.distributed import Client
from dask_cloudprovider.aws import FargateCluster
r'''
Optional:
    tags:  only use this if your organization enforces tag policies
Required:
    image:   should be a dockerhub public image. Please customize your image if needed
'''

TAGS = {
    "project-pa-number": "21691-H2001",
    "project": "pvinsight"
}

IMAGE = "smiskov/dask-sdt-sm:latest"

# AWS_KEY = os.environ['AWS_ACCESS_KEY']
# AWS_SECRET = os.environ['AWS_SECRET_ACCESS_KEY']



def _init_fargate_cluster() -> FargateCluster:
    cluster = FargateCluster(
        tags=TAGS,
        image=IMAGE,
        region_name="us-west-1",
        n_workers=10,
        worker_nthreads=1,
        environment={
            'AWS_ACCESS_KEY_ID': AWS_KEY,
            'AWS_SECRET_ACCESS_KEY': AWS_SECRET
        }
    )
    cluster.scale(12) # TODO: this probably should be accessible somewhere else
    return cluster


def _init_dask_client(cluster: FargateCluster) -> Client:
    client = Client(cluster)
    return client


def get_fargate_cluster(tags=TAGS, image=IMAGE, scale_num=12) -> Client:
    cluster = _init_fargate_cluster()

    return _init_dask_client(cluster)
