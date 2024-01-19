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
def _init_fargate_cluster(tags: dict, image: str, scale_num: int) -> FargateCluster:
    cluster = FargateCluster(tags=tags, image=image)
    cluster.scale(scale_num) # TODO: this probably should be accessible somewhere else
    return cluster

def _init_dask_client(cluster: FargateCluster) -> Client:
    client = Client(cluster)
    return client

def get_fargate_cluster(tags=TAGS, image=IMAGE, scale_num=12) -> Client:
    cluster = _init_fargate_cluster(tags, image, scale_num)
    return _init_dask_client(cluster)