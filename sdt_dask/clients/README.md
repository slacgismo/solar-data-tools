This README provides guidance on creating your own Client Plugs or use our existing 
ClientPlugs within the SDT Runner tool. ClientPlugs are classes used by SDT runner 
tool to compute data on various sources. Follow these instructions to 
create your own ClientPlug or use the existing client plug examples. The demo code 
to use each example ClientPlug are provided in `examples/*.ipynb`

## Creating Your Own ClientPlug
To create your own ClientPlug, you must provide two key files: a Python module
(`user_clientplug.py`) containing your ClientPlug class and a `requirements.txt`
file listing all necessary external Python Packages.

### Implement Your ClientPlug
1. **Inherit from the Base ClientPlug Class**: Any ClientPlug class should 
inherit from the ClientPlug base class. This inheritance guarantees that your
ClientPlug aligns with the expected structure and can seamlessly integrate with 
the SDT Runner Tool.
2. **Define Initialization Parameters**: For consistency and ease of use it is 
recommended to have `workers`, `threads` and `memory` arguments in the `__init__()` 
method and passing all other Cluster arguments as `**kwargs`. 
3. **Implement `init_client()` Method**: Core method for ClientPlug, it sets up 
the Cluster and then creates the Dask Client using the Cluster. This method should
return the Dask Client.
4. **Additional Methods**: Beyond `init_client()` you may implement any number of 
private or public methods to aid in managing the clusters, setting up encryption modes,
and getting logs before shutdown.

Example Structure:
```python
from dask.distributed import Client
from sdt_dask.clients.clientplug import ClientPlug

class UserClientPlug(ClientPlug):
    def __init__(self, workers, threads, memory, **kwargs):
        # Initialize variables and format if required
    
    def init_client(self) -> Client:
        # Cluster configurations and Client Initialization
        return self.client
```
## Existing ClientPlug Examples
Below detailed descriptions of the ClientPlugs available for use with the SDT 
Runner Tool. Each clientPlug uses the user-defined configurations required to 
set up the Clusters and initialize the dask clients. A corresponding `requirements.txt`
file for each ClientPlug is located `solar-data-tools/sdt_dask/clients/requirements/`

### 1. LocalClient ClientPlug (`clients/local_client.py`)
* **Description**: Initializing Dask Client on Local system
* **Initialization**: 
  * `workers`: Number of workers to create in the Local clusters. This number 
  should be less than the number of cores on the system. 
  * `threads`: Number of threads each worker can utilize during computation.
  * `memory`: Amount of memory in GB each worker can have. Should be less than 
  the Local system's Total Memory.
  * `**kwargs`: Additional arguments to configure the cluster. In this example 
  we have enabled multithreading by disabling process spawning in the Cluster.
  
```python
client_setup = LocalClient(workers=3,
                           threads=2, 
                           memory=6.0, 
                           processes='False')
client = client_setup.init_client()
```

Note: For more information on Local cluster arguments refer [this](https://docs.dask.org/en/stable/deploying-python.html#reference)

### 2. FargateClient ClientPlug (`clients/aws/fargate_client.py)
* **Description**: Initializing Dask Client on Fargate's ECS Service
* **Initialization**: 
  * `workers`: Number of workers to create in the Fargate clusters. This spawns 
  each worker and scheduler as a task on the Fargate Cluster.
  * `threads`: Number of threads each worker can utilize during computation. 
  Depends on the number of vCPUs allotted to each worker. Depends on the number
  of vCPUS available on the selected CPU version.
  * `memory`: Amount of memory in GB each worker can have. It is recommended to 
  keep the memory at 16 GB.
  * `**kwargs`: Additional arguments to authorize and configure the cluster.

Note: Requires aws cli, AWS credentials to be set in environment variables, 
  
```python
client_setup = FargateClient(workers=3,
                             threads=2,
                             memory=16,
                             image=IMAGE,
                             tags=TAGS,
                             vpc=VPC,
                             region_name=AWS_DEFAULT_REGION,
                             environment=ENVIRONMENT)
client = client_setup.init_client()
```

Note: For more information on Fargate Cluster arguments refer [this](https://cloudprovider.dask.org/en/latest/aws.html)

### 3. AzureClient ClientPlug (`client/azure/azure_client.py`)
* **Description**: Initializing Dask Client on Azure VM Service
* **Initialization**: 
  * `workers`: Number of workers to create in the Azure VM clusters. This spawns 
  each worker and scheduler as a resource on the Azure VM Cluster.
  * `threads`: Number of threads each worker can utilize during computation. 
  Depends on the number of CPUs available in CPU version. 
  * `memory`: Amount of memory in GB each worker can have. It is recommended to 
  keep the memory at 16 GB.
  * `**kwargs`: Additional arguments to authorize and configure the cluster.

Note: Requires azure cli and credentials to be set in environment variables, 
  
```python
client_setup = AzureClient(workers=3,
                           threads=2,
                           memory=16,
                           resource_group=resource_group,
                           vnet=vnet,
                           security_group=security_group,
                           docker_image=image,
                           location=location,
                           vm_size=cpu,
                           public_ingress=True,
                           disk_size=30)
client = client_setup.init_client()
```

Note: For more information on Azure VM Cluster arguments refer [this](https://cloudprovider.dask.org/en/latest/azure.html)

## Docker Image

### SDT Docker image
Running on the cloud with the clients we've provided (Fargate/Azure) requires a Docker image with the necessary 
packages for the installed on it. We have created an image for you to use if you'd like to run with no special packages 
for dataplugs (we include `boto3` in the environment). To use this image, simply pass `slacgismo/sdt-v1:latest` to the `image` arguments 
when instantiating the clients. 

Note that your local environment needs to have Python 3.12 installed and needs to match the provided Docker image (`slacgismo/sdt-v1:latest`) if you'd like to use it to run on the cloud. The full list of packages along with their versions is listed in [here](./clients/sdt-v1_full_pip_list.txt). The main points of mismatch are typically the following packages:
```bash
    "numpy==2.0", 
    "dask==2024.5.2", 
    "distributed==2024.5.2", 
    "dask-cloudprovider[all]==2022.10.0",
```

If any additional package needed or if you require other versions, you will need create you own image. 
We provide a sample Dockerfile [here](./clients/Dockerfile), and more instructions below.


### Creating your own
The example below explains step by step on 
creating a basic Docker image for the current version of `develop-dask` branch 
of the git repo.

#### Example:
In a terminal inside the directory where the docker file is present run the 
command: 
```shell
docker build -t <YOUR_IMAGE_NAME> .
```
```shell
docker tag <YOUR_IMAGE_NAME>:tag <YOUR_Dockerhub_ID>/<YOUR_IMAGE_NAME>:tag
```

```shell
docker push <YOUR_Dockerhub_ID>/<YOUR_IMAGE_NAME>:tag
```
A basic `DockerFile` content:

`DockerFile`:
```dockerfile
FROM python:3.12 as base

WORKDIR /root
RUN mkdir sdt
WORKDIR /root/sdt

COPY ../../requirements.txt /root/sdt/.

RUN pip install -r requirements.txt
```
A sample `requiremnets.txt` can be viewed in `clients/requirements.txt`.

The Docker image can now be used by plugging the image into the cluster as 
demonstrated below.

FargateClient:
```python
client_setup = FargateClient(workers=3,
                             threads=2,
                             memory=16,
                             image="<YOUR_Dockerhub_ID>/<YOUR_IMAGE_NAME>:tag",
                             tags=TAGS,
                             vpc=VPC,
                             region_name=AWS_DEFAULT_REGION,
                             environment=ENVIRONMENT)
client = client_setup.init_client()
```
AzureClient:

```python
client_setup = AzureClient(workers=3,
                           threads=2,
                           memory=16,
                           resource_group=resource_group,
                           vnet=vnet,
                           security_group=security_group,
                           docker_image="<YOUR_Dockerhub_ID>/<YOUR_IMAGE_NAME>:tag",
                           location=location,
                           vm_size=cpu,
                           public_ingress=True,
                           disk_size=30)
client = client_setup.init_client()
```

## Configuring the Dask Client
By default, Dask has strict configurations for worker's memory limit which includes 
pausing, restarting and terminating workers based on their memory usage. These setting 
can be changed by accessing the dask configuration and setting the required values as shown 
with a custom LocalClient plug and usage example.

### Example:
```python
import dask.config
from dask.distributed import Client, LocalCluster
from sdt_dask.clients.clientplug import ClientPlug

class LocalClient(ClientPlug):
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

    def init_client(self) -> Client:
        self.cluster = LocalCluster(n_workers=self.workers,
                                    threads_per_worker=self.threads,
                                    memory_limit=f"{self.memory:.2f}GiB",
                                    **self.kwargs)
        self.client = Client(self.cluster)

        print(f"\nDask Dashboard Link: {self.client.dashboard_link}\n")

        return self.client

client_setup = LocalClient(workers=3,
                           threads=2, 
                           memory=6.0)
# When worker's memory spill is False, workers do not use disk memory to store 
# excess data. This causes data transfers between workers.
# When set to True, workers write data to disk which prevents them for rerunning
# tasks when exceeding memory limit
client_setup.dask_config.set({'distributed.worker.memory.spill': False})

# Here worker's will not pause if they exceed memory limits. When set to True, 
# workers will pause when memory is exceeded.
client_setup.dask_config.set({'distributed.worker.memory.pause': False})

# This defines how much of the alloted memory a worker can use, Here 0.95 means 
# 95% of the alloted memory, 95% of 6.0 GB
client_setup.dask_config.set({'distributed.worker.memory.target': 0.95})

# Here the workers are prevented from being terminated when they reach or exceed
# memory limit
client_setup.dask_config.set({'distributed.worker.memory.terminate': False})

client = client_setup.init_client()
```

Note: For more information on dask configuration please visit [link](https://docs.dask.org/en/latest/configuration.html)
