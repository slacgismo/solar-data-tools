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

Note: For more information on Local cluster arguments, refer to [the Dask docs.](https://docs.dask.org/en/stable/deploying-python.html#reference)

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

We recommend using Docker images to run SDT on cloud-based clusters. For more information on the Docker 
image we provide, see [the Docker README](../../docker/README.md).

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

Note: For more information on Fargate Cluster arguments, refer to [the Dask docs.](https://cloudprovider.dask.org/en/latest/aws.html)

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

We recommend using Docker images to run SDT on cloud-based clusters. For more information on the Docker 
image we provide, see [the Docker README](../../docker/README.md).

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

Note: For more information on Azure VM Cluster arguments, refer to [the Dask docs.](https://cloudprovider.dask.org/en/latest/azure.html)


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

Note: For more information on the Dask configuration please visit [the Dask docs.](https://docs.dask.org/en/latest/configuration.html)
