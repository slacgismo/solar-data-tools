# Solar Data Tools Dask Tool

>[!WARNING]
> This feature is under active development. If you'd like to use this, contribute, or have any 
> questions, feel free to open an issue or submit a PR. You can find our Contribution Guidelines 
> [here](https://solar-data-tools.readthedocs.io/en/dev/index_dev.html).

We developed the Solar Data Tools (SDT) Dask Tool to provide users with a convenient way to run the SDT 
pipelines and analyses on large amounts of data in a scalable and parallelized way, either locally 
or on various computing infrastructures. You can read more about the goals of the project and the development process 
in our PVSC paper [here](https://drive.google.com/file/d/1uczjlfNChn6qM8hn6ary5NLaR2QmLEhz/view?usp=drive_link).

This tool is a configurable system that is accessible and relatively easy to run for users with different computational environments.
We use Dask to provide cloud deployment and parallelization of our pipelines, but you don't need to have Dask experience to run it. The tool has three main components, two of which need to be defined by the use: where the data comes from ("dataplug") and where to run the pipelines ("client"). The third is
the Dask runner that uses the other two components to pull the data and execute the SDT pipelines and computations. 

To get started, take a look at the local run example [here](./examples/tool_demo_local.ipynb) or the AWS Fargate example [here](./examples/tool_demo_fargate.ipynb). Additionally,
each component has a README.md with more information. See the dataplugs README [here](dataplugs/README.md) and the clients README [here](clients/README.md).

## Installation

To install the development version of this tool, in a fresh Python environment with Python 3.10 installed, run from the root of the project:

```bash
$ pip install -e ".[dask]"
```

Note that it's important to have your local environment (Python package versions) match the provided Docker image (`smiskov/sdt-v1:latest`) if you'd like to use it to run on AWS or Azure.
The full list of packages along with their versions is listed in [here](./clients/sdt-v1_full_pip_list.txt).
Otherwise, feel free to create you own image. We provide a sample Dockerfile [here](./clients/Dockerfile).

Any additional package needed for the dataplugs should be installed separately (e.g. boto3).

## Basic Usage 

To use the tool, you'll need a Runner class. You will need to provide a Dask Client to instantiate, and optionally the directory where you'd like the output saved. 
Then provide a Dataplug, a list of keys, and any custom arguments you wish pass to the `run_pipeline` and call `set_up(keys, dataplug, **kwargs)`. 
Keys should be a list of tuples according to the dataplug requirements (see the dataplugs [documentation](dataplugs/README.md) for more info). 
Finally, call `get_result()` to run and save the dask report and summary report.

The following is an example using a local client and local CSVs:
```python
from sdt_dask.dask_tool.runner import Runner
from sdt_dask.dataplugs.csv_plug import LocalFiles
from sdt_dask.clients.local_client import LocalClient

client_setup = LocalClient(workers=4, threads=3, memory=5)
client = client_setup.init_client()

dataplug = LocalFiles(path_to_files=path_to_files)
keys = some_list_of_file_names

runner = Runner(client, output_path="../results/")
runner.set_up(keys, dataplug, fix_shifts=True, verbose=True)
runner.get_result()
```

Examples usage of other combinations of provided dataplugs and clients can be found in the [examples](examples) directory.
