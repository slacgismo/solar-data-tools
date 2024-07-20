## Using The Dask SDT Runner

To use the Runner class, you will provide a Dask Client to initialize the class and optionally the directory where you'd like the output saved. Then provide a Dataplug, a list of keys, and any custom arguments you wish pass to the `run_pipeline` and call `set_up(keys, dataplug, **kwargs)`. Keys should be a list of tuples according to the dataplug requirements (see the dataplugs documentation for more info). Finally, call `get_result()` to run and save the dask report and summary report.

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

Examples usage of other combinations of provided dataplugs and clients can be found in the [examples](../examples/) directory.

 
