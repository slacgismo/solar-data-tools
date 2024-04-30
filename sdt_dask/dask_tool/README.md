This README provides guidance on using sdt Dask runner

## Using Runner

To use Runner class, you will provide a Dataplug and a Dask client to initialize the class. Then provide a list of `keys` and custom arguments you wish pass to the `run_pipeline` and call `set_up(keys, **kwargs)`. Keys should be a list of tuple according to dataplug's requirement. Finally, call `get_result()` to save the dask report and summary report to target directory.

```python
# example for runner
dask_tool = Runner(local_file_data_plug, local_client, output_path="../results/")
dask_tool.set_up(local_file_keys, fix_shifts=True, verbose=True)
dask_tool.get_result(dask_report = output_html, summary_report = output_csv)
```

Examples usage of all valid combination of provided dataplug and client can be found inside examples directory.

 
