# Solar Data Runner Tools

The solar data runner tool mainly aimed at serving as a plug to preprocess data input for pipeline execution with the Dask framework for parellel computing. Currently, it has been engineered to serve local CSV files and remote databases.

Below will be a more thorough description on the prerequisites to set up the runner tool and its usages, as well as the key implementations of both the local and remote dataplug and improvements/tasks for future works.

## Setting up runner tool

Make sure you have the following installed on your machine:

- **Python:** Jupyter Notebooks run on Python, so make sure you have Python installed. You can download it from [python.org](https://www.python.org/).

- **solar-data-tools:** Make sure to have the correct version of solar-data-tools:1.0.4 installed. Or catastrophic errors will happen!

- **Jupyter Notebook:** Install Jupyter Notebook using the following command in your terminal or command prompt:

```bash
  pip install notebook
```

## local CSVs

The local csv dataplug works with local csv files, where each you would have to provide a directory containing one(or multiple) csv files, and each csv containing only one column for pipeline execution. Each column contains the data for a given system to be analyzed.

The key local dataplug methods are:

- `local_csv_to_dh`

  Converts a local CSV file into a solar-data-tools DataHandler.

- `run_job`

  Processes a single unit of data using DataHandler.

- `generate_task_local`

  Generate the analysis task for a given local csv file.

- `generate_tasks_directory`

  Generate the analysis tasks for a given directory containing csv's.

- `write_reports`

  Aggregate reports and write output to a csv file.

The usage of the local dataplug with Dask is as below:

```python
dir_csv = "" # directory containing csv's
obj_list = generate_tasks_directory(dir_csv)

# using the delayed method enables Dask parellel computing at a latter point of time
out_csv = ""
aggregate_reports_task = delayed(write_reports)(obj_list, out_csv)

# run a Dask cluster
client = Client(threads_per_worker=4, n_workers=2)

dask.compute(aggregate_reports_task)

client.shutdown()
```

More demo is included in the `demo.ipynb` notebook.

## remote database

Currently the remote database used is a cassandra database. The cassandra database containes data to be analyzed for different sites. Each site contains data for one(or multiple) system to be analyzed and execution of pipeline.

The key remote dataplug methods are:

- `run_job`

  Processes a single unit of data using DataHandler.

- `remote_site_to_dhs`

  Converts a site of multiple systems into a list of solar-data-tools DataHandlers.

- `generate_tasks_remote_database`

  Generate analysis tasks for remote database.

The usage of the remote dataplug with Dask is as below:

```python
db_list = "" # txt file of one(or multiple sites separated by newlines)
obj_list = generate_tasks_remote_database(db_list)

# run a Dask cluster
client = Client(threads_per_worker=4, n_workers=2)

dask.compute(obj_list)

client.shutdown()
```

More demo is included in the `demo.ipynb` notebook.

## Goal for remote database plug

There exists a major issue with the current implementation of the remote dataplug.

Currently, the remote dataplug is engineered in a way where the import of each function is in a unidirectional way, whereas the ideal dataplug should serve as a component for a generic purpose, allowing other plugs to subscribe to it.
