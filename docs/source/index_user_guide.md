# Solar Data Tools User Guide

This guide serves as a reference for the typical Solar Data Tools user. We will go over the main
user-facing classes and methods, with examples and relevant details for you to get started. As you
will see, most of the methods in Solar Data Tools are automated, so you can get started with minimal
effort and very little input from your side.

For a more comprehensive list of all methods and functions, you can check the [API reference](index_api_reference.rst).
**However, this user guide likely provides you with all the functionality you need, while the rest of
the functions listed in the API reference are more internal/helper functions that were not meant for
users to interact with.**

If any part of this guide is unclear, or you'd like
to improve on these docs, please consider [submitting an Issue or a PR](index_dev.md)!

## Getting started with the `DataHandler` class
Most users will only need to interact with the `DataHandler` class. To instantiate, a Data Handler
object takes a DataFrame containing the power data timeseries (with a timestamps and power columns)
as an input. Note that the DataFrame must have a DatetimeIndex, or the user must set the `datetime_col` kwarg.

The timestamps are recommended to be in the local timezone of the data. If there is a small shift in the timestamps,
the pipeline will attempt to correct it. If the shift is large (8-10 hours), the pipeline will likely fail to
adjust the shift.

Let's say we have a CSV file with power data that we want to analyze. We can load it into a DataFrame:

```python
import pandas as pd
from solardatatools import DataHandler

df = pd.read_csv('path/to/your/data.csv')
```
Then we can create a DataHandler object with this DataFrame:

```python
dh = DataHandler(df)
```

If you know that your data is affected by daylight savings, you can run the following method to correct
for it:

```python
dh.fix_dst()
```

The DataHandler object is now ready to be used for data processing and analysis.

### A note on long-form vs. wide-form data

Timeseries data is often in wide-form, where you have for example a DataFrame that has a timestamp
column and one or more data columns. That's what the DataHandler typically expects. However,
it also can take data in long-form, such as for example what we have in the Redshift data where some sites
have more than one inverter (see the "Data I/O functions" section below). In this case, you will
want to instantiate the DataHandler object with the `convert_to_ts` flag set to True:

```python
dh = DataHandler(df, convert_to_ts=True)
```

This prompts the DataHandler to convert the data to wide-form before running the pipeline, given
default index and column names intended to work with GISMo's VADER Cassandra database implementation
(see `solardatatools.time_axis_manipulation.make_time_series`).

For more information on long-form vs. wide-form, you can check out [this nice writeup](https://seaborn.pydata.org/tutorial/data_structure.html#long-form-vs-wide-form-data) from
the Seaborn documentation.

## Running the pipeline
The `DataHandler.run_pipeline` method is the main data processing and analysis pipeline offered by
Solar Data Tools. It includes preprocessing, cleaning (e.g. fixing time shifts), and scoring data
quality metrics (e.g. finding clear days, capacity changes and any clipping.

To run the pipeline, simply call the method:

```python
dh.run_pipeline()
```

This method can be passed a number of optional arguments to customize the pipeline. For example, you may
need to specify the timezone of the data, or the solver to use for the capacity change detection.
Most commonly, you will want to specify the power column name and whether to run a
timeshift correction. Here is an example of how to run the pipeline with these arguments:

```python
dh.run_pipeline(power_col='power', fix_shifts=True)
```

Note that the pipeline can take a while to run, depending on the size of the dataset and the solver
you are using (from a couple of seconds up to a minute).

Once the pipeline is run, the DataHandler object will have a number of attributes that you can access
to view the results of the analysis. The top-level report is accessed by calling the `report` attribute:

```python
dh.report()
```

This will print a summary of the results of the pipeline, including the data quality metrics.
We can also make a machine-readable version, which is useful when processing many files/columns
for creating a summary table of all data sets.

```python
dh.report(return_values=True, verbose=False)
```

## Plotting some pipeline results

The DataHandler object has a number of plotting methods that can be used to visualize the results of
the pipeline. Here is a full list of the plotting methods available after running the main pipeline:
| Method | Description|
| --- | --- |
|    DataHandler.plot_heatmap | Plot a heatmap of the data |
|    DataHandler.plot_bundt | Make a "[Bundt plot](https://ieeexplore.ieee.org/abstract/document/10749393)" of the data |
|    DataHandler.plot_circ_dist | Plot the circular distribution of the data |
|    DataHandler.plot_daily_energy | Plot the daily energy |
|    DataHandler.plot_daily_signals | Plot the daily signals |
|    DataHandler.plot_density_signal | Plot the density signal |
|    DataHandler.plot_data_quality_scatter | Plot the data quality scatter |
|    DataHandler.plot_capacity_change_analysis | Plot the capacity change analysis |
|    DataHandler.plot_time_shift_analysis_results | Plot the time shift analysis results |
|    DataHandler.plot_clipping | Plot the clipping |
|    DataHandler.plot_cdf_analysis | Plot the CDF analysis |
|    DataHandler.plot_daily_max_cdf_and_pdf | Plot the daily max CDF and PDF |
|    DataHandler.plot_polar_transform | Plot the polar transform |

Note that the timeshift correction method must be run (by passing `fix_shifts=True` to `run_pipeline`)
before calling the `plot_time_shift_analysis_results` method.

Examples of some of these plotting methods are shown below in the notebooks and examples section,
such as the [demo](getting_started/notebooks/demo_default.nblink) and the
[tutorial](getting_started/notebooks/tutorial.ipynb).

## Running loss factor analysis

Once the main pipeline is run, you can run the loss factor analysis to estimate the loss factor of
the system. This is done by calling the `run_loss_factor_analysis` method:

```python
dh.run_loss_factor_analysis()
```

This method will estimate the loss factor of the system by running a Monte Carlo sampling to generate
a distributional estimate of the degradation rate. The results are stored in `dh.loss_analysis`.

Once it terminates, you can visualize some of the results by calling the following functions:

| Method                                     | Description                                                         |
|--------------------------------------------|---------------------------------------------------------------------|
| DataHandler.loss_analysis.loss_analysis.plot_pie | Create a pie plot to visualize the breakdown of energy loss factors |
| DataHandler.loss_analysis.plot_waterfall   | Create waterfall plot to visualize the breakdown of energy loss factors |
| DataHandler.loss_analysis.plot_decomposition   | Plot the estimated signal components found through decomposition    |

Head over to our [demo](getting_started/notebooks/demo_default.nblink) and
[tutorial](getting_started/notebooks/tutorial.ipynb) to see these functions in action on real data.

## Running clear sky model estimation

After the main pipeline is run, a clear sky model of the PV system power can be estimated by running:
```python
dh.fit_statistical_clear_sky_model()
```
This fits a *smooth, multiperiodic* model of the instantaneous 90th percentile of the power data, as explained in [this paper](https://ieeexplore.ieee.org/abstract/document/10749393). Under the hood, this invokes the [spcqe package](https://github.com/cvxgrp/spcqe). Check out this [demo](getting_started/notebooks/clearsky_estimation_demo.nblink) for more information.

## Running clear sky data labeling

The clear sky labeling subroutine leverages the results from the main pipeline, the loss factor estimation, and the clear sky model fitting, all described above. After the main pipeline is run, the user may run:
```python
dh.detect_clear_sky()
```
If either or both of the loss factor estimation and clear sky estimation modules have not been run, the Data Handler will run those modules automatically when `detect_clear_sky` is called. (The Data Handler will not re-run these modules if they've already been invoked and will just make use of the outputs.)

Check out this [demo](getting_started/notebooks/clearsky_detection_demo.nblink) for more details.

## Other features

### Orientation and Location estimation
The DataHandler also includes methods to estimate the position of the solar panels based on the data.
This includes the location (latitude and longitude) and orientation (tilt and azimuth) of the system.
The available methods are:

| Method                                                | Description                                                                       |
|-------------------------------------------------------|-----------------------------------------------------------------------------------|
| DataHandler.setup_location_and_orientation_estimation | Sets up the location and orientation estimation for the system given a GMT offset |
| DataHandler.estimate_latitude                         | Estimates latitude                                                                |
| DataHandler.estimate_longitude                        | Estimates longitude                                                               |
| DataHandler.estimate_location_and_orientation         | Estimates latitude, longitude, tilt and azimuth                                   |
| DataHandler.estimate_orientation         | Estimates tilt and azimuth                                                        |

To call the estimation methods, first you need to run the `setup_location_and_orientation_estimation``
method and provide a GMT offset value by passing it to the method. After that, you can call any of the
four estimation methods. A demo of this feature can be found in the [tutorial](getting_started/notebooks/tutorial.ipynb)
in cells 13-15.

### Data I/O functions

The `dataio` module in Solar Data Tools includes a number of functions to pull data from various sources.
These functions are useful for loading data into a DataFrame that can be used with the `DataHandler` class.
The available functions are:

| Method                            | Description                                                                    |
|-----------------------------------|--------------------------------------------------------------------------------|
| dataio.get_pvdaq_data      | Queries one or more years of raw PV system data from NREL's PVDAQ data service |
| dataio.load_constellation_data | Loads constellation data from a specified location                             |
| dataio.load_redshift_data  | Queries a SunPower dataset by site id and returns a Pandas DataFrame           |
| dataio.load_pvo_data    | Loads NREL data from private S3 bucket (for use by the SLAC team only)         |


The PVDAQ database is a public database of solar power data that can be accessed by anyone. The system
locations that can be accessed are shown on [this interactive map](https://openei.org/wiki/PVDAQ/PVData_Map).
You can use the "DEMO_KEY" for querying the data, but you can also get your own API key by
registering [here](https://data.openei.org/submissions/4568).
An example usage for this function for system ID 34 is shown below:
```python
df = get_pvdaq_data(sysid=34, year=range(2011, 2015), api_key='DEMO_KEY')
```

To use the `load_redshift_data` function, you will need to
request an API key by registering at [https://pvdb.slacgismo.org](https://pvdb.slacgismo.org) and emailing
slacgismotutorials@gmail.com with your information and use case. To query the data, you also must
provide a site ID and a sensor number (0, 1, 2 ...). An example usage is shown below:

```python
query = {
    'siteid': 'TABJC1027159',
    'api_key': YOUR_API_KEY,
    'sensor': 0
}

df = load_redshift_data(**query)
```
