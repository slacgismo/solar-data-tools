# Solar Data Tools User Guide

This guide serves as a reference for the typical Solar Data Tools user. We will go over the main 
classes and methods that the packages offers and that are meant to be user-facing, with examples
and relevant details for you to get started. If any part of this guide is unclear, or you'd like 
to improve on these docs, please consider [submitting an Issue or a PR](index_dev.md)!

## Main User-Facing Classes and Methods

The main class that users will interact with--as you'll see in the next section--is the DataHandler class.
<!The table below lists all the associated class methods that a typical user may call. The table also 
lists some data I/O functions that may be useful for pulling data from publicly available sources:

For a more comprehensive list of all methods and functions, you can check the [API reference](index_api_reference.rst).
**However, this user guide likely provides you with all the functionality you need, while the rest of 
the functions listed in the API reference are more internal/helper functions that were not meant for users to interact with.**>


## Getting started with the `DataHandler` class
Most users will only need to interact with the `DataHandler` class. To instantiate, a Data Handler 
object takes a DataFrame containing the power data timeseries (with a timestamps and power columns) 
as an input. 

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

## Running the pipeline
The DataHandler.run_pipeline method is the main data processing and analysis pipeline offered by 
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
the pipeline. For example, 
## Running loss factor analysis 



## Other features

### Latitude and longitude estimation