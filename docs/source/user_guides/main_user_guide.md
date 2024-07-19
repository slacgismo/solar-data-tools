# Solar Data Tools User Guide

```{danger}
This page is under development!
```

This guide serves as a reference for the typical Solar Data Tools user. We will go over the main 
classes and methods that the packages offers, with examples and relevant details for you to get started.

```{warning}
For a more comprehensive list of all methods and functions, you can check the [API reference](../reference/api.rst).
However, note that this user guide likely provides you with all the functionality you need, while the rest of 
the functions listed in the API reference are more internal/helper functions that were not meant for users to interact with.
```

If any part of this guide is unclear, or you'd like to improve on these docs, please consider [submitting an Issue or a PR](../index_dev.md)!

## Getting started with the `DataHandler` class
Most users will only need to interact with the `DataHandler` class. To instantiate, a Data Handler 
object takes a DataFrame containing the power data timeseries (with a timestamps and power columns) 
as an input. 

## Running the pipeline
The DataHandler.run_pipeline method is the main data processing and analysis pipeline offered by 
Solar Data Tools. It includes preprocessing, cleaning (e.g. fixing time shifts), and scoring data 
quality metrics (e.g. finding clear days, capacity changes and any clipping.

## Running loss factor analysis 

## Plotting

## Other features

### Latitude and longitude estimation

