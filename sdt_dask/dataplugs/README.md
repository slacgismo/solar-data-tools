
This README provides guidance on creating your own DataPlugs or use our existing Dataplugs within the SDT Dask tool. DataPlugs are classes used by runner retrieve and clean data from various sources. Follow these instructions to create your own DataPlug or use the existing data plug examples.

## Creating Your Own DataPlug

To create your own DataPlug, you must a Python module (`your_dataplug.py`) containing your DataPlug class. You also 
must install all necessary Python packages that aren't part of the SDT requirements (if you are using Docker
to run on the cloud, make sure to create your own image of the SDT environment and install your dataplug 
requirements on it as well. For more info see [the Docker README](../docker/README.md).

### Implement Your DataPlug

1. **Inherit from the Base DataPlug Class**: Any DataPlug class should inherit from the DataPlug base class. This inheritance guarantees that your DataPlug aligns with the expected structure and can seamlessly integrate with the SDT Dask tool.
2. **Define Initialization Parameters**: Customize the __init__ method to accept parameters specific to your data source. These parameters can vary widely depending on the nature of the data source, such as file paths.
3. **Implement `get_data` Method**: Core method for DataPlug, it retrieves and cleans the data before returning a pandas DataFrame. The method should accept a keys argument as a tuple.

4. **Important - Non-Serializable and Not-Thread-Safe Object**:
Since get_data method is delayed and distributed to tasks across Dask workers, we should avoid sharing any non-serializable or not-thread-safe object by creating these objects inside each delayed function called. Non-serializable object refers to the object that maintain state, open connections, or hold resources, such as botocore.client.S3 instances and botocore.client.S3 instances. You should generate new instances of resources or clients for each thread or operational context. As an example on how to create botocore.client.S3 instances in your dataplug, you could look at the provided S3 dataplug implementation.

5. **(Optional) Additional Methods**: Beyond get_data, you may implement any number of private or public methods to aid in data retrieval, transformation, or cleaning. Cleaning is a very important part that normally will be called in get_data method. Other possible examples include methods for parsing file names.

Example structure:

```python
import pandas as pd
from sdt_dask.dataplugs.dataplug import DataPlug

class YourDataPlug(DataPlug):
    def __init__(self, param1, param2):
        # Initialization code here

    def get_data(self, keys: tuple) -> pd.DataFrame:
        # Data retrieval and cleaning code here
        return df
```

## Existing DataPlug Examples

Below are detailed descriptions of the DataPlugs available for use with the SDT Dask tool. Each DataPlug serves a unique purpose, designed to retrieve and process data from different sources. A corresponding requirements.txt file for each DataPlug is located in the directory solar-data-tools/sdt_dask/dataplugs/Requirements/. 

### 1. LocalFiles DataPlug (dataplugs/csv_plug.py)

- **Description**: Retrieving and cleaning solar data from local CSV files
- **Initialization**:   
`path_to_files`: The directory path where the CSV files are stored.  
- **`get_data` Tuple input**: Expects a single string value representing the filename (without extension) of the dataset to be processed. Example to call `get_data` method:  
```python 
data_plug.get_data(("filename",))
```

### 2. PVDAQPlug (dataplugs/pvdaq_plug.py)

- **Description**: Retrieving and cleaning solar data from the PVDAQ database
- **Initialization**:   
`api_key`: Your API key for accessing the PVDAQ data.
- **`get_data` Tuple input**: Expects a tuple containing the site ID (integer) and the year (integer) for which data is to be retrieved. Example to call get_data method:
```python 
data_plug.get_data((site_id, year))
```


### 3. PVDBPlug (dataplugs/pvdb_plug.py)

- **Description**: Retrieving and cleaning solar data from the PVDB (Redshift) database
- **Initialization**: No input but assumes the API key is set as an environment variable REDSHIFT_API_KEY.
- **`get_data` Tuple input**: Expects a tuple containing the site ID (string) and the sensor type (integer), identifying the specific dataset to be retrieved. Example to call get_data method:
```python 
data_plug.get_data(("site_id", sensor_type))
```

### 4. S3Bucket DataPlug

- **Description**: Retrieving and cleaning solar data from S3 Bucket. And provides a function to get the full key list inside the given bucket name.
- **Initialization**:  
`api_key`: Your API key for accessing the PVDAQ data.  
And also assumes the AWS configuration has been set up in local environment
- **`get_data` Tuple input**: Expects a single string value in the tuple, specifying the key of the file in the bucket. Example to call get_data method:
```python 
data_plug.get_data(("s3-file-key",))
```