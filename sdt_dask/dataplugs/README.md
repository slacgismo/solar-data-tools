
This README provides guidance on creating your own DataPlugs or use our existing Dataplugs within the SDT Dask tool. DataPlugs are classes used by runner retrieve and clean data from various sources. Follow these instructions to create your own DataPlug or use the existing data plug examples. The demo code to use each example dataplug are provided under `dataplugs/examples/`

## Creating Your Own DataPlug

To create your own DataPlug, you must provide two key files: a Python module (`your_dataplug.py`) containing your DataPlug class and a `requirements.txt` file listing all necessary external Python packages.

### Implement Your DataPlug

1. **Inherit from the Base DataPlug Class**: Any DataPlug class should inherit from the DataPlug base class. This inheritance guarantees that your DataPlug aligns with the expected structure and can seamlessly integrate with the SDT Dask tool.
2. **Define Initialization Parameters**: Customize the __init__ method to accept parameters specific to your data source. These parameters can vary widely depending on the nature of the data source, such as file paths.
3. **Implement `get_data` Method**: Core method for DataPlug, it retrieves and cleans the data before returning a pandas DataFrame. The method should accept a keys argument as a tuple.

4. **Important - Non-Serializable and Not-Thread-Safe Object**:
Since get_data method is delayed and distributed to tasks across Dask workers, we should avoid sharing any non-serializable or not-thread-safe object by creating these objects inside each delayed function called. Non-serializable object refers to the object that maintain state, open connections, or hold resources, such as botocore.client.S3 instances. AND, botocore.client.S3 instances are not-thread-safe as well. You should generate new instances of resources or clients for each thread or operational context. As a example, you could look at the s3 dataplug example on how to create botocore.client.S3 instances as an example.

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
- **Important Notice 1.**: As mentioned above, botocore.client.S3 object include open connections and internal state, they cannot be correctly serialized and safely transferred across workers. Thus, we should initialize the client instance for each get_data requests. In general, it applies to all non-serializable objects.

- **Important Notice 2.**: To ensure thread safety, a new boto3.session.Session() should be created and then creat a new client instance. This is because Boto3 client is not thread safe and should be avoided to share across threads. By first creating the session then creating the instance, it prevents potential data sharing and synchronization issues when accessing AWS resources concurrently across threads. In general, it applies to all non-thread-safe objects.