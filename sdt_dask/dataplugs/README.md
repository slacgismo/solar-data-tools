
This README provides guidance on creating and using DataPlugs with the SDT Dask tool. DataPlugs are Python classes designed to retrieve and clean data from various sources for analysis and processing with Dask. Follow these instructions to create your own DataPlug or use the existing data plug examples. The demo code to use each example dataplug are provided under `dataplugs/examples/`

## Creating Your Own DataPlug

To create your own DataPlug, you must provide two key files: a Python module (`your_dataplug.py`) containing your DataPlug class and a `requirements.txt` file listing all necessary external Python packages.

### Implement Your DataPlug

1. **Inherit from the Base DataPlug Class**: Ensure your custom DataPlug class inherits from the DataPlug base class. This inheritance guarantees that your DataPlug aligns with the expected structure and can seamlessly integrate with the SDT Dask tool.
2. **Define Initialization Parameters**: Customize the __init__ method to accept parameters specific to your data source. These parameters can vary widely depending on the nature of the data source, such as file paths, API keys, database credentials, or any other configuration necessary for data access.
3. **Implement `get_data` Method**: This method is the core of your DataPlug, tasked with retrieving and cleaning the data before returning a pandas DataFrame. The method should accept a keys argument as a tuple, which contains the necessary identifiers or parameters to fetch the specific dataset. This flexible approach allows for a wide range of data retrieval scenarios, accommodating various data sources and user requirements.

4. **Important - Non-Serializable and Not-Thread-Safe Object**:
When distributing tasks across Dask workers, avoid using pre-initialized instances of objects that maintain state, open connections, or hold resources that cannot be serialized (e.g., botocore.client.S3 instances). These objects should not be serialized or transferred across processes due to their internal state and open connections. Instead, create and utilize such instances within the scope of each task. This guidance ensures that each task independently manages its resources, enhancing process safety and stability. This principle applies broadly to all non-serializable objects used in distributed computing tasks. MEANWHILE, to guarantee thread safety across our application, you are expected to adopt a consistent pattern of creating dedicated instances for any object that is not thread-safe to share across threads or processes. This involves generating new instances of resources or clients for each thread or operational context. This strategy is essential for preventing data sharing and synchronization issues in concurrent environments, ensuring that each thread operates with its own isolated instance.

5. **(Optional) Additional Methods**: Beyond get_data, you may implement any number of private or public methods to aid in data retrieval, transformation, or cleaning. Examples include methods for parsing file names, performing complex queries on databases, or applying specific data cleaning operations tailored to your data source.



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

- **Description**: Load and process data stored in local CSV files
- **Initialization**:   
`path_to_files`: The directory path where the CSV files are stored.  
- **`get_data` Tuple input**: Expects a single string value representing the filename (without extension) of the dataset to be processed. Example to call `get_data` method:  
```python 
data_plug.get_data(("filename",))
```

### 2. PVDAQPlug (dataplugs/pvdaq_plug.py)

- **Description**: Retrieves solar power generation data from the PVDAQ database via an API. Suitable for solar energy data analysis, enabling access to site-specific power generation data.
- **Initialization**:   
`api_key`: Your API key for accessing the PVDAQ data.
- **`get_data` Tuple input**: Expects a tuple containing the site ID (integer) and the year (integer) for which data is to be retrieved. Example to call get_data method:
```python 
data_plug.get_data((site_id, year))
```


### 3. PVDBPlug (dataplugs/pvdb_plug.py)

- **Description**: Designed for retrieving data from the PVDB (Redshift) database, this DataPlug suits advanced data analysis needs involving large datasets stored in Amazon Redshift.
- **Initialization**: No input but assumes the API key is set as an environment variable REDSHIFT_API_KEY.
- **`get_data` Tuple input**: Expects a tuple containing the site ID (string) and the sensor type (integer), identifying the specific dataset to be retrieved. Example to call get_data method:
```python 
data_plug.get_data(("site_id", sensor_type))
```

### 4. S3Bucket DataPlug

- **Description**: Facilitates the retrieval and processing of data stored in AWS S3 buckets. Ideal for users utilizing AWS S3 for data storage, enabling direct analysis of CSV files from S3.
- **Initialization**:  
`api_key`: Your API key for accessing the PVDAQ data.  
And also assumes the AWS configuration has been set up in local environment
- **`get_data` Tuple input**: Expects a single string value in the tuple, specifying the key of the file in the bucket. Example to call get_data method:
```python 
data_plug.get_data(("s3-file-key",))
```
- **Important Notice 1.**: You should avoid using pre-initialized S3 client objects within Dask tasks that are being distributed to Dask workers. Since instances of botocore.client.S3 include open connections and internal state, they cannot be correctly serialized and safely transferred across processes. Instead, you should create and use S3 client instances within each task, ensuring that the use of S3 clients is confined to the execution scope of individual tasks. And this notice applies to all objects that maintains state, open connections, or resources that are not serializable.

- **Important Notice 2.**: To ensure thread safety, a new boto3.session.Session() should be created and subsequently a new client instance for each operation. Boto3 client is not thread safe and should be avoided to share across threads. This approach addresses the issue of resource instances not being thread-safe and prevents potential data sharing and synchronization issues when accessing AWS resources concurrently across threads or processes. It's vital to apply this pattern for any object that is not safe to share across threads, ensuring isolated instances per thread for reliable and error-free operation.