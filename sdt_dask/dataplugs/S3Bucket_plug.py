"""Class for S3 bucket from the AWS DB"""

import pandas as pd
import boto3
from solardatatools.time_axis_manipulation import make_time_series
from sdt_dask.dataplugs.dataplug import DataPlug


class S3Bucket(DataPlug):
    """ Dataplug class for retrieving data from an S3 bucket.
    aws configurations for the AWS CLI must be set up in local environment
    """
    def __init__(self, bucket_name):
        """Initialize the S3Bucket object with the bucket name.

        :param bucket_name: The name of the S3 bucket. (type: str)
        """
        self.bucket_name = bucket_name

    def _pull_data(self, key):
        
        s3_client = boto3.client('s3')
        print(f"Loading file from S3 bucket: {key}...")
        obj = s3_client.get_object(Bucket=self.bucket_name, Key=key)

        # Assume file is CSV
        self.df = pd.read_csv(obj['Body'])

    def _clean_data(self):
        # Convert index from int to datetime object
        self.df, _ = make_time_series(self.df)

    def get_data(self, key: tuple[str]) -> pd.DataFrame:
        """
        This is the main function that the Dask tool will interact with.
        Users should keep the args and returns as defined here when writing
        their custom dataplugs.

        Note: if this example dataplug is used, the data in the S3 bucket should
        be in CSV format and the format should be consistent across all files: a
        timestamp column and a power column.

        :param key: Filename (which could be get by _pull_keys method) inside
            the tuple
        :return: Returns a pandas DataFrame with a timestamp column and
            a power column
        """
        self._pull_data(key[0])
        self._clean_data()

        return self.df

    def _pull_keys(self) -> list:
        """
        Retrieves a list of file keys from a specified S3 bucket.

        :return: A list of strings representing the file keys without their extensions. (type: list)
        """
        KEYS = []

        s3_client = boto3.client('s3')
        objects = s3_client.list_objects_v2(Bucket=self.bucket_name)
        
        if 'Contents' in objects:
            for item in objects['Contents']:
                filename = item['Key']
                KEYS.append(filename)
        
        return KEYS
