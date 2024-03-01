"""Class for files stored in an Amazon S3 bucket."""

import pandas as pd
import boto3
from sdt_dask.dataplugs.dataplug import DataPlug

class S3BucketPlug(DataPlug):
    """Dataplug class for retrieving data from an Amazon S3 bucket.
    It's designed to read files specified by their S3 object keys
    and combine them into a single pandas DataFrame.
    """
    def __init__(self):
        self.s3_client = boto3.client('s3')

    def _read_files(self, keys):
        # print(f"Loading file from S3 bucket: {key}...")
        obj = self.s3_client.get_object(Bucket=keys[0], Key=keys[1])

        # Assume file is CSV
        self.df = pd.read_csv(obj['Body'])

    def _clean_data(self):
        pass


    def get_data(self, keys: tuple[str, str]) -> pd.DataFrame:
        """This is the main function that interacts with the Dask tool.
        It reads data from the specified S3 object keys, optionally cleans it,
        and returns a combined pandas DataFrame.

        :param keys: Tuple containing the bucket and S3 object key as strings.
        :return: A pandas DataFrame containing the combined data from all keys.
        """
        self._read_files(keys)
        self._clean_data()

        return self.df
