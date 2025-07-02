# -*- coding: utf-8 -*-
"""Data IO Module

This module contains functions for obtaining data from various sources.

"""

import pandas as pd
from datetime import datetime


def get_pvdaq_data(sysid=2, api_key="DEMO_KEY", year=2011, delim=",", standardize=True):
    """
    This function queries one or more years of raw PV system data from NREL's PVDAQ data service:
            https://openei.org/wiki/PVDAQ/PVData_Map

    :param sysid: The system ID to query. Default is 2.
    :type sysid: int, optional
    :param api_key: The API key for authentication. Default is "DEMO_KEY".
    :type api_key: str, optional
    :param year: The year or list of years to query. Default is 2011.
    :type year: int or list of int, optional
    :param delim: The delimiter used in the CSV file. Default is ",".
    :type delim: str, optional
    :param standardize: Whether to standardize the time axis. Default is True.
    :type standardize: bool, optional

    :return: A dataframe containing the concatenated data for all queried years.
    :rtype: pd.DataFrame
    """
    # Force year to be a list of integers
    raise (
        "This function is no longer supported! See https://github.com/NREL/pvdaq_access for access to PVDAQ data"
    )


def load_pvo_data(
    file_index=None,
    id_num=None,
    location="s3://pv.insight.nrel/PVO/",
    metadata_fn="sys_meta.csv",
    data_fn_pattern="PVOutput/{}.csv",
    index_col=0,
    parse_dates=[0],
    usecols=[1, 3],
    fix_dst=True,
    tz_column="TimeZone",
    id_column="ID",
    verbose=True,
):
    """
    Wrapper function for loading data from NREL partnership. This data is in a
    secure, private S3 bucket for use by the GISMo team only. However, the
    function can be used to load any data that is a collection of CSV files
    with a single metadata file. The metadata file contains a sequential file
    index as well as a unique system ID number for each site. Either of these
    may be set by the user to retreive data, but the ID number will take
    precedent if both are provided. The data files are assumed to be uniquely
    identified by the system ID number. In addition, the metadata file contains
    a column with time zone information for fixing daylight savings time.

    :param file_index: the sequential index number of the system
    :param id_num: the system ID number (non-sequential)
    :param location: string identifying the directory containing the data
    :param metadata_fn: the location of the metadata file
    :param data_fn_pattern: the pattern of data file identification
    :param index_col: the column containing the index (see: pandas.read_csv)
    :param parse_dates: list of columns to parse dates (see: pandas.read_csv)
    :param usecols: columns to load from file (see: pandas.read_csv)
    :param fix_dst: boolean, if true, use provided timezone information to
        correct for daylight savings time in data
    :param tz_column: the column name in the metadata file that contains the
        timezone information
    :param id_column: the column name in the metadata file that contains the
        unique system ID information
    :param verbose: boolean, print information about retreived file
    :return: pandas dataframe containing system power data
    """
    raise ("This function is no longer supported!")


def load_cassandra_data(
    siteid,
    column="ac_power",
    sensor=None,
    tmin=None,
    tmax=None,
    limit=None,
    cluster_ip=None,
    verbose=True,
):
    """
    .. deprecated:: 1.5.0
        dataio.load_cassandra_data is deprecated. Starting in Solar Data Tools 2.0, it will be removed.
        This function is deprecated. Please use load_redshift_data function instead.
    """
    raise ("This function is no longer supported!")


def load_constellation_data(
    file_id,
    location="s3://pv.insight.misc/pv_fleets/",
    data_fn_pattern="{}_20201006_composite.csv",
    index_col=0,
    parse_dates=[0],
    json_file=False,
):
    """
    Load constellation data from a specified location.

    This function reads a CSV file from a given location and optionally loads
    additional JSON metadata.

    :param file_id: Identifier for the file to load.
    :type file_id: str
    :param location: The base location where the data files are stored. Default is "s3://pv.insight.misc/pv_fleets/".
    :type location: str, optional
    :param data_fn_pattern: The pattern for the data file name. Default is "{}_20201006_composite.csv".
    :type data_fn_pattern: str, optional
    :param index_col: Column to use as the row labels of the DataFrame. Default is 0.
    :type index_col: int, optional
    :param parse_dates: List of column indices to parse as dates. Default is [0].
    :type parse_dates: list, optional
    :param json_file: Whether to load additional JSON metadata. Default is False.
    :type json_file: bool, optional

    :return: A tuple containing the DataFrame and the JSON metadata (if json_file is True), otherwise just the DataFrame.
    :rtype: tuple[pd.DataFrame, dict] or pd.DataFrame
    """
    raise ("This function is no longer supported!")


def load_redshift_data(
    siteid: str,
    api_key: str,
    column: str = "ac_power",
    sensor: int | list[int] | None = None,
    tmin: datetime | None = None,
    tmax: datetime | None = None,
    limit: int | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Queries a SunPower dataset by site id and returns a Pandas DataFrame.

    Request an API key by registering at https://pvdb.slacgismo.org and emailing slacgismotutorials@gmail.com with your information and use case.

    :param siteid: Site id to query
    :type siteid: str
    :param api_key: API key for authentication to query data
    :type api_key: str
    :param column: Meas_name to query, defaults to "ac_power"
    :type column: str, optional
    :param sensor: Sensor index to query based on the number of sensors at the site id, defaults to None
    :type sensor: int | list[int] | None, optional
    :param tmin: Minimum timestamp to query, defaults to None
    :type tmin: datetime | None, optional
    :param tmax: Maximum timestamp to query, defaults to None
    :type tmax: datetime | None, optional
    :param limit: Maximum number of rows to query, defaults to None
    :type limit: int | None, optional
    :param verbose: Option to print out additional information, defaults to False
    :type verbose: bool, optional
    :return: Pandas DataFrame containing the queried data.
    :rtype: pd.DataFrame
    """

    raise ("This function is no longer supported!")
