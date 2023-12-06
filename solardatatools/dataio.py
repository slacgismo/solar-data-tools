# -*- coding: utf-8 -*-
""" Data IO Module

This module contains functions for obtaining data from various sources.

"""
import math
from solardatatools.time_axis_manipulation import (
    standardize_time_axis,
    fix_daylight_savings_with_known_tz,
)
from solardatatools.utilities import progress

from time import time, perf_counter
from io import StringIO, BytesIO
import base64
import os
import json
import requests
import numpy as np
import pandas as pd
from typing import Callable, TypedDict, Any, Tuple, Dict
from functools import wraps
from datetime import datetime
import gzip


class SSHParams(TypedDict):
    ssh_address_or_host: tuple[str, int]
    ssh_username: str
    ssh_private_key: str
    remote_bind_address: tuple[str, int]


class DBConnectionParams(TypedDict):
    database: str
    user: str
    password: str
    host: str
    port: int


def get_pvdaq_data(sysid=2, api_key="DEMO_KEY", year=2011, delim=",", standardize=True):
    """
    This fuction queries one or more years of raw PV system data from NREL's PVDAQ data service:
            https://maps.nrel.gov/pvdaq/
    """
    # Force year to be a list of integers
    ti = time()
    try:
        year = int(year)
    except TypeError:
        year = [int(yr) for yr in year]
    else:
        year = [year]
    # Each year must queries separately, so iterate over the years and generate a list of dataframes.
    df_list = []
    it = 0
    for yr in year:
        progress(it, len(year), "querying year {}".format(year[it]))
        req_params = {"api_key": api_key, "system_id": sysid, "year": yr}
        base_url = "https://developer.nrel.gov/api/pvdaq/v3/data_file?"
        param_list = [str(item[0]) + "=" + str(item[1]) for item in req_params.items()]
        req_url = base_url + "&".join(param_list)
        response = requests.get(req_url)
        if int(response.status_code) != 200:
            print("\n error: ", response.status_code)
            return
        df = pd.read_csv(StringIO(response.text), delimiter=delim)
        df_list.append(df)
        it += 1
    tf = time()
    progress(it, len(year), "queries complete in {:.1f} seconds       ".format(tf - ti))
    # concatenate the list of yearly data frames
    df = pd.concat(df_list, axis=0, sort=True)
    if standardize:
        print("\n")
        df, _ = standardize_time_axis(df, datetimekey="Date-Time", timeindex=False)
    return df


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
    meta = pd.read_csv(location + metadata_fn)
    if id_num is None:
        id_num = meta[id_column][file_index]
    else:
        file_index = meta[meta[id_column] == id_num].index[0]
    df = pd.read_csv(
        location + data_fn_pattern.format(id_num),
        index_col=index_col,
        parse_dates=parse_dates,
        usecols=usecols,
    )
    if fix_dst:
        tz = meta[tz_column][file_index]
        fix_daylight_savings_with_known_tz(df, tz=tz, inplace=True)
    if verbose:
        print("index: {}; system ID: {}".format(file_index, id_num))
    return df


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
    try:
        from cassandra.cluster import Cluster
    except ImportError:
        print(
            "Please install cassandra-driver in your Python environment to use this function"
        )
        return
    ti = time()
    if cluster_ip is None:
        home = os.path.expanduser("~")
        cluster_location_file = home + "/.aws/cassandra_cluster"
        try:
            with open(cluster_location_file) as f:
                cluster_ip = f.readline().strip("\n")
        except FileNotFoundError:
            msg = "Please put text file containing cluster IP address in "
            msg += "~/.aws/cassander_cluster or provide your own IP address"
            print(msg)
            return
    cluster = Cluster([cluster_ip])
    session = cluster.connect("measurements")
    cql = """
        select site, meas_name, ts, sensor, meas_val_f
        from measurement_raw
        where site = '{}'
            and meas_name = '{}'
    """.format(
        siteid, column
    )
    ts_constraint = np.logical_or(tmin is not None, tmax is not None)
    if tmin is not None:
        cql += "and ts > '{}'\n".format(tmin)
    if tmax is not None:
        cql += "and ts < '{}'\n".format(tmax)
    if sensor is not None and ts_constraint:
        cql += "and sensor = '{}'\n".format(sensor)
    elif sensor is not None and not ts_constraint:
        cql += "and ts > '2000-01-01'\n"
        cql += "and sensor = '{}'\n".format(sensor)
    if limit is not None:
        cql += "limit {}".format(np.int(limit))
    cql += ";"
    rows = session.execute(cql)
    df = pd.DataFrame(list(rows))
    df.replace(-999999.0, np.NaN, inplace=True)
    tf = time()
    if verbose:
        print("Query of {} rows complete in {:.2f} seconds".format(len(df), tf - ti))
    return df


def load_constellation_data(
    file_id,
    location="s3://pv.insight.misc/pv_fleets/",
    data_fn_pattern="{}_20201006_composite.csv",
    index_col=0,
    parse_dates=[0],
    json_file=False,
):
    df = pd.read_csv(
        location + data_fn_pattern.format(file_id),
        index_col=index_col,
        parse_dates=parse_dates,
    )

    if json_file:
        try:
            from smart_open import smart_open
        except ImportError:
            print(
                "Please install smart_open in your Python environment to use this function"
            )
            return df, None

        for line in smart_open(location + str(file_id) + "_system_details.json", "rb"):
            file_json = json.loads(line)
            file_json
        return df, file_json
    return df


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
    """Loads data based on a site id from a Redshift database into a Pandas DataFrame using an SSH tunnel

    Parameters
    ----------
        siteid : str
            site id to query
        api_key : str
            api key for authentication to redshift
        column : str
            meas_name to query (default ac_power)
        sensor : int, list[int], optional
            sensor index to query based on number of sensors at the site id (default None)
        tmin : timestamp, optional
            minimum timestamp to query (default None)
        tmax : timestamp, optional
            maximum timestamp to query (default None)
        limit : int, optional
            maximum number of rows to query (default None)
        verbose : bool, optional
            whether to print out timing information (default False)

    Returns
    ------
    df : pd.DataFrame
        Pandas DataFrame containing the queried data
    """

    def decompress_data_to_dataframe(encoded_data):
        # Decompress gzip data
        decompressed_buffer = BytesIO(encoded_data)
        with gzip.GzipFile(fileobj=decompressed_buffer, mode="rb") as gz:
            decompressed_data = gz.read().decode("utf-8")

        # Attempt to read the decompressed data as CSV
        df = pd.read_csv(StringIO(decompressed_data))

        return df

    def timing(verbose: bool = True):
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = perf_counter()
                result = func(*args, **kwargs)
                end_time = perf_counter()
                execution_time = end_time - start_time
                if verbose:
                    print(f"{func.__name__} took {execution_time:.3f} seconds to run")
                return result

            return wrapper

        return decorator

    @timing(verbose)
    def query_redshift_w_api(page: int, is_batch: bool = False) -> requests.Response:
        url = "https://lmojfukey3rylrbqughzlfu6ca0ujdby.lambda-url.us-west-1.on.aws/"
        payload = {
            "api_key": api_key,
            "siteid": siteid,
            "column": column,
            "sensor": sensor,
            "tmin": str(tmin),
            "tmax": str(tmax),
            "limit": str(limit),
            "page": str(page),
            "batch_num": str(is_batch),
        }

        if sensor is None:
            payload.pop("sensor")
        if tmin is None:
            payload.pop("tmin")
        if tmax is None:
            payload.pop("tmax")
        if limit is None:
            payload.pop("limit")

        response = requests.post(url, json=payload, timeout=60 * 5)
        if response.status_code != 200:
            raise Exception(f"Error {response.status_code} returned from API")

        if verbose:
            print(f"Content size: {len(response.content)}")

        return response

    def fetch_data(df_list: list[pd.DataFrame], index: int, page: int):
        try:
            response = query_redshift_w_api(page)
            new_df = decompress_data_to_dataframe(response.content)

            if new_df.empty:
                raise Exception("Empty dataframe returned from query")
            if verbose:
                print(f"Page: {page}, Rows: {len(new_df)}")
            df_list[index] = new_df

        except Exception as e:
            print(e)
            # raise e

    import threading

    batch_df: requests.Response = query_redshift_w_api(0, is_batch=True)
    data = batch_df.json()
    max_limit = int(data["max_limit"])
    total_count = int(data["total_count"])
    batches = int(data["batches"])
    if verbose:
        print("total number rows for query: ", total_count)
        print("Max number of rows per API call", max_limit)
        print("Total number of batches", batches)

    batch_size = 2  # Max number of threads to run at once (limited by redshift)

    loops = math.ceil(batches / batch_size)
    if batches <= batch_size:
        loops = 1
        batch_size = batches
    page = 0
    df = pd.DataFrame()
    list_of_dfs: list[pd.DataFrame] = []
    for _ in range(loops):
        df_list = [pd.DataFrame() for _ in range(batch_size)]
        page_batch = list(range(page, page + batch_size))
        threads: list[threading.Thread] = []

        # Create threads for each batch of pages
        for i in range(len(page_batch)):
            thread = threading.Thread(
                target=fetch_data, args=(df_list, i, page_batch[i])
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Move to the next batch of pages
        page += batch_size

        # Concatenate the dataframes
        valid_df_list = [new_df for new_df in df_list if not new_df.empty]

        list_of_dfs.extend(valid_df_list)

    df = pd.concat(list_of_dfs, ignore_index=True)
    # If any batch returns an empty DataFrame, stop querying
    if df.empty:
        raise Exception("Empty dataframe returned from query")
    return df
