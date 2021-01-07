# -*- coding: utf-8 -*-
''' Data IO Module

This module contains functions for obtaining data from various sources.

'''
from solardatatools.time_axis_manipulation import standardize_time_axis, \
    fix_daylight_savings_with_known_tz
from solardatatools.utilities import progress

from time import time
from io import StringIO
import os
import json
import requests
import numpy as np
import pandas as pd


def get_pvdaq_data(sysid=2, api_key='DEMO_KEY', year=2011, delim=',',
                   standardize=True):
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
        progress(it, len(year), 'querying year {}'.format(year[it]))
        req_params = {
            'api_key': api_key,
            'system_id': sysid,
            'year': yr
        }
        base_url = 'https://developer.nrel.gov/api/pvdaq/v3/data_file?'
        param_list = [str(item[0]) + '=' + str(item[1]) for item in req_params.items()]
        req_url = base_url + '&'.join(param_list)
        response = requests.get(req_url)
        if int(response.status_code) != 200:
            print('\n error: ', response.status_code)
            return
        df = pd.read_csv(StringIO(response.text), delimiter=delim)
        df_list.append(df)
        it += 1
    tf = time()
    progress(it, len(year), 'queries complete in {:.1f} seconds       '.format(tf - ti))
    # concatenate the list of yearly data frames
    df = pd.concat(df_list, axis=0, sort=True)
    if standardize:
        df = standardize_time_axis(df, datetimekey='Date-Time', timeindex=False)
    return df


def load_pvo_data(file_index=None, id_num=None, location='s3://pv.insight.nrel/PVO/',
                  metadata_fn='sys_meta.csv', data_fn_pattern='PVOutput/{}.csv',
                  index_col=0, parse_dates=[0], usecols=[1, 3], fix_dst=True,
                  tz_column='TimeZone', id_column='ID',
                  verbose=True):
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
    df = pd.read_csv(location + data_fn_pattern.format(id_num), index_col=index_col,
                     parse_dates=parse_dates, usecols=usecols)
    if fix_dst:
        tz = meta[tz_column][file_index]
        fix_daylight_savings_with_known_tz(df, tz=tz, inplace=True)
    if verbose:
        print('index: {}; system ID: {}'.format(file_index, id_num))
    return df


def load_cassandra_data(siteid, column='ac_power', sensor=None, tmin=None,
                        tmax=None, limit=None, cluster_ip=None, verbose=True):
    try:
        from cassandra.cluster import Cluster
    except ImportError:
        print('Please install cassandra-driver in your Python environment to use this function')
        return
    ti = time()
    if cluster_ip is None:
        home = os.path.expanduser("~")
        cluster_location_file = home + '/.aws/cassandra_cluster'
        try:
            with open(cluster_location_file) as f:
                cluster_ip = f.readline().strip('\n')
        except FileNotFoundError:
            msg = 'Please put text file containing cluster IP address in '
            msg += '~/.aws/cassander_cluster or provide your own IP address'
            print(msg)
            return
    cluster = Cluster([cluster_ip])
    session = cluster.connect('measurements')
    cql = """
        select site, meas_name, ts, sensor, meas_val_f 
        from measurement_raw
        where site = '{}'
            and meas_name = '{}'
    """.format(siteid, column)
    ts_constraint = np.logical_or(
        tmin is not None,
        tmax is not None
    )
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
    cql += ';'
    rows = session.execute(cql)
    df = pd.DataFrame(list(rows), )
    df.replace(-999999.0, np.NaN, inplace=True)
    tf = time()
    if verbose:
        print('Query of {} rows complete in {:.2f} seconds'.format(
            len(df), tf - ti)
        )
    return df


def load_constellation_data(file_id, location='s3://pv.insight.misc/pv_fleets/',
                            data_fn_pattern='{}_20201006_composite.csv',
                            index_col=0, parse_dates=[0], json_file=False):
    df = pd.read_csv(location + data_fn_pattern.format(file_id), index_col=index_col, parse_dates=parse_dates)

    if json_file:
        try:
            from smart_open import smart_open
        except ImportError:
            print('Please install smart_open in your Python environment to use this function')
            return df, None

        for line in smart_open(location + str(file_id) + '_system_details.json', 'rb'):
            file_json = json.loads(line)
            file_json
        return df, file_json
    return df
