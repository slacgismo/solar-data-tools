# -*- coding: utf-8 -*-
''' Data IO Module

This module contains functions for obtaining data from various sources.

'''
from solardatatools.data_transforms import standardize_time_axis
from solardatatools.utilities import progress

from time import time
from io import StringIO

import requests
import pandas as pd


def get_pvdaq_data(sysid=2, api_key = 'DEMO_KEY', year=2011, delim=',',
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
        df = standardize_time_axis(df, datetimekey='Date-Time')
    return df