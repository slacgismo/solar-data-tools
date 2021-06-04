''' Polar Transform Module

This module contains a class for tranforming a power signal into azimuth-
elevation space, which requires system latitude and longitude to calculate.

'''

import numpy as np
import pandas as pd
from pvlib.solarposition import get_solarposition
import seaborn as sns
import matplotlib.pyplot as plt


class PolarTransform():
    def __init__(self, series, latitude, longitude, tz_offset=-8,
                 boolean_selection=None, normalize_data=False):
        self.data = series
        self.lat = latitude
        self.lon = longitude
        if normalize_data:
            self.normed_data = self.normalize_days()
        else:
            self.normed_data = self.data.copy()
            self.normed_data.name = 'normalized-power'
        self.tz_offset = tz_offset
        if boolean_selection is None:
            self.bix = np.ones(len(series), dtype=bool)
        else:
            if len(boolean_selection) == len(series):
                self.bix = boolean_selection
            else:
                e = 'boolean index for data selection must'
                e += ' match length of data series'
                raise Exception(e)
        self.tranformed_data = None

    def normalize_days(self):
        series = self.data
        quantiles = series.groupby(series.index.date).aggregate(
            lambda x: np.nanquantile(x, .95)
        )
        normed_data = pd.Series(index=series.index, dtype=float,
                                name='normalized-power')
        for ix in quantiles.index:
            day = ix.strftime('%x')
            if quantiles.loc[ix] >= 0:
                power_normed = series.loc[day] / \
                               quantiles.loc[ix]
            else:
                power_normed = np.zeros_like(
                    series.loc[day])
            normed_data.loc[day] = power_normed
        return normed_data

    def transform(self, elevation_round=1, azimuth_round=2,
                  agg_func=np.nanmean):
        # Solar position subroutine requires TZ-aware time stamps or UTC time
        # stamps, but we typically have non-TZ-aware local time, without DST
        # shifts (which is the most natural for solar data). So, step 1 is to
        # convert the time to UTC.
        times = self.data.index.shift(-self.tz_offset, 'H')
        # run solar position subroutine
        solpos = get_solarposition(times, self.lat, self.lon)
        # reset the time from UTC to local
        solpos.index = self.data.index
        # join the calculated angles with the power measurments
        triples = solpos[['apparent_elevation', 'azimuth']].join(
            self.normed_data.loc[self.bix])
        triples = triples.dropna()
        # cut off all the entries corresponding to the sun below the horizon
        triples = triples[triples['apparent_elevation'] >= 0]
        # a function for rounding to the nearest integer c (e.g. 2, 5, 10...)
        my_round = lambda x, c: c * np.round(x / c, 0)
        # create elevation and azimuth bins
        triples['elevation_angle'] = my_round(triples['apparent_elevation'],
                                          elevation_round)
        triples['azimuth_angle'] = my_round(triples['azimuth'],
                                        azimuth_round)
        # group by bins
        grouped = triples.groupby(['azimuth_angle', 'elevation_angle'])
        # calculation aggregation function for each bin
        grouped = grouped.agg(agg_func)
        # remove other columns
        grouped = grouped['normalized-power']
        # convert tall data format to wide data format, fill misisng values
        # with zero, and transpose
        estimates = grouped.unstack().fillna(0).T
        # reverse the first axis
        estimates = estimates.iloc[::-1]
        self.tranformed_data = estimates

    def plot_transformation(self, figsize=(10,6)):
        plt.figure(figsize=figsize)
        sns.heatmap(self.tranformed_data, ax=plt.gca(), cmap='plasma')
        return plt.gcf()