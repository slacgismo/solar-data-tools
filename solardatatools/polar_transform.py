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
        times = self.data.index.shift(-self.tz_offset, 'H')
        solpos = get_solarposition(times, self.lat, self.lon)
        solpos.index = self.data.index
        triples = solpos[['apparent_elevation', 'azimuth']].join(
            self.normed_data.loc[self.bix])
        triples = triples.dropna()
        triples = triples[triples['apparent_elevation'] >= 0]
        my_round = lambda x, c: c * np.round(x / c, 0)
        triples['elevation_angle'] = my_round(triples['apparent_elevation'],
                                          elevation_round)
        triples['azimuth_angle'] = my_round(triples['azimuth'],
                                        azimuth_round)
        grouped = triples.groupby(['azimuth_angle', 'elevation_angle'])
        grouped = grouped.agg(agg_func)
        grouped = grouped['normalized-power']
        estimates = grouped.unstack().fillna(0).T
        estimates = estimates.iloc[::-1]
        self.tranformed_data = estimates

    def plot_transformation(self, figsize=(10,6)):
        plt.figure(figsize=figsize)
        sns.heatmap(self.tranformed_data, ax=plt.gca(), cmap='plasma')
        return plt.gcf()