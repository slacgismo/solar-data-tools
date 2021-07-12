# -*- coding: utf-8 -*-
''' Plotting Module



'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_2d(D, figsize=(12, 6), units='kW', clear_days=None, dates=None,
            year_lines=False, ax=None, color='red'):
    """
    A function for plotting the power heat map for solar power data
    
    :param D: PV power data arranged as a matrix, typically the output of
        `data_transforms.make_2d()`
    :param figsize: the size of the desired figure (passed to `matplotlib`)
    :param units: the units of the power data
    :param clear_days: a boolean array marking the location of clear days in
        the data set, typically the output of `clear_day_detection.find_clear_days()`
    :return: `matplotlib` figure
    """
    if D is not None:
        with sns.axes_style("white"):
            if ax is None:
                fig, ax = plt.subplots(nrows=1, figsize=figsize)
            else:
                fig = ax.get_figure()
            foo = ax.imshow(D, cmap='plasma', interpolation='none', aspect='auto')
            ax.set_title('Measured power')
            plt.colorbar(foo, ax=ax, label=units)
            ax.set_xlabel('Day number')
            ax.set_yticks([])
            ax.set_ylabel('(sunset)        Time of day        (sunrise)')
            if clear_days is not None:
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                use_day = clear_days
                days = np.arange(D.shape[1])
                y1 = np.ones_like(days[use_day]) * D.shape[0] * .999
                ax.scatter(days[use_day], y1, marker='|', color=color, s=2)
                ax.scatter(days[use_day], .995*y1, marker='|', color=color, s=2)
                ax.scatter(days[use_day], .99*y1, marker='|', color=color, s=2)
                ax.scatter(days[use_day], .985*y1, marker='|', color=color, s=2)
                ax.scatter(days[use_day], .98*y1, marker='|', color=color, s=2)
                ax.scatter(days[use_day], .975*y1, marker='|', color=color, s=2)
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
        if dates is not None:
            if D.shape[1] >= 356 * 1.5:
                mask = np.logical_and(dates.month == 1, dates.day == 1)
                day_ticks = np.arange(D.shape[1])[mask]
                plt.xticks(day_ticks, dates[day_ticks].year)
                plt.xlabel('Year')
            else:
                mask = dates.day == 1
                day_ticks = np.arange(D.shape[1])[mask]
                plt.xticks(day_ticks, dates[day_ticks].month)
                plt.xlabel('Month')
            if year_lines:
                for d in day_ticks:
                    plt.axvline(d, ls='--', color='gray', linewidth=1)
        return fig
    else:
        return
