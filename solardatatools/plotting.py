# -*- coding: utf-8 -*-
''' Plotting Module



'''

import matplotlib.pyplot as plt
import seaborn as sns

def plot_2d(D, figsize=(12, 6)):
    if D is not None:
        with sns.axes_style("white"):
            fig, ax = plt.subplots(nrows=1, figsize=figsize)
            foo = ax.imshow(D, cmap='hot', interpolation='none', aspect='auto', vmin=0)
            ax.set_title('Measured power')
            plt.colorbar(foo, ax=ax, label='kW')
            ax.set_xlabel('Day number')
            ax.set_yticks([])
            ax.set_ylabel('(sunset)        Time of day        (sunrise)')
        return fig
    else:
        return
