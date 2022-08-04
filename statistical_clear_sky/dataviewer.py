# -*- coding: utf-8 -*-
"""
This module contains the a data viewer class for data set investigation.
"""
from statistical_clear_sky.utilities.data_loading import load_results
from statistical_clear_sky.utilities.data_loading import load_sys
from statistical_clear_sky.algorithm.iterative_fitting import IterativeFitting
from statistical_clear_sky.configuration import CONFIG1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
from matplotlib.gridspec import GridSpec
import seaborn as sns
sns.set(context='paper', style='darkgrid', palette='colorblind')
import s3fs
import logging, warnings, time, os
logging.basicConfig(filename='data_viewer.log', level=logging.INFO)

TZ_LOOKUP = {
    'America/Anchorage': 9,
    'America/Chicago': 6,
    'America/Denver': 7,
    'America/Los_Angeles': 8,
    'America/New_York': 5,
    'America/Phoenix': 7,
    'Pacific/Honolulu': 10
}

class PointBrowser(object):
    """
    See "Event Handling" example from matplotlib documentation:
    https://matplotlib.org/examples/event_handling/data_browser.html

    Click on a point to select and highlight it -- the data that
    generated the point will be shown in the lower axes.  Use the 'a'
    and 's' keys to browse through the next and previous points along x-axis (ordered by RdTools estimate).
    """

    def __init__(self, data, xlim=None, ylim=None, prcntl=95):
        logging.info('NEW SESSION')
        warnings.filterwarnings("ignore")
        self.scsf_cache_dir = './local_cache/'
        if not os.path.exists(self.scsf_cache_dir):
            os.makedirs(self.scsf_cache_dir)

        ordering = np.argsort(data['rd']).values
        self.data = data.iloc[ordering]
        self.xs = self.data['rd'].values
        self.ys = self.data['deg'].values

        gs = GridSpec(4, 3)
        fig = plt.figure('DataViewer', figsize=(8, 16))
        ax = [plt.subplot(gs[0, :2])]  # Main scatter plot
        with sns.axes_style('white'):
            ax.append(plt.subplot(gs[0, -1]))  # Record viewing panel
            ax[-1].set_axis_off()
            ax.append(plt.subplot(gs[1, :]))  # Timeseries heatmap view
            ax.append(plt.subplot(gs[2, :]))  # ClearSky heatmap view
        ax.append(plt.subplot(gs[3, :]))  # Daily Energy view
        self.fig = fig
        self.ax = ax

        self.ax[0].set_title('click on point to view record')
        self.ax[0].set_xlabel('RdTools Estimate YoY deg (%)')
        self.ax[0].set_ylabel('SCSF Estimate YoY deg (%)')
        self.ax[2].set_title('Measured power')
        self.ax[2].set_xlabel('Day number')
        self.ax[2].set_yticks([])
        self.ax[2].set_ylabel('(sunset)        Time of day        (sunrise)')

        self.line, = self.ax[0].plot(self.xs, self.ys, '.', picker=5)  # 5 points tolerance
        m = np.logical_and(
            np.logical_and(
                self.data['res-median'] < np.percentile(self.data['res-median'], prcntl),
                self.data['res-var'] < np.percentile(self.data['res-var'], prcntl)
            ),
            self.data['res-L0norm'] < np.percentile(self.data['res-L0norm'], prcntl)
        )
        m = np.logical_not(m.values)
        self.ax[0].plot(self.xs[m], self.ys[m], '.')
        if xlim is None:
            xlim = self.ax[0].get_xlim()
        if ylim is None:
            ylim = self.ax[0].get_ylim()
        pts = (
            min(xlim[0], ylim[0]),
            max(xlim[1], ylim[1])
        )
        self.ax[0].plot(pts, pts, ls='--', color='red')
        self.ax[0].set_xlim(xlim)
        self.ax[0].set_ylim(ylim)
        self.text = self.ax[0].text(0.05, 0.95, 'system ID: none',
                                    transform=self.ax[0].transAxes, va='top')
        self.selected, = self.ax[0].plot([self.xs[0]], [self.ys[0]], 'o', ms=6, alpha=0.4,
                                         color='yellow', visible=False)

        with sns.axes_style('white'):
            ax.append(plt.axes([.77, .5 * (1 + .57), .2, .05 / 2]))  # Text box entry
            ax.append(plt.axes([.82, .5 * (1 + .5), .1, .05 / 2]))  # run SCSF button
        self.text_box = TextBox(self.ax[-2], 'ID Number')
        self.button = Button(self.ax[-1], 'run SCSF', color='red')
        self.lastind = None
        self._power_signals_d = None
        self._iterative_fitting = None
        self.cb = None
        self.cb2 = None
        self.local_cash = {}
        self.prcntl = prcntl
        plt.tight_layout()

        self.fig.canvas.mpl_connect('pick_event', self.onpick)
        self.fig.canvas.mpl_connect('key_press_event', self.onpress)
        self.text_box.on_submit(self.submit)
        self.button.on_clicked(self.clicked)

        plt.show()

    def submit(self, text):
        logging.info('submit: ' + str(text))
        asrt = np.argsort(np.abs(self.data.index - float(text)))
        sysid = self.data.index[asrt[0]]
        bool_list = self.data.index == sysid
        # bool_list = self.data.index == int(text)
        index_lookup = np.arange(self.data.shape[0])
        self.lastind = int(index_lookup[bool_list])
        logging.info('selected index: ' + str(self.lastind))
        self.update()

    def clicked(self, event):
        if self.lastind is None:
            logging.info('button click: nothing selected!')
            return
        sysid = self.data.iloc[self.lastind].name
        logging.info('button click: current ID: {}'.format(sysid))

        self.ax[3].cla()
        self.ax[3].text(0.05, 0.95, 'initializing algorithm...', transform=self.ax[3].transAxes,
                        va='top', fontname='monospace')
        self.ax[3].set_xlabel('Day number')
        self.ax[3].set_yticks([])
        self.ax[3].set_ylabel('(sunset)        Time of day        (sunrise)')
        plt.tight_layout()
        self.fig.canvas.draw()

        self.ax[4].cla()

        power_signals_d = self._power_signals_d
        cached_files = os.listdir(self.scsf_cache_dir)
        fn = 'pvo_' + str(sysid) + '.scsf'
        if fn in cached_files:
            iterative_fitting = IterativeFitting.load_instance(
                self.scsf_cache_dir + fn)
            self._iterative_fitting = iterative_fitting
            self.ax[4].plot(np.sum(iterative_fitting.power_signals_d, axis=0)
                 * 24 / iterative_fitting.power_signals_d.shape[0],
                 linewidth=1, label='raw data')
            use_day = iterative_fitting.weights > 1e-1
            days = np.arange(iterative_fitting.power_signals_d.shape[1])
            self.ax[4].scatter(days[use_day],
                np.sum(iterative_fitting.power_signals_d, axis=0)[use_day]
                    * 24 / iterative_fitting.power_signals_d.shape[0],
                color='orange', alpha=0.7, label='days selected')
            self.ax[4].legend()
            self.ax[4].set_title('Daily Energy')
            self.ax[4].set_xlabel('Day Number')
            self.ax[4].set_ylabel('kWh')
            self.ax[3].cla()
            self.ax[3].text(0.05, 0.95, 'loading cached results...',
                             transform=self.ax[3].transAxes,
                             va='top', fontname='monospace')
            self.ax[3].set_xlabel('Day number')
            self.ax[3].set_yticks([])
            self.ax[3].set_ylabel('(sunset)        Time of day        (sunrise)')
            self.show_ticks(self.ax[2])
            plt.tight_layout()
            self.fig.canvas.draw()
        else:
            iterative_fitting = IterativeClearSky(power_signals_d)
            self._iterative_fitting = iterative_fitting
            self.ax[4].plot(np.sum(iterative_fitting.power_signals_d, axis=0) * 24 / iterative_fitting.power_signals_d.shape[0], linewidth=1, label='raw data')
            use_day = iterative_fitting.weights > 1e-1
            days = np.arange(iterative_fitting.power_signals_d.shape[1])
            self.ax[4].scatter(days[use_day], np.sum(iterative_fitting.power_signals_d, axis=0)[use_day] * 24 / iterative_fitting.power_signals_d.shape[0],
                               color='orange', alpha=0.7, label='days selected')
            self.ax[4].legend()
            self.ax[4].set_title('Daily Energy')
            self.ax[4].set_xlabel('Day Number')
            self.ax[4].set_ylabel('kWh')
            self.ax[3].cla()
            self.ax[3].text(0.05, 0.95, 'running algorithm...', transform=self.ax[3].transAxes,
                            va='top', fontname='monospace')
            self.ax[3].set_xlabel('Day number')
            self.ax[3].set_yticks([])
            self.ax[3].set_ylabel(
                '(sunset)        Time of day        (sunrise)')
            self.show_ticks(self.ax[2])
            plt.tight_layout()
            self.fig.canvas.draw()
            logging.info('starting algorithm')
            config_l = CONFIG1.copy()
            config_l['max_iteration'] = 1
            obj_vals = iterative_fitting.calculate_objective_with_result(False)
            old_obj = np.sum(obj_vals)
            ti = time.time()
            for cntr in range(CONFIG1['max_iteration']):
                iterative_fitting.execute(**config_l)
                logging.info('min iteration {} complete'.format(cntr + 1))
                obj_vals = iterative_fitting.calculate_objective_with_result(
                    False)
                new_obj = np.sum(obj_vals)
                improvement = (old_obj - new_obj) * 1. / old_obj

                self.ax[3].cla()
                self.ax[3].set_xlabel('Day number')
                self.ax[3].set_yticks([])
                self.ax[3].set_ylabel('(sunset)        Time of day        (sunrise)')
                s1 = 'Iteration {} complete: obj = {:.2f}, f1 = {:.2f}'.format(cntr + 1, new_obj, obj_vals[0])
                s2 = 'Improvement: {:.2f}%'.format(100 * improvement)
                tf = time.time()
                s3 = 'Time elapsed: {:.2f} minutes'.format((tf - ti) / 60.)
                textout = '\n'.join([s1, s2, s3])
                logging.info(textout)
                self.ax[3].text(0.05, 0.95, textout, transform=self.ax[3].transAxes,
                                va='top', fontname='monospace')
                plt.tight_layout()
                self.fig.canvas.draw()
                old_obj = new_obj
                if improvement <= CONFIG1['eps']:
                    break
            iterative_fitting.save_instance(self.scsf_cache_dir + fn)

        logging.info('algorithm complete')
        self.ax[4].plot((iterative_fitting.r_cs_value[0] *
            np.sum(iterative_fitting.l_cs_value[:, 0])) * 24 /
            iterative_fitting.power_signals_d.shape[0],
            linewidth=1, label='clear sky estimate')
        self.ax[4].legend()
        logging.info('first plot complete')
        with sns.axes_style('white'):
            self.ax[3].cla()
            bar = self.ax[3].imshow(
                iterative_fitting.clear_sky_signals(), cmap='hot',
                vmin=0, vmax=np.max(iterative_fitting.power_signals_d),
                interpolation='none', aspect='auto')
            if self.cb2 is not None:
                self.cb2.remove()
            self.cb2 = plt.colorbar(bar, ax=self.ax[3], label='kW')
        self.show_ticks(self.ax[3])
        self.ax[3].set_title('Estimated clear sky power')
        self.ax[3].set_xlabel('Day number')
        self.ax[3].set_yticks([])
        self.ax[3].set_ylabel('(sunset)        Time of day        (sunrise)')
        logging.info('second plot complete')
        plt.tight_layout()
        self.fig.canvas.draw()
        return

    def show_ticks(self, ax):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        use_day = self._iterative_fitting.weights > 1e-1
        days = np.arange(self._iterative_fitting.power_signals_d.shape[1])
        y1 = np.ones_like(days[use_day]) * self._power_signals_d.shape[0] * .99
        ax.scatter(days[use_day], y1, marker='|', color='yellow', s=2)
        ax.scatter(days[use_day], .995 * y1, marker='|', color='yellow', s=2)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        return

    def onpress(self, event):
        if self.lastind is None:
            return

        logging.info('press event: ' + str(event.key))

        if event.key == 'a':
            inc = -1
            self.lastind += inc
            self.lastind = np.clip(self.lastind, 0, len(self.xs) - 1)
        elif event.key == 's':
            inc = 1
            self.lastind += inc
            self.lastind = np.clip(self.lastind, 0, len(self.xs) - 1)
        else:
            return

        self.update()

    def onpick(self, event):
        if event.artist != self.line:
            return True

        N = len(event.ind)
        if not N:
            return True

        # the click locations
        x = event.mouseevent.xdata
        y = event.mouseevent.ydata

        logging.info('pick: ' + str(x) + ', ' + str(y))

        distances = np.hypot(x - self.xs[event.ind], y - self.ys[event.ind])
        indmin = distances.argmin()
        dataind = event.ind[indmin]

        self.lastind = dataind
        self.update()

    def update(self):
        if self.lastind is None:
            return

        dataind = self.lastind
        prcntl = self.prcntl

        logging.info('updating, ID = {}'.format(self.data.iloc[dataind].name))

        self.selected.set_visible(True)
        self.selected.set_data(self.xs[dataind], self.ys[dataind])

        out1 = 'system ID: {:d}'.format(self.data.iloc[dataind].name)
        out2 = str(self.data.iloc[dataind])
        # self.text_box.set_val('')

        idxs = np.arange(len(self.data.columns))
        if self.data.iloc[dataind]['res-median'] > np.percentile(self.data['res-median'], prcntl):
            l1 = out2.split('\n')
            i = idxs[self.data.columns == 'res-median'][0]
            l1[i] = '*' + l1[i][:-2] + '*'
            out2 = '\n'.join(l1)
        if self.data.iloc[dataind]['res-var'] > np.percentile(self.data['res-var'], prcntl):
            l1 = out2.split('\n')
            i = idxs[self.data.columns == 'res-var'][0]
            l1[i] = '*' + l1[i][:-2] + '*'
            out2 = '\n'.join(l1)
        if self.data.iloc[dataind]['res-L0norm'] > np.percentile(self.data['res-L0norm'], prcntl):
            l1 = out2.split('\n')
            i = idxs[self.data.columns == 'res-L0norm'][0]
            l1[i] = '*' + l1[i][:-2] + '*'
            out2 = '\n'.join(l1)

        self.text.set_text(out1)
        self.ax[1].cla()
        self.ax[1].text(0.00, 0.95, out2, transform=self.ax[1].transAxes, va='top', fontname='monospace')
        self.ax[1].set_axis_off()
        self.ax[2].cla()
        self.ax[2].text(0.05, 0.95, 'data loading...', transform=self.ax[2].transAxes, va='top', fontname='monospace')
        self.ax[2].set_xlabel('Day number')
        self.ax[2].set_yticks([])
        self.ax[2].set_ylabel('(sunset)        Time of day        (sunrise)')
        self.ax[3].cla()
        self.ax[4].cla()
        self._iterative_fitting = None
        plt.tight_layout()
        self.fig.canvas.draw()

        with sns.axes_style('white'):
            idnum = self.data.iloc[dataind].name
            if idnum in self.local_cash.keys():
                df = self.local_cash[idnum]
            else:
                df = load_sys(idnum=idnum, local=False)
                self.local_cash[idnum] = df
            days = df.resample('D').max().index[1:-1]
            start = days[0]
            end = days[-1]
            power_signals_d = df.loc[start:end].iloc[:-1].values.reshape(
                288, -1, order='F')
            self._power_signals_d = power_signals_d
            self.ax[2].cla()
            foo = self.ax[2].imshow(power_signals_d, cmap='hot', interpolation='none', aspect='auto')
            if self.cb is not None:
                self.cb.remove()
            self.cb = plt.colorbar(foo, ax=self.ax[2], label='kW')

        self.ax[2].set_xlabel('Day number')
        self.ax[2].set_yticks([])
        self.ax[2].set_ylabel('(sunset)        Time of day        (sunrise)')
        self.ax[2].set_title('Measured power')

        self.text_box.set_val('')
        self.fig.canvas.draw()

    @property
    def iterative_fitting(self):
        return self._iterative_fitting

def view_ts(pb, clear_day_start=None, day_start=None):
    if pb.iterative_fitting is not None:
        clear_days = np.arange(
            len(pb.iterative_fitting.weights))[
                pb.iterative_fitting.weights >= 1e-3]
        fig = pb.iterative_fitting.ts_plot_with_weights(
            num_days=len(pb.iterative_fitting.weights), figsize=(9, 6),
            fig_title='System ID: {}'.format(pb.data.iloc[pb.lastind].name))
        if clear_day_start is not None:
            N = clear_day_start
            plt.xlim(clear_days[N] - 2, clear_days[N] - 2 + 5)
        elif day_start is not None:
            plt.xlim(day_start, day_start+5)
        else:
            plt.xlim(0, 5)
        plt.show()
