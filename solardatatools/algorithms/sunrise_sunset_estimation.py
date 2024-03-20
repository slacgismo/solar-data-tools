""" Sunrise Sunset Estimation Algorithm Module

This module contains an algorithm for robust estimation of sunrise and sunset
times from measured power data. This algorithm utilizes the following prior
information:

 - That sunrise and sunset times vary slowly from day to day (smooth signals)
 - That sunrise and sunset times repeat on a yearly basis (periodic signals)

This algorithm attempt to estimate the true sunrise and sunset times, in
accordance with standard geometric models of sun position. It uses holdout
validation to tune the theshold parameter, as opposed to the subroutines in
the DataHandler pipeline which do not tune this parameter. This algorithm
should be used in situations where exact estimates of true sunrise and sunset
times are required, such as for latitude and longitude estimation from data.

Bennet Meyers, 7/2/20

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from solardatatools.daytime import detect_sun
from solardatatools.sunrise_sunset import rise_set_rough, rise_set_smoothed
from solardatatools.signal_decompositions import tl1_l2d2p365


class SunriseSunset:
    def __init__(self):
        self.sunrise_estimates = None
        self.sunset_estimates = None
        self.sunrise_measurements = None
        self.sunset_measurements = None
        self.sunup_mask_measured = None
        self.sunup_mask_estimated = None
        self.threshold = None
        self.total_rmse = None
        self.true_times = None
        self.sunrise_tau = 0.1
        self.sunset_tau = 0.9

    def calculate_times(
        self,
        data,
        threshold=None,
        plot=False,
        figsize=(12, 10),
        zoom_fit=False,
        solver="OSQP",
    ):
        # print('Calculating times')
        if threshold is None:
            if self.threshold is not None:
                threshold = self.threshold
            else:
                print("Please run optimizer or provide a threshold")
                return
        if self.true_times is not None:
            sr_true = self.true_times["sunrise times"].values
            ss_true = self.true_times["sunset times"].values
        else:
            sr_true = None
            ss_true = None
        bool_msk = detect_sun(data, threshold)
        measured = rise_set_rough(bool_msk)
        smoothed = rise_set_smoothed(
            measured, sunrise_tau=self.sunrise_tau, sunset_tau=self.sunset_tau, solver=solver
        )
        self.sunrise_estimates = smoothed["sunrises"]
        self.sunset_estimates = smoothed["sunsets"]
        self.sunrise_measurements = measured["sunrises"]
        self.sunset_measurements = measured["sunsets"]
        self.sunup_mask_measured = bool_msk
        data_sampling = int(24 * 60 / data.shape[0])
        num_days = data.shape[1]
        mat = np.tile(np.arange(0, 24, data_sampling / 60), (num_days, 1)).T
        sr_broadcast = np.tile(self.sunrise_estimates, (data.shape[0], 1))
        ss_broadcast = np.tile(self.sunset_estimates, (data.shape[0], 1))
        self.sunup_mask_estimated = np.logical_and(
            mat >= sr_broadcast, mat < ss_broadcast
        )
        self.threshold = threshold

        if plot:
            fig, ax = plt.subplots(nrows=4, figsize=figsize, sharex=True)
            ylims = []
            ax[0].set_title("Sunrise Times")
            ax[0].plot(self.sunrise_estimates, ls="--", color="blue")
            ylims.append(ax[0].get_ylim())
            ax[0].plot(
                self.sunrise_measurements,
                label="measured",
                marker=".",
                ls="none",
                alpha=0.3,
                color="green",
            )
            ax[0].plot(self.sunrise_estimates, label="estimated", ls="--", color="blue")
            if self.true_times is not None:
                ax[0].plot(sr_true, label="true", color="orange")
            ax[1].set_title("Sunset Times")
            ax[1].plot(self.sunset_estimates, ls="--", color="blue")
            ylims.append(ax[1].get_ylim())
            ax[1].plot(
                self.sunset_measurements,
                label="measured",
                marker=".",
                ls="none",
                alpha=0.3,
                color="green",
            )
            ax[1].plot(self.sunset_estimates, label="estimated", ls="--", color="blue")
            if self.true_times is not None:
                ax[1].plot(ss_true, label="true", color="orange")
            ax[2].set_title("Solar Noon")
            ax[2].plot(
                np.average([self.sunrise_estimates, self.sunset_estimates], axis=0),
                ls="--",
                color="blue",
            )
            ylims.append(ax[2].get_ylim())
            ax[2].plot(
                np.average(
                    [self.sunrise_measurements, self.sunset_measurements], axis=0
                ),
                label="measured",
                marker=".",
                ls="none",
                alpha=0.3,
                color="green",
            )
            ax[2].plot(
                np.average([self.sunrise_estimates, self.sunset_estimates], axis=0),
                label="estimated",
                ls="--",
                color="blue",
            )
            if self.true_times is not None:
                ax[2].plot(
                    np.average([sr_true, ss_true], axis=0), label="true", color="orange"
                )
            ax[3].set_title("Daylight Hours")
            ax[3].plot(
                self.sunset_estimates - self.sunrise_estimates, ls="--", color="blue"
            )
            ylims.append(ax[3].get_ylim())
            ax[3].plot(
                self.sunset_measurements - self.sunrise_measurements,
                label="measured",
                marker=".",
                ls="none",
                alpha=0.3,
                color="green",
            )
            ax[3].plot(
                self.sunset_estimates - self.sunrise_estimates,
                label="estimated",
                ls="--",
                color="blue",
            )
            if self.true_times is not None:
                ax[3].plot(ss_true - sr_true, label="true", color="orange")
            for i in range(4):
                ax[i].legend(loc=1)
            if zoom_fit:
                for ax_it, ylim_it in zip(ax, ylims):
                    ax_it.set_ylim(0.97 * ylim_it[0], 1.03 * ylim_it[1])
            # plt.tight_layout()
            return fig
        else:
            return

    def calculate_true(self, dh, lat, lon, tz_offset):
        outtab = sunrise_sunset_times(
            lat, lon, dh.day_index.dayofyear.values, tz_offset
        )
        outtab.index = dh.day_index
        self.true_times = outtab

    def run_optimizer(
        self,
        data,
        random_seed=None,
        search_pts=21,
        plot=False,
        figsize=(8, 6),
        solver="OSQP",
    ):
        if self.true_times is not None:
            sr_true = self.true_times["sunrise times"].values
            ss_true = self.true_times["sunset times"].values
        else:
            sr_true = None
            ss_true = None
        ths = np.logspace(-5, -1, search_pts)
        ho_error = []
        full_error = []
        for th in ths:
            bool_msk = detect_sun(data, th)
            measured = rise_set_rough(bool_msk)
            sunrises = measured["sunrises"]
            sunsets = measured["sunsets"]
            np.random.seed(random_seed)
            use_set_sr = np.arange(len(sunrises))[~np.isnan(sunrises)]
            use_set_ss = np.arange(len(sunsets))[~np.isnan(sunsets)]
            if (
                len(use_set_sr) / len(sunrises) > 0.6
                and len(use_set_ss) / len(sunsets) > 0.6
            ):
                run_ho_errors = []
                num_trials = 1  # if > 1, average over multiple random selections
                for run in range(num_trials):
                    np.random.shuffle(use_set_sr)
                    np.random.shuffle(use_set_ss)
                    # 80-20 train test split
                    split_at_sr = int(len(use_set_sr) * 0.8)
                    split_at_ss = int(len(use_set_ss) * 0.8)
                    train_sr = use_set_sr[:split_at_sr]
                    train_ss = use_set_ss[:split_at_ss]
                    test_sr = use_set_sr[split_at_sr:]
                    test_ss = use_set_ss[split_at_ss:]
                    train_msk_sr = np.zeros_like(sunrises, dtype=bool)
                    train_msk_ss = np.zeros_like(sunsets, dtype=bool)
                    train_msk_sr[train_sr] = True
                    train_msk_ss[train_ss] = True
                    test_msk_sr = np.zeros_like(sunrises, dtype=bool)
                    test_msk_ss = np.zeros_like(sunsets, dtype=bool)
                    test_msk_sr[test_sr] = True
                    test_msk_ss[test_ss] = True
                    sr_smoothed = tl1_l2d2p365(
                        sunrises, train_msk_sr, tau=self.sunrise_tau, solver=solver
                    )
                    ss_smoothed = tl1_l2d2p365(
                        sunsets, train_msk_ss, tau=self.sunset_tau, solver=solver
                    )
                    r1 = (sunrises - sr_smoothed)[test_msk_sr]
                    r2 = (sunsets - ss_smoothed)[test_msk_ss]
                    ho_resid = np.r_[r1, r2]
                    # TESTING
                    # print(th)
                    # plt.plot(ho_resid)
                    # plt.show()
                    #####

                    # 7/30/20:
                    # Some sites can have "consistent" fit (low holdout error)
                    # that is not the correct estimate. We impose the restriction
                    # that the range of sunrise times and sunset times must be
                    # greater than 15 minutes. Any solution that is less than
                    # that must be non-physical. (See: PVO ID# 30121)
                    cond1 = np.max(sr_smoothed) - np.min(sr_smoothed) > 0.25
                    cond2 = np.max(ss_smoothed) - np.min(ss_smoothed) > 0.25
                    if cond1 and cond2:
                        # L1-loss instead of L2
                        # L1-loss is better proxy for goodness of fit when using
                        # quantile loss function
                        ###
                        run_ho_errors.append(np.mean(np.abs(ho_resid)))
                    else:
                        run_ho_errors.append(1e2)
                ho_error.append(np.average(run_ho_errors))
                if self.true_times is not None:
                    full_fit = rise_set_smoothed(
                        measured, sunrise_tau=self.sunrise_tau, sunset_tau=self.sunset_tau, solver=solver
                    )
                    sr_full = full_fit["sunrises"]
                    ss_full = full_fit["sunsets"]
                    e1 = sr_true - sr_full
                    e2 = ss_true - ss_full
                    e_both = np.r_[e1, e2]
                    full_error.append(np.sqrt(np.mean(e_both**2)))
            else:
                ho_error.append(1e2)
                full_error.append(1e2)
        ho_error = np.array(ho_error)
        min_val = np.min(ho_error)
        slct_vals = ho_error < 1.1 * min_val  # everything within 10% of min val
        selected_th = np.min(ths[slct_vals])
        bool_msk = detect_sun(data, selected_th)
        measured = rise_set_rough(bool_msk)
        smoothed = rise_set_smoothed(measured, sunrise_tau=self.sunrise_tau, sunset_tau=self.sunset_tau, solver=solver)
        self.sunrise_estimates = smoothed["sunrises"]
        self.sunset_estimates = smoothed["sunsets"]
        self.sunrise_measurements = measured["sunrises"]
        self.sunset_measurements = measured["sunsets"]
        self.sunup_mask_measured = bool_msk
        data_sampling = int(24 * 60 / data.shape[0])
        num_days = data.shape[1]
        mat = np.tile(np.arange(0, 24, data_sampling / 60), (num_days, 1)).T
        sr_broadcast = np.tile(self.sunrise_estimates, (data.shape[0], 1))
        ss_broadcast = np.tile(self.sunset_estimates, (data.shape[0], 1))
        self.sunup_mask_estimated = np.logical_and(
            mat >= sr_broadcast, mat < ss_broadcast
        )
        self.threshold = selected_th
        if self.true_times is not None:
            sr_residual = sr_true - self.sunrise_estimates
            ss_residual = ss_true - self.sunset_estimates
            total_rmse = np.sqrt(np.mean(np.r_[sr_residual, ss_residual] ** 2))
            self.total_rmse = total_rmse
        else:
            self.total_rmse = None

        if plot:
            fig = plt.figure(figsize=figsize)
            plt.plot(ths, ho_error, marker=".", color="blue", label="HO error")
            plt.yscale("log")
            plt.xscale("log")

            plt.plot(
                ths[slct_vals], ho_error[slct_vals], marker=".", ls="none", color="red"
            )
            plt.axvline(selected_th, color="blue", ls="--", label="optimized parameter")
            if self.true_times is not None:
                best_th = ths[np.argmin(full_error)]
                plt.plot(
                    ths, full_error, marker=".", color="orange", label="true error"
                )
                plt.axvline(best_th, color="orange", ls="--", label="best parameter")
            plt.legend()
            return fig
        else:
            return

    def calculate_errors(self):
        if self.true_times is not None:
            sr_true = self.true_times["sunrise times"].values
            ss_true = self.true_times["sunset times"].values
        else:
            print("please run .calculate_true() first")
            return
        r_sr_m = sr_true - self.sunrise_measurements
        r_ss_m = ss_true - self.sunset_measurements
        r_sr_e = sr_true - self.sunrise_estimates
        r_ss_e = ss_true - self.sunset_estimates
        r_tt_m = np.r_[r_sr_m, r_ss_m]
        r_tt_e = np.r_[r_sr_e, r_ss_e]
        r_sn_m = np.nanmean([sr_true, ss_true]) - np.nanmean(
            [self.sunrise_measurements, self.sunset_measurements]
        )
        r_sn_e = np.nanmean([sr_true, ss_true]) - np.nanmean(
            [self.sunrise_estimates, self.sunset_estimates]
        )
        r_dh_m = (ss_true - sr_true) - (
            self.sunset_measurements - self.sunrise_measurements
        )
        r_dh_e = (ss_true - sr_true) - (self.sunset_estimates - self.sunrise_estimates)

        def rmse(residual):
            return np.sqrt(np.mean(np.power(residual[~np.isnan(residual)], 2)))

        results_array = np.array(
            [
                [rmse(r_sr_m), rmse(r_sr_e)],
                [rmse(r_ss_m), rmse(r_ss_e)],
                [rmse(r_tt_m), rmse(r_tt_e)],
                [rmse(r_sn_m), rmse(r_sn_e)],
                [rmse(r_dh_m), rmse(r_dh_e)],
            ]
        )
        table = pd.DataFrame(
            columns=["measured", "estimated"],
            index=["sunrise", "sunset", "total_time", "solar_noon", "daylight_hours"],
            data=results_array,
        )
        return table


def sunset_hour_angle(doy, lat):
    b = np.deg2rad((360 / 365) * (doy - 1))
    delta = (
        0.006918
        - 0.399912 * np.cos(b)
        + 0.070257 * np.sin(b)
        - 0.006758 * np.cos(2 * b)
        + 0.000907 * np.sin(2 * b)
        - 0.002697 * np.cos(3 * b)
        + 0.00148 * np.sin(3 * b)
    )
    sunset_hour_angle = np.arccos(-np.tan(np.deg2rad(lat)) * np.tan(delta))
    return np.rad2deg(sunset_hour_angle)


def num_daylight_hours(doy, lat):
    return (2 / 15) * sunset_hour_angle(doy, lat)


def sunrise_sunset_times(lat, lon, doy, gmt_offset, eot="duffie"):
    ss_ha = sunset_hour_angle(doy, lat)
    ss_st = 12 + ss_ha / 15
    sr_st = 12 - ss_ha / 15
    ss_st *= 60
    sr_st *= 60
    ss_ct = solar_to_clock(ss_st, lon, doy, gmt_offset, eot)
    sr_ct = solar_to_clock(sr_st, lon, doy, gmt_offset, eot)
    ss_ct /= 60
    sr_ct /= 60
    output_table = pd.DataFrame(
        data={"day of year": doy, "sunrise times": sr_ct, "sunset times": ss_ct}
    )
    return output_table


def solar_to_clock(solar_time, lon, doy, gmt_offset, eot="duffie"):
    """
    :param solar_time: solar time in minutes since midnight (float or array)
    :param lon: longitude (float)
    :param doy: day of year (float or array)
    :param gmt_offset: local timezone offset in hours from UTC/GMT (float or int)
    :param eot: string specifying which equation of time formulation to use
    :return:
    """
    if eot.lower() in ("duffie", "d"):
        eot = eot_duffie(doy)
    elif eot.lower() in ("da_rosa", "dr"):
        eot = eot_da_rosa(doy)
    else:
        print("Please select either Duffie or Da Rosa for the equation of time")
        return
    st = solar_time
    ct = st - eot - 4 * (lon - 15 * gmt_offset)
    return ct


def clock_to_solar(clock_time, lon, doy, gmt_offset, eot="duffie"):
    if eot.lower() in ("duffie", "d"):
        eot = eot_duffie(doy)
    elif eot.lower() in ("da_rosa", "dr"):
        eot = eot_da_rosa(doy)
    else:
        print("Please select either Duffie or Da Rosa for the equation of time")
        return
    ct = clock_time
    st = ct + eot + 4 * (lon - 15 * gmt_offset)
    return st


def eot_da_rosa(day_of_year):
    """
    The equation of time as defined in:
        Haghdadi, Navid, et al. "A method to estimate the location and
        orientation of distributed photovoltaic systems from their generation
        output data." Renewable Energy 108 (2017): 390-400.
    These are equations (7) and (8) in the paper.
    :param day_of_year: the day of year, can be int, float, or numpy array
    :return: the difference between clock time and solar time for a given day of year
    """
    b = np.deg2rad((360 / 365) * (day_of_year - 81))
    eot = 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)
    try:
        return eot.values
    except AttributeError:
        return eot


def eot_duffie(day_of_year):
    """
    The equation of time as defined in:
        Duffie, John A., and William A. Beckman. Solar engineering of thermal
        processes. New York: Wiley, 1991.
    These are equations (1.4.2) and (1.5.3) in the book
    :param day_of_year: the day of year, can be int, float, or numpy array
    :return: the difference between clock time and solar time for a given day of year
    """
    b = np.deg2rad((360 / 365) * (day_of_year - 1))
    A = 1440 / (2 * np.pi)  # book uses approximation of 229.2
    eot = A * (
        0.000075
        + 0.001868 * np.cos(b)
        - 0.032077 * np.sin(b)
        - 0.014615 * np.cos(2 * b)
        - 0.04089 * np.sin(2 * b)
    )
    try:
        return eot.values
    except AttributeError:
        return eot
