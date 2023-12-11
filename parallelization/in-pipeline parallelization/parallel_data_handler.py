# from solardatatools import DataHandler
from solardatatools.data_handler import DailySignals, DataHandler, DailyFlags, DailyScores, BooleanMasks
from dask.distributed import Client
import dask.delayed
import pandas as pd
import os

from time import time
from datetime import timedelta
from datetime import datetime
import numpy as np
import pandas as pd
import cvxpy as cvx
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import traceback, sys
from solardatatools.time_axis_manipulation import (
    make_time_series,
    standardize_time_axis,
    remove_index_timezone,
    get_index_timezone,
)
from solardatatools.matrix_embedding import make_2d
from solardatatools.data_quality import (
    make_density_scores,
    make_linearity_scores,
    make_quality_flags,
)
from solardatatools.data_filling import zero_nighttime, interp_missing
from solardatatools.clear_day_detection import ClearDayDetection
from solardatatools.plotting import plot_2d
from solardatatools.clear_time_labeling import find_clear_times
from solardatatools.solar_noon import avg_sunrise_sunset
from solardatatools.algorithms import (
    CapacityChange,
    TimeShift,
    SunriseSunset,
    ClippingDetection,
)
from pandas.plotting import register_matplotlib_converters
from copy import deepcopy

register_matplotlib_converters()
from solardatatools.polar_transform import PolarTransform

def make_data_matrix(inputs, use_col=None, start_day_ix=None, end_day_ix=None):
    df = inputs["data_frame"]
    if use_col is None:
        use_col = df.columns[0]
    inputs["raw_data_matrix"], day_index = make_2d(df, key=use_col, return_day_axis=True)
    inputs["raw_data_matrix"] = inputs["raw_data_matrix"][:, start_day_ix:end_day_ix]
    inputs["num_days"] = inputs["raw_data_matrix"].shape[1]
    if inputs["raw_data_matrix"].shape[0] <= 1400:
        inputs["data_sampling"] = int(24 * 60 / inputs["raw_data_matrix"].shape[0])
    else:
        inputs["data_sampling"] = 24 * 60 / inputs["raw_data_matrix"].shape[0]
    inputs["use_column"] = use_col
    inputs["day_index"] = day_index[start_day_ix:end_day_ix]
    d1 = inputs["day_index"][0].strftime("%x")
    d2 = inputs["day_index"][-1].strftime("%x")
    inputs["data_frame"] = inputs["data_frame"][d1:d2]
    inputs["start_doy"] = inputs["day_index"].dayofyear[0]
    return

def make_filled_data_matrix(inputs, zero_night=True, interp_day=True):
    inputs["filled_data_matrix"] = np.copy(inputs["raw_data_matrix"])
    if zero_night:
        inputs["filled_data_matrix"] = zero_nighttime(
            inputs["raw_data_matrix"], night_mask=~inputs["boolean_masks"].daytime
        )
    if interp_day:
        inputs["filled_data_matrix"] = interp_missing(inputs["filled_data_matrix"])
    else:
        msk = np.isnan(inputs["filled_data_matrix"])
        inputs["filled_data_matrix"][msk] = 0
    inputs["daily_signals"].energy = (
        np.sum(inputs["filled_data_matrix"], axis=0)
        * 24
        / inputs["filled_data_matrix"].shape[1]
    )
    return


def get_daily_scores(inputs, threshold=0.2, solver=None):
    get_density_scores(inputs, threshold=threshold, solver=solver)
    get_linearity_scores(inputs)
    return

def get_daily_flags(
    inputs,
    density_lower_threshold=0.6,
    density_upper_threshold=1.05,
    linearity_threshold=0.1,
):
    df, lf = make_quality_flags(
        inputs["daily_scores"].density,
        inputs["daily_scores"].linearity,
        density_lower_threshold=density_lower_threshold,
        density_upper_threshold=density_upper_threshold,
        linearity_threshold=linearity_threshold,
    )
    inputs["daily_flags"].density = df
    inputs["daily_flags"].linearity = lf
    inputs["daily_flags"].flag_no_errors()
    # scores should typically cluster within threshold values, if they
    # don't, we mark normal_quality_scores as false
    scores = np.c_[inputs["daily_scores"].density, inputs["daily_scores"].linearity]
    db = DBSCAN(eps=0.03, min_samples=int(max(0.01 * scores.shape[0], 3))).fit(
        scores
    )
    # Count the number of days that cluster to the main group but fall
    # outside the decision boundaries
    day_counts = [
        np.logical_or(
            inputs["daily_scores"].linearity[db.labels_ == lb] > linearity_threshold,
            np.logical_or(
                inputs["daily_scores"].density[db.labels_ == lb]
                < density_lower_threshold,
                inputs["daily_scores"].density[db.labels_ == lb]
                > density_upper_threshold,
            ),
        )
        for lb in set(db.labels_)
    ]
    inputs["normal_quality_scores"] = np.any(
        [
            np.sum(day_count) <= max(5e-3 * inputs["num_days"], 1)
            for day_count in day_counts
        ]
    )
    inputs["__density_lower_threshold"] = density_lower_threshold
    inputs["__density_upper_threshold"] = density_upper_threshold
    inputs["__linearity_threshold"] = linearity_threshold
    inputs["daily_scores"].quality_clustering = db.labels_


def get_density_scores(inputs, threshold=0.2, solver=None):
    if inputs["raw_data_matrix"] is None:
        print("Generate a raw data matrix first.")
        return
    s1, s2, s3 = make_density_scores(
        inputs["raw_data_matrix"],
        threshold=threshold,
        return_density_signal=True,
        return_fit=True,
        solver=solver,
    )
    inputs["daily_scores"].density = s1
    inputs["daily_signals"].density = s2
    inputs["daily_signals"].seasonal_density_fit = s3
    return


def get_linearity_scores(inputs):
    if inputs["capacity_estimate"] is None:
        inputs["capacity_estimate"] = np.quantile(inputs["filled_data_matrix"], 0.95)
    if inputs["daily_signals"].seasonal_density_fit is None:
        print("Run the density check first")
        return
    ls, im = make_linearity_scores(
        inputs["filled_data_matrix"],
        inputs["capacity_estimate"],
        inputs["daily_signals"].seasonal_density_fit,
    )
    inputs["daily_scores"].linearity = ls
    inputs["boolean_masks"].infill = im
    return

def run_pipeline_in_parallel(df,
                             power_col=None,
                             min_val=-5,
                             max_val=None,
                             zero_night=True,
                             interp_day=True,
                             fix_shifts=False,
                             density_lower_threshold=0.6,
                             density_upper_threshold=1.05,
                             linearity_threshold=0.1,
                             clear_day_smoothness_param=0.9,
                             clear_day_energy_param=0.8,
                             verbose=True,
                             start_day_ix=None,
                             end_day_ix=None,
                             w1=None,
                             w2=1e5,
                             periodic_detector=False,
                             solar_noon_estimator="srss",
                             correct_tz=True,
                             extra_cols=None,
                             daytime_threshold=0.005,
                             units="W",
                             solver="QSS",
                             solver_convex="CLARABEL",
                             reset=True,):
    def initialize_inputs(data_frame=None,
                          raw_data_matrix=None,
                          datetime_col=None,
                          convert_to_ts=False,
                          no_future_dates=True,
                          aggregate=None,
                          how=lambda x: x.mean(),
                          gmt_offset=None,):
        results = {}
        if data_frame is not None:
            if convert_to_ts:
                data_frame, keys = make_time_series(data_frame)
                results["keys"] = keys
            else:
                results["keys"] = list(data_frame.columns)
            results["data_frame_raw"] = data_frame.copy()
            seq_index = np.arange(len(results["data_frame_raw"]))
            if isinstance(results["keys"][0], tuple) and not convert_to_ts:
                num_levels = len(results["keys"][0])
                results["seq_index_key"] = tuple(["seq_index"] * num_levels)
            else:
                results["seq_index_key"] = "seq_index"
            results["data_frame_raw"][results["seq_index_key"]] = seq_index
            if not isinstance(results["data_frame_raw"].index, pd.DatetimeIndex):
                if datetime_col is not None:
                    df = results["data_frame_raw"]
                    df[datetime_col] = pd.to_datetime(df[datetime_col])
                    df.set_index(datetime_col, inplace=True)
                else:
                    e = "Data frame must have a DatetimeIndex or"
                    e += "the user must set the datetime_col kwarg."
                    raise Exception(e)
            results["tz_info"] = get_index_timezone(results["data_frame_raw"])
            results["data_frame_raw"] = remove_index_timezone(results["data_frame_raw"])
            if no_future_dates:
                now = datetime.now()
                results["data_frame_raw"] = results["data_frame_raw"][
                    results["data_frame_raw"].index <= now
                ]
            results["data_frame"] = None
            if aggregate is not None:
                new_data = how(results["data_frame_raw"].resample(aggregate))
                results["data_frame_raw"] = new_data
        else:
            results["data_frame_raw"] = None
            results["data_frame"] = None
            results["keys"] = None
        results["raw_data_matrix"] = raw_data_matrix
        if results["raw_data_matrix"] is not None:
            results["num_days"] = results["raw_data_matrix"].shape[1]
            if results["raw_data_matrix"].shape[0] <= 1400:
                results["data_sampling"] = int(24 * 60 / results["raw_data_matrix"].shape[0])
            else:
                results["data_sampling"] = 24 * 60 / results["raw_data_matrix"].shape[0]
        else:
            results["num_days"] = None
            results["data_sampling"] = None
        results["gmt_offset"] = gmt_offset
        results["power_units"] = units
        
        results["filled_data_matrix"] = None
        results["use_column"] = None
        results["capacity_estimate"] = None
        results["start_doy"] = None
        results["day_index"] = None
        ## "Extra" data, i.e. additional columns to process from the table ##
        # Matrix views of extra columns
        results["extra_matrices"] = {}
        # Relative quality: fraction of non-NaN values in column during
        # daylight time periods, as defined by the main power columns
        results["extra_quality_scores"] = {}
        ## Scores for the entire data set ##
        # Fraction of days without data acquisition errors
        results["data_quality_score"] = None
        # Fraction of days that are approximately clear/sunny
        results["data_clearness_score"] = None
        ##  Flags for the entire data set ##
        # True if there is inverter clipping, false otherwise
        results["inverter_clipping"] = None
        # If clipping, the number of clipping set points
        results["num_clip_points"] = None
        # True if the apparent capacity seems to change over the data set
        results["capacity_changes"] = None
        # True if clustering of data quality scores are within decision boundaries
        results["normal_quality_scores"] = None
        # True if time shifts detected and corrected in data set
        results["time_shifts"] = None
        # TZ correction factor (determined during pipeline run)
        results["tz_correction"] = 0
        # Daily scores (floats), flags (booleans), and boolean masks
        results["daily_scores"] = DailyScores()  # 1D arrays of floats
        results["daily_flags"] = DailyFlags()  # 1D arrays of Booleans
        results["boolean_masks"] = BooleanMasks()  # 2D arrays of Booleans
        # Useful daily signals defined by the data set
        results["daily_signals"] = DailySignals()
        # Algorithm objects
        results["scsf"] = None
        results["capacity_analysis"] = None
        results["time_shift_analysis"] = None
        results["daytime_analysis"] = None
        results["clipping_analysis"] = None
        results["clear_day_analysis"] = None
        results["parameter_estimation"] = None
        results["polar_transform"] = None
        # Private attributes
        results["_ran_pipeline"] = False
        results["_error_msg"] = ""
        results["__density_lower_threshold"] = None
        results["__density_upper_threshold"] = None
        results["__linearity_threshold"] = None
        results["__recursion_depth"] = 0
        results["__initial_time"] = None
        results["__fix_dst_ran"] = False
        return results


    def preprocess(inputs,
                 power_col=None,
                   ):
        if inputs["data_frame_raw"] is not None:
            # If power_col not passed, assume that the first column contains the
            # data to be processed
            if power_col is None:
                power_col = inputs["data_frame_raw"].columns[0]
            if power_col not in inputs["data_frame_raw"].columns:
                print("Power column key not present in data frame.")
                return
            # Pandas operations to make a time axis with regular intervals.
            # If correct_tz is True, it will also align the median daily maximum
            inputs["data_frame"], sn_deviation = standardize_time_axis(
                inputs["data_frame_raw"],
                timeindex=True,
                power_col=power_col,
                correct_tz=correct_tz,
                verbose=verbose,
            )
            if correct_tz:
                inputs["tz_correction"] = sn_deviation
        # Embed the data as a matrix, with days in columns. Also, set some
        # attributes, like the scan rate, day index, and day of year arary.
        # Almost never use start_day_ix and end_day_ix, but they're there
        # if a user wants to use a portion of the data set.
        if inputs["data_frame"] is not None:
            make_data_matrix(
                inputs, use_col=power_col, start_day_ix=start_day_ix, end_day_ix=end_day_ix
            )

        if max_val is not None:
            mat_copy = np.copy(inputs["raw_data_matrix"])
            mat_copy[np.isnan(mat_copy)] = -9999
            slct = mat_copy > max_val
            if np.sum(slct) > 0:
                inputs["raw_data_matrix"][slct] = np.nan
        if min_val is not None:
            mat_copy = np.copy(inputs["raw_data_matrix"])
            mat_copy[np.isnan(mat_copy)] = 9999
            slct = mat_copy < min_val
            if np.sum(slct) > 0:
                inputs["raw_data_matrix"][slct] = np.nan
        inputs["capacity_estimate"] = np.nanquantile(inputs["raw_data_matrix"], 0.95)
        if inputs["capacity_estimate"] <= 500 and inputs["power_units"] == "W":
            inputs["power_units"] = "kW"
        inputs["boolean_masks"].missing_values = np.isnan(inputs["raw_data_matrix"])
        # Run once to get a rough estimate. Update at the end after cleaning
        # is finished
        ss = SunriseSunset()
        try:
            ss.run_optimizer(inputs["raw_data_matrix"], plot=False, solver=solver_convex)
            inputs["boolean_masks"].daytime = ss.sunup_mask_estimated
        except:
            msg = "Sunrise/sunset detection failed."
            # self._error_msg += "\n" + msg
            if verbose:
                print(msg)
                traceback.print_exception(*sys.exc_info())
        inputs["daytime_analysis"] = ss
        return inputs

    def cleaning(inputs):
        try:
            make_filled_data_matrix(inputs, zero_night=zero_night, interp_day=interp_day)
        except:
            msg = "Matrix filling failed."
            # self._error_msg += "\n" + msg
            if verbose:
                print(msg)
                traceback.print_exception(*sys.exc_info())
        num_raw_measurements = np.count_nonzero(
            np.nan_to_num(inputs["raw_data_matrix"], copy=True, nan=0.0)[
                inputs["boolean_masks"].daytime
            ]
        )
        num_filled_measurements = np.count_nonzero(
            np.nan_to_num(inputs["filled_data_matrix"], copy=True, nan=0.0)[
                inputs["boolean_masks"].daytime
            ]
        )
        if num_raw_measurements > 0:
            ratio = num_filled_measurements / num_raw_measurements
        else:
            msg = "Error: data set contains no non-zero values!"
            inputs["_error_msg"] += "\n" + msg
            if verbose:
                print(msg)
            inputs["daily_scores"] = None
            inputs["daily_flags"] = None
            inputs["data_quality_score"] = 0.0
            inputs["data_clearness_score"] = 0.0
            inputs["_ran_pipeline"] = True
            return
        if ratio < 0.85:
            msg = "Error: data was lost during NaN filling procedure. "
            msg += "This typically occurs when\nthe time stamps are in the "
            msg += "wrong timezone. Please double check your data table.\n"
            inputs["_error_msg"] += "\n" + msg
            if verbose:
                print(msg)
                print(f"ratio of filled to raw nonzero measurements was {ratio:.2f}")
            inputs["daily_scores"] = None
            inputs["daily_flags"] = None
            inputs["data_quality_score"] = None
            inputs["data_clearness_score"] = None
            inputs["_ran_pipeline"] = True
            return
        return inputs

    def get_daily_scores_and_flags(inputs):
        try:
            # density scoring
            get_daily_scores(inputs, threshold=0.2, solver=solver_convex)
        except:
            msg = "Daily quality scoring failed."
            inputs["_error_msg"] += "\n" + msg
            if verbose:
                print(msg)
                traceback.print_exception(*sys.exc_info())
            inputs["daily_scores"] = None
            return
        try:
            get_daily_flags(
                inputs,
                density_lower_threshold=density_lower_threshold,
                density_upper_threshold=density_upper_threshold,
                linearity_threshold=linearity_threshold,
            )
        except:
            msg = "Daily quality flagging failed."
            inputs["_error_msg"] += "\n" + msg
            if verbose:
                print(msg)
                traceback.print_exception(*sys.exc_info())
            inputs["daily_flags"] = None
            return
        return inputs

    # @dask.delayed
    def clipping_check():
        return


    @dask.delayed
    def detect_clear_days(inputs,
                          smoothness_threshold=0.9, 
                          energy_threshold=0.8, 
                          solver=None
                          ):
        results = {}
        if inputs["filled_data_matrix"] is None:
            print("Generate a filled data matrix first.")
            return
        results["clear_day_analysis"] = ClearDayDetection()
        clear_days = results["clear_day_analysis"].find_clear_days(
            inputs["filled_data_matrix"],
            smoothness_threshold=smoothness_threshold,
            energy_threshold=energy_threshold,
            solver=solver,
        )
        # Remove days that are marginally low density, but otherwise pass
        # the clearness test. Occasionally, we find an early morning or late
        # afternoon inverter outage on a clear day is still detected as clear.
        # Added July 2020 --BM
        clear_days = np.logical_and(clear_days, inputs["daily_scores"].density > 0.9)
        results["daily_flags"] = deepcopy(inputs["daily_flags"])
        results["daily_flags"].flag_clear_cloudy(clear_days)
        return results


    # @dask.delayed
    def score_dataset():
        return
    # @dask.delayed
    def post_processing():
        return

    if solver == "MOSEK":
        # Set all problems to use MOSEK
        # and check that MOSEK is installed
        solver_convex = "MOSEK"
        try:
            x = cvx.Variable()
            prob = cvx.Problem(cvx.Minimize(cvx.sum_squares(x)))
            prob.solve(solver="MOSEK")
        except Exception as e:
            print("VALID MOSEK LICENSE NOT AVAILABLE")
            print(
                "please check that your license file is in [HOME]/mosek and is current\n"
            )
            print("error msg:", e)
            return

    inputs = initialize_inputs(df, convert_to_ts=True)
    t = np.zeros(6)
    t[0] = time()
    inputs = preprocess(inputs, power_col=power_col)
    t[1] = time()
    inputs = cleaning(inputs)
    t[2] = time()
    t_clean = np.zeros(6)
    t_clean[0] = time()
    inputs = get_daily_scores_and_flags(inputs)
    t_clean[1] = time()
    try:
        clear_day_results = detect_clear_days(
            inputs,
            smoothness_threshold=clear_day_smoothness_param,
            energy_threshold=clear_day_energy_param,
            solver=solver_convex,
        )
    except:
        msg = "Clear day detection failed."
        inputs["_error_msg"] += "\n" + msg
        if verbose:
            print(msg)
            traceback.print_exception(*sys.exc_info())
        return
    t_clean[2] = time()
    print(inputs)
    print(clear_day_results)


def run_job(data, data_retrieval_fn):
    """
    Processes a single unit of data using DataHandler.

    Parameters:
    - data: The input data to be processed.
    - data_retrieval_fn: Function to retrieve and format format each data entry.
                         Should return a tuple with the name of the site and a
                         pandas data_frame of the solar data.

    Returns:
    - A dictionary containing the name of the data and the processed report.
    """
    name, data_frame = data_retrieval_fn(data)
    # data_handler = DataHandler(data_frame, convert_to_ts=True,)
    # data_handler.run_pipeline(power_col='ac_power_01', solver_convex="OSQP")
    # data_handler.run_pipeline(power_col='ac_power_01', )
    data = run_pipeline_in_parallel(data_frame, power_col='ac_power_01')
    report = {}
    # report["name"] = name
    # report["data"] = data
    return report


def local_csv_to_df(file):
    """
    Converts a local CSV file into a pandas DataFrame.

    Parameters:
    - file: Path to the CSV file.

    Returns:
    - A tuple of the file name and its corresponding DataFrame.
    """
    df = pd.read_csv(file)
    name = os.path.basename(file)
    return name, df


class ParallelDataHandler:
    """
    A class to handle and process multiple data entries in parallel.
    """
    def __init__(self, data_list, data_retrieval_fn):
        """
        Initializes the ParallelDataHandler with a list of data and a retrieval function.

        Parameters:
        - data_list: A list containing data entries to be processed.
    - data_retrieval_fn: Function to retrieve and format format each data entry.
                         Should return a tuple with the name of the site and a
                         pandas data_frame of the solar data.
        """

        self.data_retrieval_fn = data_retrieval_fn
        self.data_list = data_list

    def run_pipelines_in_parallel(self):
        """
        Executes the data processing pipelines in parallel on the provided data list.

        Returns:
        - A list of reports generated from processing each data entry.
        """
        reports = []
        for data in self.data_list:
            reports.append(run_job(data, self.data_retrieval_fn))
        computed_reports = dask.compute(*reports)
        return computed_reports


if __name__ == "__main__":
    # Set up a Dask client for parallel computation
    client = Client()
    files = ['~/Schoolwork/Practicum/solar-data-tools/tmp_data/low_time_0022F200152D.csv',]

    # Initialize the ParallelDataHandler with the list of files and the CSV-to-DataFrame conversion function
    local_hander = ParallelDataHandler(files, local_csv_to_df)

    reports = local_hander.run_pipelines_in_parallel()
    print(reports)
