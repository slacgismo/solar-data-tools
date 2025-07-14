import pandas as pd
import numpy as np



def divide_df(data_frame,datetime_col = None):
    """
    This function divide a dataframe with a single column timeIndex and multiple columns for the values
    into dataframes with a uniuqe column timeIndex and unique column values.
    """
    if datetime_col is not None:
        if datetime_col not in data_frame.columns:
            raise ValueError(f"'{datetime_col}' is not a column in the DataFrame.")
        df = data_frame.set_index(pd.to_datetime(data_frame[datetime_col]))
        df = df.drop(columns=[datetime_col])
    else:
        if not isinstance(data_frame.index, pd.DatetimeIndex):
            raise ValueError("Data frame must have a DatetimeIndex or the user must set the datetime_col kwarg.")
        df = data_frame

    result = {col: df[[col]].copy() for col in df.columns}
    return result


def common_days(dh_dict):
    """
    This function takes a dictionary of DataHandlers and returns a dictionary of boolean NumPy arrays,
    where each value indicates whether the corresponding day is a common good day.
    """
    common_good_days = None  
    for dh in dh_dict.values():
        good_days = set(dh.day_index[dh.daily_flags.no_errors])
        if common_good_days is None:
            common_good_days = good_days
        else:
            common_good_days &= good_days 
    result = {}
    for name, dh in dh_dict.items():
        result[name] = np.isin(
            dh.day_index.astype('datetime64[ns]'),
            np.array(list(common_good_days)).astype('datetime64[ns]')
        )
    return result

