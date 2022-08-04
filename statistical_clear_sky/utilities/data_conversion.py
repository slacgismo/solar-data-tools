import pandas as pd

def make_time_series(df, return_keys=True, localize_time=-8, filter_length=200):
    '''
    Accepts a Pandas data frame extracted from the Cassandra database. Returns a data frame with a single timestamp
    index and the data from different systems split into columns.
    :param df: A Pandas data from generated from a CQL query to the VADER Cassandra database
    :param return_keys: If true, return the mapping from data column names to site and system ID
    :param localize_time: If non-zero, localize the time stamps. Default is PST or UTC-8
    :param filter_length: The number of non-null data values a single system must have to be included in the output
    :return: A time-series data frame
    '''
    df.sort_values('ts', inplace=True)
    start = df.iloc[0]['ts']
    end = df.iloc[-1]['ts']
    time_index = pd.date_range(start=start, end=end, freq='5min')
    output = pd.DataFrame(index=time_index)
    site_keys = []
    site_keys_a = site_keys.append
    grouped = df.groupby(['site', 'sensor'])
    keys = grouped.groups.keys()
    counter = 1
    for key in keys:
        df_view = df.loc[grouped.groups[key]]
        ############## data cleaning ####################################
        df_view = df_view[pd.notnull(df_view['meas_val_f'])]            # Drop records with nulls
        df_view.set_index('ts', inplace=True)                           # Make the timestamp column the index
        df_view.sort_index(inplace=True)                                # Sort on time
        df_view = df_view[~df_view.index.duplicated(keep='first')]      # Drop duplicate times
        df_view.reindex(index=time_index).interpolate()                 # Match the master index, interp missing
        #################################################################
        meas_name = str(df_view['meas_name'][0])
        col_name = meas_name + '_{:02}'.format(counter)
        output[col_name] = df_view['meas_val_f']
        if output[col_name].count() > filter_length:  # final filter on low data count relative to time index
            site_keys_a((key, col_name))
            counter += 1
        else:
            del output[col_name]
    if localize_time:
        output.index = output.index + pd.Timedelta(hours=localize_time)  # Localize time

    if return_keys:
        return output, site_keys
    else:
        return output
