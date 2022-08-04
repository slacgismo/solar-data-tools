import numpy as np
import pandas as pd

TZ_LOOKUP = {
    'America/Anchorage': 9,
    'America/Chicago': 6,
    'America/Denver': 7,
    'America/Los_Angeles': 8,
    'America/New_York': 5,
    'America/Phoenix': 7,
    'Pacific/Honolulu': 10
}

def load_results():
    base = 's3://pvinsight.nrel/output/'
    nrel_data = pd.read_csv(base + 'pvo_results.csv')
    slac_data = pd.read_csv(base + 'scsf-unified-results.csv')
    slac_data['all-pass'] = np.logical_and(
        np.alltrue(np.logical_not(slac_data[['solver-error', 'f1-increase', 'obj-increase']]), axis=1),
        np.isfinite(slac_data['deg'])
    )
    cols = ['ID', 'rd', 'deg', 'rd_low', 'rd_high', 'all-pass',
            'fix-ts', 'num-days', 'num-days-used', 'use-frac',
            'res-median', 'res-var', 'res-L0norm']
    df = pd.merge(nrel_data, slac_data, how='left', left_on='datastream', right_on='ID')
    df = df[cols]
    df.set_index('ID', inplace=True)
    df = df[df['all-pass'] == True]
    df['deg'] = df['deg'] * 100
    df['difference'] = df['rd'] - df['deg']
    df['rd_range'] = df['rd_high'] - df['rd_low']
    cols = ['rd', 'deg', 'difference', 'rd_range',
            'res-median', 'res-var', 'res-L0norm', 'rd_low', 'rd_high', 'all-pass',
            'fix-ts', 'num-days', 'num-days-used', 'use-frac']
    df = df[cols]
    return df

def load_sys(n=None, idnum=None, local=True, meta=None):
    if local:
        base = '../data/PVO/'
    if not local:
        base = 's3://pvinsight.nrel/PVO/'
    if meta is None:
        meta = pd.read_csv('s3://pvinsight.nrel/PVO/sys_meta.csv')
    if n is not None:
        idnum = meta['ID'][n]
    elif idnum is not None:
        n = meta[meta['ID'] == idnum].index[0]
    else:
        print('must provide index or ID')
        return
    df = pd.read_csv(base+'PVOutput/{}.csv'.format(idnum), index_col=0,
                      parse_dates=[0], usecols=[1, 3])
    tz = meta['TimeZone'][n]
    df.index = df.index.tz_localize(tz).tz_convert('Etc/GMT+{}'.format(TZ_LOOKUP[tz]))   # fix daylight savings
    start = df.index[0]
    end = df.index[-1]
    time_index = pd.date_range(start=start, end=end, freq='5min')
    df = df.reindex(index=time_index, fill_value=0)
    print(n, idnum)
    return df

def resample_index(length=365*5):
    indices = np.arange(length)
    resampled = np.random.choice(indices, size=length, replace=True)
    weights = np.zeros(length)
    for index in resampled:
        weights[index] += 1
    return weights
