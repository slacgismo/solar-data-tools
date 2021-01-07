Usage
-----

Clear Day Detection
^^^^^^^^^^^^^^^^^^^

This algorithm estimates the clear days in a data set two ways and then
combines the estimates for the final estimations. The first estimate is
based on the "smoothness" of each daily power signal. The second
estimate is based on the seasonally adjusted daily energy output of the
system.

.. code:: python

    import numpy as np
    from solardatatools.clear_day_detection import find_clear_days
    from solardatatools.data_transforms import make_2d
    from solardatatools.dataio import get_pvdaq_data

    pv_system_data = get_pvdaq_data(sysid=35, api_key='DEMO_KEY', year=[2011, 2012, 2013])

    power_signals_d = make_2d(pv_system_data, key='dc_power')

    clear_days = find_clear_days(power_signals_d)

Time Shift Detection and Fixing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This algorithm determines if the time stamps provided with the data have
"shifted" at any point and then corrects the shift if found. These
shifts can often be caused by incorrect handling of daylight savings
time, but can come from other sources as well.

.. code:: python

    from solardatatools.data_transforms import fix_time_shifts, make_2d
    from solardatatools.dataio import get_pvdaq_data
    from solardatatools.plotting import plot_2d

    pv_system_data = get_pvdaq_data(sysid=1199, year=[2015, 2016, 2017], api_key='DEMO_KEY')

    power_signals_d = make_2d(pv_system_data, key='dc_power')

    fixed_power_signals_d, time_shift_days_indices_ixs = fix_time_shifts(
        power_signals_d, return_ixs=True)
