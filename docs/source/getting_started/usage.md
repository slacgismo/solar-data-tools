# General Usage

Users will primarily interact with this software through the `DataHandler` class. By default, Solar Data 
Tools uses CLARABEL as the solver for all signal decomposition problems. If you would like 
to specify a solver (such as MOSEK), just pass the keyword argument `solver` to `DataHandler.pipeline` with the solver of choice.

```python
from solardatatools import DataHandler
from solardatatools.dataio import get_pvdaq_data

pv_system_data = get_pvdaq_data(sysid=35, api_key='DEMO_KEY', year=[2011, 2012, 2013])

dh = DataHandler(pv_system_data)
dh.run_pipeline(power_col='dc_power')
```

If everything is working correctly, you should see something like the following:

```bash
total time: 25.99 seconds
--------------------------------
Breakdown
--------------------------------
Preprocessing              6.76s
Cleaning                   0.41s
Filtering/Summarizing      18.83s
    Data quality           0.21s
    Clear day detect       0.44s
    Clipping detect        15.51s
    Capacity change detect 2.67s
```

You can find more in-depth usage examples in the [demo](./notebooks/demo_default.ipynb) and [tutorial](./notebooks/tutorial.ipynb) notebooks.