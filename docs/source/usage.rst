*************
General Usage
*************

Users will primarily interact with this software through the ``DataHandler`` class. If you would like to specify a solver, just pass the keyword argument ``solver`` to ``dh.pipeline`` with the solver of choice. Passing QSS will keep the convex problems solver as OSQP, unless ``solver_convex=QSS`` is passed as well. Setting ``solver=MOSEK`` will set the solver to MOSEK for convex and non-convex problems by default.

.. code:: python

    from solardatatools import DataHandler
    from solardatatools.dataio import get_pvdaq_data

    pv_system_data = get_pvdaq_data(sysid=35, api_key='DEMO_KEY', year=[2011, 2012, 2013])

    dh = DataHandler(pv_system_data)
    dh.run_pipeline(power_col='dc_power')

If everything is working correctly, you should see something like the following:

.. code:: bash

    total time: 24.27 seconds
    --------------------------------
    Breakdown
    --------------------------------
    Preprocessing              11.14s
    Cleaning                   0.94s
    Filtering/Summarizing      12.19s
        Data quality           0.25s
        Clear day detect       1.75s
        Clipping detect        7.77s
        Capacity change detect 2.42s

For more in-depth examples, see the `notebooks <https://github.com/slacgismo/solar-data-tools/tree/master/notebooks/examples>`_ folder.