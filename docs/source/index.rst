solar-data-tools
================

|PyPI release| |Anaconda Cloud release|

Tools for performing common tasks on solar PV data signals. These tasks include finding clear days in a data set, common data transforms, and fixing time stamp issues. These tools are designed to be automatic and require little if any input from the user. Libraries are included to help with data IO and plotting as well.

There is close integration between this repository and the `Statistical Clear Sky <https://github.com/slacgismo/StatisticalClearSky>`_ repository, which provides a "clear sky model" of system output, given only measured power as an input.

See `notebooks <https://github.com/slacgismo/solar-data-tools/tree/master/notebooks>`_ folder for examples.

Contents
^^^^^^^^

.. toctree::
   :maxdepth: 4

   setup
   usage
   versioning
   authors
   license
   modules



Indices and tables
==================

* :ref:`genindex`
* :ref:`search`


.. |PyPI release| image:: https://img.shields.io/pypi/v/solar-data-tools.svg
   :target: https://pypi.org/project/solar-data-tools/
.. |Anaconda Cloud release| image:: https://anaconda.org/slacgismo/solar-data-tools/badges/version.svg
   :target: https://anaconda.org/slacgismo/solar-data-tools
