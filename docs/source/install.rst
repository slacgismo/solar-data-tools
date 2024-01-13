.. _install:

***************
Install & Setup
***************

There are three ways of setting up, either approach works. Below we describe all three methods, in addition to the solvers needed to run the Solar Data Tools pipeline.

1) Set up `conda` environment with provided `.yml` file (Recommended)
---------------------------------------------------------------------

We recommend setting up a fresh Python virtual environment in which to use `solar-data-tools`. We recommend using the `Conda <https://docs.conda.io/projects/conda/en/latest/index.html>`_ package management system, and creating an environment with the environment configuration file named `pvi-user.yml`, provided in the top level of this repository.

Creating the env:

.. code:: sh

    $ conda env create -f pvi-user.yml

Starting the env:

.. code:: sh

    $ conda activate pvi_user

Stopping the env

.. code:: sh

    $ conda deactivate

Updating the env with latest

.. code:: sh

    $ conda env update -f pvi-user.yml

Additional documentation on setting up the Conda environment is available `here <https://github.com/slacgismo/pvinsight-onboarding/blob/main/README.md>`_.


2) PIP Package
--------------

.. code:: sh

    $ pip install solar-data-tools

Alternative: Clone repo from GitHub

Mimic the pip package by setting up locally.

.. code:: sh

    $ pip install -e path/to/root/folder

3) Anaconda Package
-------------------

.. code:: sh

    $ conda install -c slacgismo solar-data-tools


Solvers
=======

QSS & OSQP
----------

By default, `QSS <https://github.com/cvxgrp/qss>`_ and OSQP solvers are used for non-convex and convex problems, respectively. Both are supported by `OSD <https://github.com/cvxgrp/signal-decomposition/tree/main>`_, the modeling language used to solve signal decomposition problems in Solar Data Tools, and both are open source.

MOSEK
-----

MOSEK is a commercial software package. It is more stable and offers faster solve times. The included YAML/requirements.txt file will install MOSEK for you, but you will still need to obtain a license. More information is available here:

- `mosek <https://www.mosek.com/resources/getting-started/>`_
- `Free 30-day trial <https://www.mosek.com/products/trial/>`_
- `Personal academic license <https://www.mosek.com/products/academic-licenses/>`_