# solar-data-tools

<table>
<tr>
  <td>Latest Release</td>
  <td>
    <a href="https://pypi.org/project/solar-data-tools/">
        <img src="https://img.shields.io/pypi/v/solar-data-tools.svg" alt="latest release" />
    </a>
    <a href="https://anaconda.org/slacgismo/solar-data-tools">
        <img src="https://anaconda.org/slacgismo/solar-data-tools/badges/version.svg" />
    </a>
    <a href="https://anaconda.org/slacgismo/solar-data-tools">
        <img src="https://anaconda.org/slacgismo/solar-data-tools/badges/latest_release_date.svg" />
    </a>
</tr>
<tr>
  <td>License</td>
  <td>
    <a href="https://github.com/slacgismo/solar-data-tools/blob/master/LICENSE">
        <img src="https://img.shields.io/pypi/l/solar-data-tools.svg" alt="license" />
    </a>
</td>
</tr>
<tr>
  <td>Build Status</td>
  <td>
    <a href="https://solar-data-tools.readthedocs.io/">
        <img src="https://readthedocs.org/projects/solar-data-tools/badge/?version=stable" alt="documentation build status" />
    </a>
        <a href="https://github.com/slacgismo/solar-data-tools/actions/workflows/test.yml">
        <img src="https://github.com/slacgismo/solar-data-tools/actions/workflows/test.yml/badge.svg?branch=master" alt="Actions build status" />
    </a>
    <!-- switch below from tadatoshi to gismo -->
    <a href="https://travis-ci.com/tadatoshi/solar-data-tools.svg?branch=development">
        <img src="https://travis-ci.com/tadatoshi/solar-data-tools.svg?branch=development">
    </a>
  </td>
</tr>
<tr>
    <td>Publications</td>
    <td>
        <a href="https://zenodo.org/badge/latestdoi/171066536">
            <img src="https://zenodo.org/badge/171066536.svg" alt="DOI">
        </a>
    </td>
</tr>
<tr>
    <td>PyPI Downloads</td>
    <td>
        <a href="https://pepy.tech/project/solar-data-tools">
            <img src="https://img.shields.io/pypi/dm/solar-data-tools" alt="PyPI downloads" />
        </a>
    </td>
</tr>
<tr>
    <td>Conda Downloads</td>
    <td>
        <a href="https://anaconda.org/slacgismo/solar-data-tools">
            <img src="https://anaconda.org/slacgismo/solar-data-tools/badges/downloads.svg" alt="conda-forge downloads" />
        </a>
    </td>
</tr>
<tr>
    <td>Test-Coverage</td>
    <td>
        <img src="https://img.shields.io/badge/test--coverage-45%25-yellowgreen" alt="test-coverage" />
    </td>
</tr>
</table>

Tools for performing common tasks on solar PV data signals. These tasks include finding clear days in
a data set, common data transforms, and fixing time stamp issues. These tools are designed to be
automatic and require little if any input from the user. Libraries are included to help with data IO
and plotting as well.

See the [notebooks](https://github.com/slacgismo/solar-data-tools/blob/main/notebooks/examples) folder for examples.

## Install & Setup

#### Recommended: Install with pip

In a fresh Python virtual environment, simply run:

```bash
$ pip install solar-data-tools
```

or if you would like to use MOSEK, install the optional dependency as well:

```bash
$ pip install "solar-data-tools[mosek]"
```

#### Install with conda

>[!WARNING]
> When installing solar-data-tools using conda, you will need to add three channels, conda-forge, slacgismo, and stanfordcvxgrp, to your conda config (or alternatively specify them using the `-c` flag as shown in the examples below). Failure to do so will result in the installation of an outdated solar-data-tools version. Note that we will be moving solar-data-tools to conda-forge soon, which will simplify the installation process. Check back soon for an update! For more on conda channels, see the [conda docs](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-channels.html).

Creating the environment and directly installing the package and its dependencies from the appropriate conda channels:

```bash
$ conda create -n pvi-user solar-data-tools -c conda-forge -c slacgismo -c stanfordcvxgrp 
```

Starting the environment:

```bash
$ conda activate pvi-user
```

Stopping the environment:

```bash
$ conda deactivate
```

Or alternatively install the package in an already existing environment:

```bash
$ conda install solar-data-tools -c conda-forge -c slacgismo -c stanfordcvxgrp 
```

### Solvers

#### QSS & CLARABEL

By default, [QSS](https://github.com/cvxgrp/qss) and CLARABEL solvers are used for non-convex and convex problems, respectively. Both are supported by [OSD](https://github.com/cvxgrp/signal-decomposition/tree/main), the modeling language used to solve signal decomposition problems in Solar Data Tools, and both are open source. 

#### MOSEK

MOSEK is a commercial software package. Since it is more stable and offers faster solve times, we provide continuing support for it, however you will still need to obtain a license. If installing with pip, you can install the optional MOSEK dependency by running `pip install "solar-data-tools[mosek]"`. If installing from conda, you will have to manually install MOSEK if you desire to use it as conda does not support optional dependencies like pip. 

More information about MOSEK and how to obtain a license is available here:

* [mosek](https://www.mosek.com/resources/getting-started/)
* [Free 30-day trial](https://www.mosek.com/products/trial/)
* [Personal academic license](https://www.mosek.com/products/academic-licenses/)

## Usage

Users will primarily interact with this software through the `DataHandler` class. If you would like to specify a solver, just pass the keyword argument `solver` to `dh.pipeline` with the solver of choice. Passing QSS will keep the convex problems solver as OSQP, unless `solver_convex=QSS` is passed as well. Setting `solver=MOSEK` will set the solver to MOSEK for convex and non-convex problems by default.

```python
from solardatatools import DataHandler
from solardatatools.dataio import get_pvdaq_data

pv_system_data = get_pvdaq_data(sysid=35, api_key='DEMO_KEY', year=[2011, 2012, 2013])

dh = DataHandler(pv_system_data)
dh.run_pipeline(power_col='dc_power')
```
If everything is working correctly, you should see something like the following

```
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
```

## Contributors

Must enable pre-commit hook before pushing any contributions
```
pip install pre-commit
pre-commit install
```

Run pre-commit hook on all files
```
pre-commit run --all-files
```

## Test Coverage

In order to view the current test coverage metrics, run:
```
coverage run --source solardatatools -m unittest discover && coverage html
open htmlcov/index.html
```

## Versioning

We use [Semantic Versioning](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/slacgismo/solar-data-tools/tags).

## Authors

* **Bennet Meyers** - *Initial work and Main research work* - [Bennet Meyers GitHub](https://github.com/bmeyers)

See also the list of [contributors](https://github.com/bmeyers/solar-data-tools/contributors) who participated in this project.
