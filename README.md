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
    <a href="https://solar-data-tools.readthedocs.io/en/stable/">
        <img src="https://readthedocs.org/projects/solar-data-tools/badge/?version=stable" alt="documentation build status" />
    </a>
    <a href="https://app.circleci.com/pipelines/github/slacgismo/solar-data-tools">
        <img src="https://circleci.com/gh/slacgismo/solar-data-tools.svg?style=svg" alt="CircleCi build status" />
    </a>
    <!-- switch below from tadatoshi to gismo -->
    <a href="https://travis-ci.com/tadatoshi/solar-data-tools.svg?branch=development">
        <img src="https://travis-ci.com/tadatoshi/solar-data-tools.svg?branch=development">
    </a>
  </td>
</tr>
<tr>
    <td>Code Quality</td>
    <td>
        <a href="https://lgtm.com/projects/g/slacgismo/solar-data-tools/context:python">
            <img alt="Language grade: Python" src="https://img.shields.io/lgtm/grade/python/g/slacgismo/solar-data-tools.svg?logo=lgtm&logoWidth=18"/>
        </a>
        <a href="https://lgtm.com/projects/g/slacgismo/solar-data-tools/alerts/">
            <img alt="Total alerts" src="https://img.shields.io/lgtm/alerts/g/slacgismo/solar-data-tools.svg?logo=lgtm&logoWidth=18"/>
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

There is close integration between this repository and the [Statistical Clear Sky](https://github.com/slacgismo/StatisticalClearSky) repository, which provides a "clear sky model" of system output, given only measured power as an input.

See [notebooks](/notebooks) folder for examples.

## Install & Setup

### 3 ways of setting up, either approach works:

#### 1) Recommended: Set up `conda` environment with provided `.yml` file

We recommend setting up a fresh Python virtual environment in which to use `solar-data-tools`. We recommend using the [Conda](https://docs.conda.io/projects/conda/en/latest/index.html) package management system, and creating an environment with the environment configuration file named `pvi-user.yml`, provided in the top level of this repository. This will install the `statistical-clear-sky` package as well.

Creating the env:

```bash
$ conda env create -f pvi-user.yml
```

Starting the env:

```bash
$ conda activate pvi_user
```

Stopping the env

```bash
$ conda deactivate
```

Updating the env with latest

```bash
$ conda env update -f pvi-user.yml
```

Additional documentation on setting up the Conda environment is available [here](https://github.com/slacgismo/pvinsight-onboarding/blob/main/README.md).


#### 2) PIP Package

```sh
$ pip install solar-data-tools
```

Alternative: Clone repo from GitHub

Mimic the pip package by setting up locally.

```bash
$ pip install -e path/to/root/folder
```

#### 3) Anaconda Package

```sh
$ conda install -c slacgismo solar-data-tools
```

### Solvers

#### ECOS

By default, ECOS solver is used, which is supported by cvxpy because it is Open Source.
However, it is found that Mosek solver is more stable. Thus, we encourage you to install it separately as below and obtain the license on your own.

#### MOSEK

 MOSEK is a commercial software package. The included YAML file will install MOSEK for you, but you will still need to obtain a license. More information is available here:

* [mosek](https://www.mosek.com/resources/getting-started/)
* [Free 30-day trial](https://www.mosek.com/products/trial/)
* [Personal academic license](https://www.mosek.com/products/academic-licenses/)

## Usage

Users will primarily interact with this software through the `DataHandler` class.

```python
from solardatatools import DataHandler
from solardatatools.dataio import get_pvdaq_data

pv_system_data = get_pvdaq_data(sysid=35, api_key='DEMO_KEY', year=[2011, 2012, 2013])

dh = DataHandler(pv_system_data)
dh.run_pipeline(power_col='dc_power')
```
If everything is working correctly, you should see something like the following

```
total time: 16.67 seconds
--------------------------------
Breakdown
--------------------------------
Preprocessing              6.52s
Cleaning                   8.62s
Filtering/Summarizing      1.53s
    Data quality           0.23s
    Clear day detect       0.19s
    Clipping detect        0.21s
    Capacity change detect 0.91s
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

## License

This project is licensed under the BSD 2-Clause License - see the [LICENSE](LICENSE) file for details
