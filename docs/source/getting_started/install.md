# Install & Setup

## Recommended: Install with pip

In a fresh Python virtual environment, simply run:

```bash
$ pip install solar-data-tools
```

or if you would like to use MOSEK, install the optional dependency as well:

```bash
$ pip install "solar-data-tools[mosek]"
```

## Install with conda

```{warning}
`solar-data-tools` is now available on conda-forge! You can specify
the channel using the `-c` flag as shown in the examples below.
The use of the slacgismo channel is **deprecated** and packages
on that channel will **not** be up-to-date with the latest releases.
```

Creating the environment and directly installing the package and its dependencies from the appropriate conda channels:

```bash
$ conda create -n pvi-user solar-data-tools -c conda-forge

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
$ conda install solar-data-tools -c conda-forge
```

# Solvers

## CLARABEL

By default, the CLARABEL solver is used to solve the signal decomposition problems. CLARABEL (as well as other solvers) is compatible with [OSD](https://github.com/cvxgrp/signal-decomposition/tree/main), the modeling language used to solve signal decomposition problems in Solar Data Tools, and both are open source.

## MOSEK

MOSEK is a commercial software package. Since it is more stable and offers faster solve times,
we provide continuing support for it (with signal decomposition problem formulations using CVXPY). However,
you will still need to obtain a license. If installing with pip, you can install the optional MOSEK dependency by running
`pip install "solar-data-tools[mosek]"`.
If installing from conda, you will have to manually install MOSEK if you desire to use it as
conda does not support optional dependencies like pip.

More information about MOSEK and how to obtain a license is available here:

* [mosek](https://www.mosek.com/resources/getting-started/)
* [Free 30-day trial](https://www.mosek.com/products/trial/)
* [Personal academic license](https://www.mosek.com/products/academic-licenses/)
