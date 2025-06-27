---
title: 'Solar Data Tools: a Python library for automated analysis of unlabeled PV data'
tags:
  - Python
  - photovoltaics
  - solar power
  - signal decomposition
  - convex optimization
authors: # TODO: generate full list of authors
  - name: Sara A. Miskovich
    orcid: 0000-0002-3302-838X
    corresponding: true
    affiliation: 1
  - name: Bennet E. Meyers
    orcid: 0000-0002-4089-0744
    corresponding: true
    affiliation: 1
  - name: "Elpiniki Apostolaki-Iosifidou"
    affiliation: 1
  - name: "Claire Berschauer"
    affiliation: 1
  - name: "Chengcheng Ding"
    affiliation: 3
  - name: "Aramis Dufour"
    affiliation: 2
  - name: "David Jose Florez Rodriguez"
    affiliation: 3
  - name: "Jonathan Goncalves"
    affiliation: 1
  - name: "Alejandro Londono-Hurtado"
    affiliation: 1
  - name: "Victor-Haoyang Lian"
    affiliation: 3
  - name: "Tristan Lin"
    affiliation: 3
  - name: "Junlin Luo"
    affiliation: 3
  - name: "Xiao Ming"
    affiliation: 3
  - name: "Duncan Ragsdale"
    affiliation: 1
  - name: "Derin Serbetcioglu"
    affiliation: 1
  - name: "Shixian Sheng"
    affiliation: 3
  - name: "Jose St Louis"
    affiliation: 3
  - name: "Tadatoshi Takahashi"
    affiliation: 1
  - name: "Nimish Telang"
    affiliation: 4
  - name: "Mitchell Victoriano"
    affiliation: 1
  - name: "Haoxi Zhang"
    affiliation: 3
  - name: "Nimish Yadav"
    affiliation: 3

affiliations:
  - name: SLAC National Accelerator Laboratory, Menlo Park, CA, 94025, USA
    index: 1
  - name: Stanford University, Stanford, CA, 94305, USA
    index: 2
  - name: Carnegie Mellon University, Pittsburgh, PA 15213, USA
    index: 3
  - name: Independent Researcher, USA
    index: 4
date: 19 August 2024
bibliography: paper.bib
---

# Summary

[//]: # (high level summary, for non expert audience)

Effectively processing and leveraging the growing volume of photovoltaic (PV) system performance data is essential for
the operation and maintenance of PV systems globally. However, many distributed rooftop PV systems suffer from lower
data quality, are difficult to model, and lack access to reliable environmental data.

Solar Data Tools is an open-source Python library designed for automated data quality and loss factor
analysis of _unlabeled_ PV time-series data, i.e. without requiring a system model, meteorological
data, or performance indices. Solar Data Tools empowers PV system operators and fleet
owners to better understand their system's performance using only basic power output data.

Solar Data Tools is user-friendly, requiring minimal setup, and is compatible with all types of PV systems, from small
rooftop generators to large utility power plants. Using
advanced signal decomposition techniques [@Meyers2023], the library enables the performance and reliability analysis of large
volumes of PV power time-series data across various formats and quality levels. It eliminates the need for site-specific
meteorological inputs or pre-defined system models, simplifying the analysis process.  This library can be valuable
for a wide range of users that work with unlabeled solar power data, including
professionals in the private solar industry or utility companies, researchers and students in the solar energy field,
community solar owners, and rooftop PV system owners.

Solar Data Tools is developed openly on GitHub [@sdt-repo] under the permissive BSD-2 license and is available to
install via the Python Package Index (PyPI) [@pypi] and the conda-forge repository [@conda]. The library is actively
maintained and contributions
from the community are welcome. The documentation is hosted on Read the Docs [@sdt-docs]. More detailed information
about Solar Data Tools, its algorithms and
its features can be found in the PVSC 2020 [@pvsc2020] and PVSC 2024 [@pvsc2024] papers.

# Statement of need

With the growing number of photovoltaic (PV) systems worldwide,
it is crucial to have tools that can process and analyze data
from systems of all sizes and configurations. The data
typically consists of time-series measurements of real
power production, reported as average power over intervals ranging from
one minute to one hour, spanning several years and possibly
containing missing entries.

Historically, PV data analysis tools have focused on data combined
with local meteorological measurements and system configuration
information, such as those from large power plants. Data cleaning tasks
have largely been manual, and analyses have relied on metrics like the
performance index [@Townsend1994], which require accurate site models and
meteorological data. For smaller, distributed rooftop PV systems,
meteorological information is often lacking, making accurate system
modeling difficult. For such systems, insights must be derived from just
the PV power data in isolation (referred to as \emph{unlabeled} data),
for which forming a performance index is difficult or impossible. Given
that distributed rooftop PV systems accounted for over 40\% of the
installed capacity in 2020 [@SEIA2021], there is a clear need for
automated and model-free data processing and analysis tools that
enable remote monitoring of system health and optimization of operations
and maintenance activities of these systems.

[//]: #
Solar Data Tools (SDT) [@pvsc2020; @sdt-zenodo] is an open-source Python
library designed for the automatic processing and analysis of unlabeled PV
data signals. SDT automates the cleaning, filtering, and analysis of PV power
data, including loss factor estimation, eliminating the need for user
configuration or "babysitting" regardless of data quality or system
configuration. It is suitable for a wide range of applications and system types, from large
utility-scale trackers to small, multi-pitch rooftops. SDT provides practical
tools for both small and fleet-scale PV performance analyses without
requiring the calculation of performance indices for each system. It may be used to fully automate a
quality and loss factor estimation pipeline, or it may be used to onboard, visualize, and explore new
data or prepare data for a custom analysis.

The software has been used by researchers at Stanford University [@Ogut2024], Case Western Reserve University [@Pierce2024],
and LBNL [@Li2023; @Li2024] to prepare data in the development data-driven performance models for the solar PV domain.
In addition, the software has been used in loss factor estimation intercomparison studies by NREL [@osti_1990039] and the IEA [@Lindig2021].

Two other libraries offer similar data analysis tools for solar applications:
PVAnalytics [@pvanalytics] and RdTools [@rdtools-zenodo]. Unlike SDT,
these libraries are model-driven and require users to define their own analyses.
PVAnalytics focuses on preprocessing and quality assurance, while RdTools specializes in
loss factor analysis. SDT, on the other hand, provides both data quality and
loss factor analysis, operates _automatically_ with minimal setup, and is
**model-free**, requiring no weather or other external information. To our knowledge, no other open-source software
provides flexible model-free automated analysis for unlabeled PV data.
SDT is particularly suited for users who need a pre-defined pipeline to analyze
complex systems that cannot be easily modeled and lack meteorological data—a
common scenario for small, distributed systems.



# Acknowledgements

This work is supported by the U.S. Department of Energy’s Office of
Energy Efficiency and Renewable Energy (EERE) under the Solar Energy
Technologies Office Award Numbers 34368 and 38529.

# References
