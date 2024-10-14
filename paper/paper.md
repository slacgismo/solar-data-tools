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
  - name: Bennet Meyers
    orcid: 0000-0002-4089-0744
    corresponding: true
    affiliation: 1
affiliations:
 - name: SLAC National Accelerator Laboratory, Menlo Park, CA, 94025, USA
   index: 1
date: 19 August 2024
bibliography: paper.bib
---

# Summary
[//]: # (high level summary, for non expert audience)

Solar Data Tools is an open-source Python library for analyzing 
photovoltaic (PV) power (and irradiance) time-series data. It 
enables automated analysis of _unlabeled_ PV data, meaning no 
model, meteorological data, or performance index is required. The
library uses signal decomposition techniques in its main data 
processing pipeline.

Solar Data Tools offers
methods for data I/O, cleaning, filtering, plotting, and data 
quality and loss analysis. These methods are largely automated,
requiring minimal user input regardless of the PV system 
type.  This library is ideal for anyone working with PV data, 
especially when meteorological information is unavailable. This 
includes photovoltaic professionals in the private solar industry or 
utility companies, researchers and students in the 
solar power domain, community solar owners, and
rooftop system owners. The primary goal of the library is to enable 
PV system fleet owners or operators to analyze system 
performance using only the basic power output data.

# Statement of need

With the growing number
of real-world installations of photovoltaic (PV) systems worldwide, 
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
performance index [@townsned], which require accurate site models and 
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

[//]: # (Cite dask IEEE short paper too? I don't mention cloud 
deploymetn at all here)
Solar Data Tools (SDT) [@Meyers2020b; @sdt-zenodo] is an open-source Python
library designed for the automatic processing and analysis of unlabeled PV
data signals. SDT automates the cleaning, filtering, and analysis of PV power
data, including loss factor estimation, eliminating the need for user
configuration or "babysitting" regardless of data quality or system
configuration. It is suitable for a wide range of systems, from large
utility-scale trackers to small, multi-pitch rooftops. SDT provides practical
tools for both small and fleet-scale PV performance analyses without
requiring the calculation of performance indices for each system.

Two other libraries offer similar data analysis tools for solar applications:
PVAnalytics [@pvanalytics] and RdTools [@rdTools-zenodo]. Unlike SDT, 
these libraries are model-driven and require users to define their own analyses. 
PVAnalytics focuses on preprocessing and quality assurance, while RdTools specializes in
loss factor analysis. SDT, on the other hand, provides both data quality and
loss factor analysis, operates _automatically_ with minimal setup, and is
**model-free**, requiring no weather or other external information. SDT is
particularly suited for users who need a pre-defined pipeline to analyze
complex systems that cannot be easily modeled and lack meteorological data—a
common scenario for small, distributed systems.
(cite tutorial here for more info?)

# Figures

Add example plots (heatmaps, loss analysis, what else?)

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

This work is supported by the U.S. Department of Energy’s Office of 
Energy Efficiency and Renewable Energy (EERE) under the Solar Energy 
Technologies Office Award Number 38529.

# References