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
enables automated analysis of _unlabeled_ PV data, i.e. with no model, 
no meteorological data, and no performance index required, by taking a 
statistical signal processing approach in the algorithms used in 
the library’s main data processing pipeline.

Solar Data Tools provides
methods for data I/O, cleaning, filtering, plotting, and data 
quality and loss analysis. These methods are largely automated and 
require little to no input from the user regardless of system 
type.  This library is for anyone dealing with photovoltaic data, 
especially data with no meteorological information (unlabeled). This 
includes photovoltaic professionals (in private solar industry or 
utility companies for example), researchers and students in the 
solar power domain, community solar owners, and anyone with a 
rooftop system. The scientific goal of the library is to empower 
PV system fleet owners or operators to analyze system performance, even 
when they only have access to the most basic data stream—power 
output of the system.

# Statement of need

There are two other libraries that are similar in that they offer data 
analysis tools for solar applications: [PVAnalytics](https://github.
com/pvlib/pvanalytics) and [RdTools](https://github.com/NREL/rdtools)
. They are both model driven, and require the user to define their 
own analysis. PVAnalytics focuses on preprocessing and QA, while 
RdTools focuses on loss factor analysis. Solar Data Tools provides 
both data quality and loss factor analysis, runs _automatically_ with little to no setup, and is **model-free** and does not require any weather or other information. Solar Data Tools is most suited for when users want a pre-defined pipeline to get information on complex systems/sites that can't be modeled easily and that no meteorological data. A recent tutorial that was part of a virtual tutorial series on open-source tools and open-access solar data held by DOE’s Solar Technology Office in March 2024 goes over the differences in these libraries and when each tool is appropriate to use. You can find the recording [here](https://www.youtube.com/watch?v=XKbqIlAEwOQ) and the slide deck [here](https://www.energy.gov/sites/default/files/2024-05/Data_Bounty_webinar_part_2.pdf) (see slide 16 for a summary).

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

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