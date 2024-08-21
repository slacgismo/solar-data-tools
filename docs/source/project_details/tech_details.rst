*****************
Technical Details
*****************

The `PVInsight Technical Report <https://www.osti.gov/biblio/1897181>`_ is an in-depth report that includes details of
the Solar Data Tools project, including detailed descriptions of the tools and the main pipeline algorithms provided by the project.

This `IEEE PVSC 2024 paper <https://ieee-pvsc.org/online/manuscripts/pvsc_52-manuscript-553-1717793929.pdf>`_ provides
details specifically on the automatic loss factor analysis of the unlabeled PV energy data
(a method of the main DataHandler class), including a description how the total energy loss was attributed to different
loss factors using Shapley values.

The estimation of the location and orientation algorithm (also a method of the DataHandler class) is described in this
`IEEE 2021 PVSC paper <https://ieeexplore.ieee.org/abstract/document/9518783>`_.

Most of the algorithms make use of a "signal decomposition" framework that was developed in parallel with this software. You can find a monograph explaining this concept `here <https://www.nowpublishers.com/article/Details/SIG-122>`__ (mirrored `here <https://web.stanford.edu/~boyd/papers/sig_decomp_mprox.html>`__) and a "no math, no code" tutorial on signal decomposition `here <https://marimo.io/@public/signal-decomposition>`__.
