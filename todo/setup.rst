Setup
-----

Recommended: Set up ``conda`` environment with provided ``.yml`` file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Updated September 2020*

We recommend seting up a fresh Python virutal environment in which to
use ``solar-data-tools``. We recommend using the
`Conda <https://docs.conda.io/projects/conda/en/latest/index.html>`__
package management system, and creating an environment with the
environment configuration file named ``pvi-user.yml``, provided in the
top level of this repository. This will install the
``statistical-clear-sky`` package as well.

Please see the Conda documentation page, "`Creating an environment from
an environment.yml
file <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file>`__\ "
for more information.

Installing this project as PIP package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sh

    $ pip install solar-data-tools

As of March 6, 2019, it fails because scs package installed as a
dependency of cxvpy expects numpy to be already installed. `scs issue
85 <https://github.com/cvxgrp/scs/issues/85>`__ says, it is fixed.
However, it doesn't seem to be reflected in its pip package. Also, cvxpy
doesn't work with numpy version less than 1.16. As a work around,
install numpy separatly first and then install this package. i.e.

.. code:: sh

    $ pip install 'numpy>=1.16'
    $ pip install statistical-clear-sky

Solvers
^^^^^^^

By default, ECOS solver is used, which is supported by cvxpy because it
is Open Source.

However, it is found that Mosek solver is more stable. Thus, we
encourage you to install it separately as below and obtain the license
on your own.

-  `mosek <https://www.mosek.com/resources/getting-started/>`__ - For
   using MOSEK solver.

   .. code:: sh

       $ pip install -f https://download.mosek.com/stable/wheel/index.html Mosek

Installing this project as Anaconda package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sh

    $ conda install -c slacgismo solar-data-tools

If you are using Anaconda, the problem described in the section for PIP
package above doesn't occur since numpy is already installed. And during
solar-data-tools installation, numpy is upgraded above 1.16.

Solvers
^^^^^^^

By default, ECOS solver is used, which is supported by cvxpy because it
is Open Source.

However, it is found that Mosek solver is more stable. Thus, we
encourage you to install it separately as below and obtain the license
on your own.

-  `mosek <https://www.mosek.com/resources/getting-started/>`__ - For
   using MOSEK solver.

   .. code:: sh

       $ conda install -c mosek mosek

Using this project by cloning this GIT repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From a fresh ``python`` environment, run the following from the base
project folder:

.. code:: bash

    $ pip install -r requirements.txt

As of March 6, 2019, it fails because scs package installed as a
dependency of cxvpy expects numpy to be already installed. `scs issue
85 <https://github.com/cvxgrp/scs/issues/85>`__ says, it is fixed.
However, it doesn't seem to be reflected in its pip package. Also, cvxpy
doesn't work with numpy version less than 1.16. As a work around,
install numpy separatly first and then install this package. i.e.

.. code:: bash

    $ pip install 'numpy>=1.16'
    $ pip install -r requirements.txt

To test that everything is working correctly, launch

.. code:: bash

    $ jupyter notebook

and run the two notebooks in the ``notebooks/`` folder.
