# Contribution Guidelines

.. Important:: These documentation pages are under construction. If you would like to contribute, please contact `Bennet Meyers`_.

Contributions are welcome, and are greatly appreciated! Even small contributions are helpful, and credit will always be given!

## Types of Contributions

### Report Bugs

To report a bug, first make sure there isn't already an open `GitHub Issue`_ about it. If not, you can submit a new issue, making sure to include:

* Your OS name and version.
* Any details about your local setup (e.g., your Python environment setup) that might be helpful in troubleshooting.
* The Solar Data Tools version you are using.
* Detailed steps to reproduce the bug.
* A tag (if appropriate).

### Fix Open Issues

Look through the `GitHub Issues`_ for any reported bugs. Anything tagged with "bug" and "help
wanted" is open to contributions.

### Implement Features

Look through the `GitHub Issues`_ for any feature requests. Anything tagged with "enhancement" and "help
wanted" is open to contributions.

### Write Documentation

You can never have enough documentation! Please feel free to contribute to any
part of the documentation, such as the official docs, docstrings, or even
on the web in blog posts, articles, and such.

### Submit Feedback

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

## Get Started!

Ready to contribute? Here's how to set up `cardsort` for local development.

1. Download a copy of `cardsort` locally.
2. Install `cardsort` using `poetry`:

    ```console
    $ poetry install
    ```

3. Use `git` (or similar) to create a branch for local development and make your changes:

    ```console
    $ git checkout -b name-of-your-bugfix-or-feature
    ```

4. When you're done making changes, check that your changes conform to any code formatting requirements and pass any tests.

5. Commit your changes and open a pull request.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include additional tests if appropriate.
2. If the pull request adds functionality, the docs should be updated.
3. The pull request should work for all currently supported operating systems and versions of Python.

## Code of Conduct

Please note that the `cardsort` project is released with a
Code of Conduct. By contributing to this project you agree to abide by its terms.

## Pre-commit Hooks
Must enable pre-commit hook before pushing any contributions:

.. code-block:: bash

    pip install pre-commit
    pre-commit install


Run pre-commit hook on all files:

.. code-block:: python

    pre-commit run --all-files

.. _Bennet Meyers: mailto:bennetm@stanford.edu
.. _GitHub Issue: https://github.com/slacgismo/solar-data-tools/issues
.. _GitHub Issues: https://github.com/slacgismo/solar-data-tools/issues