##Instructions to run locally:

```sh
conda env create -f environment.yml
conda activate venv_rdt
```

Navigate to docs/source and run

```sh
$ make html
$ open _build/html/index.html
```

##Updating the documentation:

All the documentation files live in docs/source. Index.rst is the landing page. Under contents is a list of all the sections to this documentation which are linked to their own rst files.

In order to add a new section, first update the index.rst to have the name of the section under contents. Then create a "section_name".rst file making sure that the "section_name" matches the one in index.rst. If it makes it easier, https://cloudconvert.com/md-to-rst, allows to convert mark down files to rst format.


##Changes merged to master auto update on [solar-data-tools docs](https://solar-data-tool.readthedocs.io/en/latest/) via a web hook


##Link for [getting started with sphinx](https://docs.readthedocs.io/en/stable/intro/getting-started-with-sphinx.html)

Credential for read the docs can be found in gismo team drive under PV-Insight.


##Auto generate API Documentation Based on Python Files in solardatatools directory
force deletes existing to update with new
```sh
$ sphinx-apidoc -o ./source ../solardatatools -f
```
