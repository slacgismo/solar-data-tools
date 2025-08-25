# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import subprocess
import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "Solar Data Tools"
copyright = "%s, Bennet Meyers" % date.today().year
author = "Bennet Meyers"
# Get version from git tags
# (so this will always show latest tagged version and not local/dev version)
# Get all the git tags from the cmd line that follow our versioning pattern
git_tags = subprocess.Popen(
    ["git", "tag", "--list", "v*[0-9]", "--sort=version:refname"],
    stdout=subprocess.PIPE,
)
tags = git_tags.stdout.read()
git_tags.stdout.close()
tags = tags.decode("utf-8").split("\n")
tags.sort()
release = tags[-1][1:]

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "nbsphinx",  # for notebooks integration
    "nbsphinx_link",  # for linking to notebooks from docs source
]

autosummary_generate = True

source_suffix = [".rst", ".md"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "_templates", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_title = f"{project} v{release}"
html_static_path = ["_static"]
html_css_files = ["solardatatools.css"]
html_copy_source = False
html_favicon = "_static/SDT_v1_icon_only_dark_background_small.ico"
# html_logo = "_static/SDT_v1_icon_only_dark_background_small.png"
html_short_title = f"{project}"

master_doc = "index"

# Below is not working (having a different icon in header for light and dark modes
# without making the browser tab say "<no title>"
# and seems there is no way to assign an html title separately from that header
# todo: check a way to adjust this
# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_theme_options = {
    "logo": {
        # In a left-to-right context, screen readers will read the alt text
        # first, then the text, so this example will be read as "P-G-G-P-Y
        # (short pause) Home A pretty good geometry package"
        "alt_text": f"{project}",
        "text": f"{project}",
        "image_light": "_static/SDT_v1_icon_only_light_background_small.png",
        "image_dark": "_static/SDT_v1_icon_only_dark_background_small.png",
    },
    "external_links": [],
    "navbar_align": "left",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_version_warning_banner": True,
    "github_url": "https://github.com/slacgismo/solar-data-tools",
    "show_toc_level": 1,
    "footer_start": ["copyright", "sphinx-version"],
    "navigation_with_keys": False,
}
