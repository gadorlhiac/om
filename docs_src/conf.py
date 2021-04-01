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
import os
import sys
sys.path.insert(0, '../src')


# -- Project information -----------------------------------------------------

project = 'om'
copyright = '2021, Author'
author = 'Author'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_click.ext"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Extension configuration -------------------------------------------------
add_module_names = False
autodoc_mock_imports = [
    "hidra_api",
    "mpi4py",
    "om.lib.peakfinder8_extension.peakfinder8_extension",
    "psana",
]
autodoc_member_order = "bysource"
autodoc_inherit_doctrings = True
autodoc_typehints = "none"
autoclass_content = "init"
napoleon_use_param = True


# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Theme configuration -----------------------------------------------------
html_logo = "_static/OMLogo.png"


# ----------------------------------------------------------------------------
#def process_module_docstring(app, what, name, obj, options, lines):
#    if what == "module":
#        prefix = [
#            "*Path:* {}".format(obj.__name__),
#            "",
#            "**{}**".format(lines[0][:-1]),
#            "",
#        ]
#        lines.remove(lines[0])
#        for line in reversed(prefix):
#            lines.insert(0, line)
#    elif what != "attribute":
#        line = "**{}**".format(lines[0][:-1])
#        lines.remove(lines[0])
#        lines.insert(0, line)
#
#def setup(app):
#    app.connect("autodoc-process-docstring", process_module_docstring)
