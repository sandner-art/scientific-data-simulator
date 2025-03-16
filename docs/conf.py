# docs/conf.py
import os
import sys
sys.path.insert(0, os.path.abspath('../'))  # Add project root to path

# -- Project information -----------------------------------------------------

project = 'Scientific Data Simulator'
copyright = '2025, Daniel Sandner'  # Replace with your name
author = 'Daniel Sandner'          # Replace with your name

# The full version, including alpha/beta/rc tags
release = '0.1.0'  # Update with each release


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',  # For generating API documentation from docstrings
    'sphinx.ext.napoleon', # Support for NumPy/Google style docstrings
    'sphinx.ext.viewcode', # Add links to source code from documentation
    'sphinx.ext.todo',     # Support for todo items
    'sphinx.ext.mathjax',  # Render math equations
    'sphinx.ext.intersphinx', # Link to other Sphinx documentation
    'myst_parser', # Support for Markdown (optional, if you use MyST)
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'  # Read the Docs theme (recommended)

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------

# Napoleon settings (for docstring parsing)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Intersphinx mapping (for linking to external documentation)
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'plotly': ('https://plotly.com/python-api-reference/', None),
}

# Autodoc settings
autodoc_member_order = 'bysource'  # Order members by their source order
autodoc_default_options = {
    'members': True,          # Document members (functions, classes, etc.)
    'undoc-members': True,      # Include members that don't have docstrings
    'show-inheritance': True,  # Show inheritance for classes
}

# -- Options for MyST (if you use Markdown) ----------------------------------
# myst_enable_extensions = [  # Enable MyST extensions (if needed)
#     "amsmath",
#     "colon_fence",
#     "deflist",
#     "dollarmath",
#     "fieldlist",
#     "html_admonition",
#     "html_image",
#     "replacements",
#     "smartquotes",
#     "strikethrough",
#     "substitution",
#     "tasklist",
# ]