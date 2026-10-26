# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import re

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'satlib'
copyright = '2026, Kevin Anderson, Mandy Lewis'
author = 'Kevin Anderson, Mandy Lewis'

import satlib
release = satlib.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.extlinks',
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
templates_path = ['_templates']

html_theme_options = {
    "github_url": "https://github.com/kandersolar/satlib",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/satlib/",
            "icon": "fab fa-python",
        },
    ],
    "show_toc_level": 1,
    # "footer_start": [],  # "copyright", "sphinx-version"
    # "footer_center": [],
    "footer_end": [],
    # "primary_sidebar_end": [],
    # https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/layout.html#hide-the-previous-and-next-buttons
    "show_prev_next": False,  # disable next/previous links
}  # noqa: E501

extlinks = {
    "issue": ("https://github.com/kandersolar/satlib/issues/%s", "GH%s"),
    "pull": ("https://github.com/kandersolar/satlib/pull/%s", "GH%s"),
    "doi": ("http://dx.doi.org/%s", "DOI: %s"),
    "ghuser": ("https://github.com/%s", "@%s"),
}

os_re_sep = re.escape(os.path.sep)
sphinx_gallery_conf = {
    'examples_dirs': ['gallery'],  # location of gallery scripts
    'gallery_dirs': ['examples'],  # location of generated output

    'filename_pattern': '(.*)',  # everything

    # directory where function/class granular galleries are stored
    'backreferences_dir': 'reference/gallery_backreferences',

    # Modules for which function/class level galleries are created. In
    # this case only satlib, could include others though. Must be tuple of str
    'doc_module': ('satlib',),

    # https://sphinx-gallery.github.io/dev/configuration.html#removing-config-comments  # noqa: E501
    'remove_config_comments': True,
}
