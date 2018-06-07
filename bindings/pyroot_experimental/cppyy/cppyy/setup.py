#!/usr/bin/env python

import os, glob
from setuptools import setup, find_packages, Extension
from codecs import open


here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

add_pkg = ['cppyy']
try:
    import __pypy__, sys
    version = sys.pypy_version_info
    requirements = ['cppyy-backend']
    if version[0] == 5:
        if version[1] <= 9:
            requirements = ['cppyy-cling<6.12', 'cppyy-backend<0.3']
            add_pkg += ['cppyy_compat']
        elif version[1] <= 10:
            requirements = ['cppyy-cling', 'cppyy-backend<0.4']
except ImportError:
    # CPython
    requirements = ['CPyCppyy>=1.0.0']

setup(
    name='cppyy',
    version='1.0.0',
    description='Cling-based Python-C++ bindings',
    long_description=long_description,

    url='http://cppyy.readthedocs.org',

    # Author details
    author='Wim Lavrijsen',
    author_email='WLavrijsen@lbl.gov',

    license='LBNL BSD',

    classifiers=[
        'Development Status :: 5 - Production/Stable',

        'Intended Audience :: Developers',

        'Topic :: Software Development',
        'Topic :: Software Development :: Interpreters',

        'License :: OSI Approved :: BSD License',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: C',
        'Programming Language :: C++',

        'Natural Language :: English'
    ],

    install_requires=requirements,

    keywords='C++ bindings data science',

    package_dir={'': 'python'},
    packages=find_packages('python', include=add_pkg),
)
