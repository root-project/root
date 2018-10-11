#!/usr/bin/env python

import os, glob, subprocess
from setuptools import setup, find_packages, Extension
from distutils.command.build_ext import build_ext as _build_ext
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
from codecs import open


here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

try:
    root_install = os.environ["ROOTSYS"]
except KeyError:
    root_install = None

def get_cflags():
    config_exec = 'cling-config'
    if root_install:
        config_exec = 'root-config'
    cli_arg = subprocess.check_output([config_exec, '--auxcflags'])
    return cli_arg.decode("utf-8").strip()

class my_build_extension(_build_ext):
    def build_extension(self, ext):
        ext.extra_compile_args = ['-O2']+get_cflags().split()
        return _build_ext.build_extension(self, ext)

class my_bdist_wheel(_bdist_wheel):
    def run(self, *args):
     # wheels do not respect dependencies; make this a no-op so that it fails (mostly) silently
        pass

setup(
    name='CPyCppyy',
    version='1.0.1',
    description='Cling-based Python-C++ bindings for CPython',
    long_description=long_description,

    url='http://cppyy.readthedocs.io/',

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
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: C',
        'Programming Language :: C++',

        'Natural Language :: English'
    ],

    install_requires=['cppyy-backend>=0.5'],

    keywords='C++ bindings data science',

    cmdclass = {
        'build_ext': my_build_extension,
        'bdist_wheel': my_bdist_wheel
    },

    ext_modules=[Extension('libcppyy',
        sources=glob.glob('src/*.cxx'),
        include_dirs=['include'])],
)
