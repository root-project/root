import os
import subprocess

from setuptools import setup, find_packages, Extension
import numpy.distutils.misc_util

ROOTBIN = "/scratch/pivarski/root-install/bin"

def rootconfig(arg, filter, drop):
    rootconfig = subprocess.Popen([os.path.join(ROOTBIN, "root-config"), arg], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if rootconfig.wait() != 0:
        raise IOError(rootconfig.stderr.read())
    return [x.strip()[drop:] for x in rootconfig.stdout.read().decode().split(" ") if filter(x)]

setup(name = "numpyinterface",
      version = "0.0.1",
      packages = find_packages(),
      scripts = [],
      description = "",
      long_description = """""",
      author = "Jim Pivarski (DIANA-HEP)",
      author_email = "pivarski@fnal.gov",
      maintainer = "Jim Pivarski (DIANA-HEP)",
      maintainer_email = "pivarski@fnal.gov",
      url = "",
      download_url = "",
      license = "???",
      # test_suite = "tests",
      install_requires = ["numpy"],
      tests_require = [],
      classifiers = ["Development Status :: 2 - Pre-Alpha",
                     "Environment :: Console",
                     "Intended Audience :: Science/Research",
                     "License :: OSI Approved :: Apache Software License",
                     "Topic :: Scientific/Engineering :: Information Analysis",
                     "Topic :: Scientific/Engineering :: Mathematics",
                     "Topic :: Scientific/Engineering :: Physics",
                     ],
      platforms = "Any",
      ext_modules = [Extension("numpyinterface",
                               [os.path.join("numpyinterface.cxx")],
                               include_dirs = rootconfig("--cflags", lambda x: x.startswith("-I"), 2) + numpy.distutils.misc_util.get_numpy_include_dirs(),
                               library_dirs = rootconfig("--libdir", lambda x: True, 0),
                               libraries = rootconfig("--libs", lambda x: x.startswith("-l"), 2),
                               extra_compile_args = rootconfig("--cflags", lambda x: not x.startswith("-I"), 0),
                               )],
      )
