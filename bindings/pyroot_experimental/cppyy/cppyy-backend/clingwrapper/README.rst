cppyy-backend
=============

C/C++ wrapper around Cling for use by cppyy.

This package will pull in cppyy-cling, which contains a version of LLVM
that is patched for interactive use.

Compilation of LLVM may take a long time, so when building from source, it is
recommended to set MAKE_NPROCS to the number of cores on your machine and to
use the verbose flag to see progress:

  $ MAKE_NPROCS=32 pip install --verbose cppyy-backend

Alternatively, there are binary wheels (Mac 10.12, Linux/Gentoo)
available here:
  https://cern.ch/wlav/wheels

Use '--extra-index https://cern.ch/wlav/wheels' as an argument to pip to
pick them up.

Cling documentation is here:
  https://root.cern.ch/cling

----

Find the cppyy documentation here:
  http://cppyy.readthedocs.io
