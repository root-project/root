.. -*- mode: rst -*-

cppyy: Python-C++ bindings interface based on Cling/LLVM
========================================================

cppyy provides fully automatic, dynamic Python-C++ bindings by leveraging
the Cling C++ interpreter and LLVM.

These bindings work both ways, so you can call C++ code from Python and the other way around.

You need no boilerplate code, and you can access Python APIs from Cpp, and the other way around.

It supports both PyPy (natively), CPython, and C++ language standards
through C++20 (and parts of C++23).

Details and performance are described in
`this paper <http://cern.ch/wlav/Cppyy_LavrijsenDutta_PyHPC16.pdf>`_,
originally presented at PyHPC'16, but since updated with improved performance
numbers.

Full documentation: `cppyy.readthedocs.io <http://cppyy.readthedocs.io/>`_.

Notebook-based tutorial: `Cppyy Tutorial <https://github.com/wlav/cppyy/blob/master/doc/tutorial/CppyyTutorial.ipynb>`_.

For Anaconda/miniconda, install cppyy from `conda-forge <https://anaconda.org/conda-forge/cppyy>`_.

----

Change log:
  https://cppyy.readthedocs.io/en/latest/changelog.html

Bug reports/feedback:
  https://github.com/wlav/cppyy/issues
