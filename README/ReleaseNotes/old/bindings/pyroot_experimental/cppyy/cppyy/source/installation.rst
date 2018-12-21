.. _installation:

Installation
============

The ``cppyy`` module and its dependencies are available through `PyPI`_ for
both CPython (2 and 3) and PyPy (5.9.0 and later).
Build-time only dependencies are ``cmake`` (for general build), ``python2.7``
(for LLVM), and a modern C++ compiler (one that supports at least C++11).
The cleanest/easiest way to install cppyy is using `virtualenv`_.

Compilation of the backend, which contains a customized version of
Clang/LLVM, can take a long time, so by default the setup script will use all
cores (x2 if hyperthreading is enabled).
To change that behavior, set the MAKE_NPROCS environment variable to the
desired number of processes to use.
To see progress while waiting, use ``--verbose``::

 $ MAKE_NPROCS=32 pip install --verbose cppyy

The bdist_wheel of the backend is reused by pip for all versions of CPython
and PyPy, thus the long compilation is needed only once.
Prepared wheels of cppyy-cling (which contains LLVM) for Mac 10.12 and
Linux/Gentoo `are available`_.
To use them, tell ``pip``::

 $ pip install --extra-index https://cern.ch/wlav/wheels cppyy

If you use the ``--user`` option to pip, make sure that the PATH envar points
to the bin directory that will contain the installed entry points during the
installation, as the build process needs them.
You may also need to install wheel first.
Example::

 $ pip install wheel --user
 $ PATH=$HOME/.local/bin:$PATH pip install cppyy --user

PyPy 5.7 and 5.8 have a built-in module ``cppyy``.
You can still install the ``cppyy`` package, but the built-in module takes
precedence.
To use ``cppyy``, first import a compatibility module::

 $ pypy
 [PyPy 5.8.0 with GCC 5.4.0] on linux2
 >>>> import cppyy_compat, cppyy
 >>>>

You will have to set ``LD_LIBRARY_PATH`` appropriately if you get an
``EnvironmentError`` (it will indicate the needed directory).

Note that your python interpreter (whether CPython or ``pypy-c``) may not have
been linked by the C++ compiler.
This can lead to problems during loading of C++ libraries and program shutdown.
In that case, re-linking is highly recommended.

Older versions of PyPy (5.6.0 and earlier) have a built-in ``cppyy`` based on
`Reflex`_, which is less feature-rich and no longer supported.
However, both the :doc:`distribution tools <dictionaries>` and user-facing
Python codes are very backwards compatible.

.. _`PyPI`: https://pypi.python.org/pypi/cppyy/
.. _`virtualenv`: https://pypi.python.org/pypi/virtualenv
.. _`are available`: https://cern.ch/wlav/wheels/
.. _`Reflex`: https://root.cern.ch/how/how-use-reflex
