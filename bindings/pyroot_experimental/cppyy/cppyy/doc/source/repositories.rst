.. _repositories:

Repositories
============

The ``cppyy`` module is a frontend that requires an intermediate (Python
interpreter dependent) layer, and a backend (see
:ref:`Package Structure <package-structure>`).
Because of this layering and because it leverages several existing packages
through reuse, the relevant codes are contained across a number of
repositories.

* Frontend, cppyy: https://bitbucket.org/wlav/cppyy
* CPython (v2/v3) intermediate: https://bitbucket.org/wlav/cpycppyy
* PyPy intermediate (module _cppyy): https://bitbucket.org/pypy/pypy/
* Backend, cppyy: https://bitbucket.org/wlav/cppyy-backend

The backend repo contains both the cppyy-cling (under "cling") and
cppyy-backend (under "clingwrapper") packages.


Building
--------

Except for cppyy-cling, the structure in the repositories follows a normal
PyPA package and they are thus are ready to build with `setuptools`_: simply
checkout the package and either run ``python setup.py``, or use ``pip``.
It is highly recommended to follow the dependency chain when manually
upgrading packages individually (i.e. cppyy-cling, cppyy-backend, CPyCppyy
if on CPython, and then finally cppyy).

As an example, to upgrade CPyCppyy to the latest version in the repository,
do::

 $ git clone https://bitbucket.org/wlav/CPyCppyy.git
 $ pip install ./CPyCppyy --upgrade

Installation of the cppyy package works the same way (just replace "CPyCppyy"
with "cppyy").
Please see the `pip documentation`_ for more options, such as developer mode.

For the clingwrapper part of the backend (package "cppyy-backend"), which
lives in a subdirectory in the cppyy-backend repository, do::

 $ git clone https://bitbucket.org/wlav/cppyy-backend.git
 $ pip install ./cppyy-backend/clingwrapper --upgrade

Finally, the cppyy-cling package (subdirectory "cling") requires sources being
pulled in from upstream, and thus takes a few extra steps::

 $ git clone https://bitbucket.org/wlav/cppyy-backend.git
 $ cd cppyy-backend/cling
 $ python setup.py egg_info
 $ python create_src_directory.py
 $ pip install . --upgrade

The egg_info command is needed for ``create_src_directory.py`` to find the
right version.
It in turn downloads the proper release from upstream, trims and patches it,
and installs the result in the "src" directory.
When done, the structure of cppyy-cling looks again like a PyPA package and
can be used as expected.

.. _`setuptools`: https://setuptools.readthedocs.io/
.. _`pip documentation`: https://pip.pypa.io/
