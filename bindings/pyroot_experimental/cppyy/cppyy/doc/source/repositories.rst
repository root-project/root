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
