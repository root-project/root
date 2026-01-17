.. _history:

History
=======

.. toctree::
   :hidden:

What is now called `cppyy` started life as `RootPython` from `CERN`_, but
cppyy is not associated with CERN (it is still used there, however,
underpinning `PyROOT`_).

Back in late 2002, Pere Mato of CERN, had the idea of using the `CINT`_ C++
interpreter, which formed the interactive interface to `ROOT`_, to call from
Python into C++: this became RootPython.
This binder interfaced with Python through `boost.python`_ (v1), transpiling
Python code into C++ and interpreting the result with CINT.
In early 2003, I ported this code to boost.python v2, then recently released.
In practice, however, re-interpreting the transpiled code was unusably slow,
thus I modified the code to make direct use of CINT's internal reflection
system, gaining about 25x in performance.
I presented this work as `PyROOT` at the ROOT Users' Workshop in early 2004,
and, after removing the boost.python dependency by using the C-API directly
(gaining another factor 7 in speedup!), it was included in ROOT.
PyROOT was presented at the SciPy'06 conference, but was otherwise not
advocated outside of High Energy Physics (HEP).

In 2010, the PyPy core developers and I held a `sprint at CERN`_ to use
`Reflex`, a standalone alternative to CINT's reflection of C++, to add
automatic C++ bindings, PyROOT-style, to `PyPy`_.
This is where the name "cppyy" originated.
Coined by Carl Friedrich Bolz, if you want to understand the meaning, just
pronounce it slowly: cpp-y-y.

After the ROOT team replaced CINT with `Cling`_, PyROOT soon followed.
As part of Google's Summer of Code '16, Aditi Dutta moved PyPy/cppyy to Cling
as well, and packaged the code for use through `PyPI`_.
I continued this integration with the Python eco-system by forking PyROOT,
reducing its dependencies, and repackaging it as CPython/cppyy.
The combined result is the current cppyy project.
Mid 2018, version 1.0 was released.


.. _`CERN`: https://cern.ch/
.. _`PyROOT`: https://root.cern.ch/root/htmldoc/guides/users-guide/ROOTUsersGuide.html#python-interface
.. _`CINT`: https://en.wikipedia.org/wiki/CINT
.. _`ROOT`: https://root.cern.ch
.. _`boost.python`: https://wiki.python.org/moin/boost.python/GettingStarted
.. _`sprint at CERN`: https://morepypy.blogspot.com/2010/07/cern-sprint-report-wrapping-c-libraries.html
.. _`PyPy`: https://www.pypy.org/
.. _`Cling`: https://github.com/vgvassilev/cling
.. _`PyPI`: https://pypi.org/
