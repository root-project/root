.. _toplevel:


Top Level
=========

cppyy provides a couple of helper functions at the module level that provide
(direct) access to the Cling interpreter (any C++ code is always accessed
through the global namespace ``cppyy.gbl``).
The documentation makes use of these helpers throughout, so they are listed
here first, but their documentation is more conveniently accessible through
the Python interpreter itself, using the ``help()`` function::

    $ python
    >>> import cppyy
    >>> help(cppyy)


`Loading C++`
-------------

C++ code can be loaded as text to be JITed, or be compiled ahead of time and
supplied in the form of a shared library.
In the latter case, C++ headers need to be loaded as well to declare
classes, functions, and variables to Cling.
Instead of headers, pre-compiled code can be used; in particular all of the
standard C++ headers and several system headers are pre-compiled at startup.
cppyy provides the following helpers to load C++ code:

* ``cppdef``: direct access to the interpreter.
  This function accepts C++ declarations as a string and JITs them (bindings
  are not created until actual use).
  The code is loaded into the global scope, thus any previously loaded code
  is available from one ``cppdef`` call to the next, as are all standard
  C++ headers that have been loaded through pre-compiled headers.
  Example::

    >>> cppyy.cppdef(r"""\
    ... void hello() {
    ...     std::cout << "Hello, World!" << std::endl;
    ... }""")
    True
    >>> cppyy.gbl.hello()
    Hello, World!
    >>> 

* ``cppexec``: direct access to the interpreter.
  This function accepts C++ statements as a string, JITs and executes them.
  Just like ``cppdef``, execution is in the global scope and all previously
  loaded code is available.
  If the statements are declarations, the effect is the same as ``cppdef``,
  but ``cppexec`` also accepts executable lines.
  Example::

    >>> cppyy.cppexec(r"""std::string hello = "Hello, World!";""")
    True
    >>> cppyy.cppexec("std::cout << hello << std::endl;")
    Hello, World!
    True
    >>> 

* ``include``: load declarations into the interpreter.
  This function accepts C++ declarations from a file, typically a header.
  Files are located through include paths given to the Cling.
  Example::

    >>> cppyy.include("vector")   # equivalent to "#include <vector>"
    True
    >>> 

* ``c_include``: load declarations into the interpreter.
  This function accepts C++ declarations from a file, typically a header.
  Name mangling is an important difference between C and C++ code.
  The use of ``c_include`` instead of ``include`` prevents mangling.

* ``load_library``: load compiled C++ into the interpreter.
  This function takes the name of a shared library and loads it into current
  process, exposing all external symbols to Cling.
  Libraries are located through load paths given to Cling, either through the
  "-L" compiler flag or the dynamic search path environment variable (system
  dependent).
  Any method that brings symbols into the process (including normal linking,
  e.g. when embedding Python in a C++ application) is suitable to expose
  symbols.
  An alternative for ``load_library`` is for example ``ctypes.CDLL``, but
  that function does not respect dynamic load paths on all platforms.

If a compilation error occurs during JITing of C++ code in any of the above
helpers, a Python ``SyntaxError`` exception is raised.
If a compilation warning occurs, a Python warning is issued.


`Configuring Cling`
-------------------

It is often convenient to add additional search paths for Cling to find
headers and libraries when loading a module (Python does not have standard
locations to place headers and libraries, but their locations can usually
be inferred from the location of the module, i.e. it's ``__file__``
attribute).
cppyy provides the following two helpers:

* ``add_include_path``: add additional paths for Cling to look for headers.

* ``add_library_path``: add additional paths for Cling to look for libraries.

Both functions accept either a string (a single path) or a list (for adding
multiple paths).
Paths are allowed to be relative, but absolute paths are recommended.


`C++ language`
--------------

Some C++ compilation-time features have no Python equivalent.
Instead, convenience functions are provided:

* ``sizeof``: takes a proxied C++ type or its name as a string and returns
  the storage size (in units of ``char``).

* ``typeid``: takes a proxied C++ type or its name as a string and returns
  the the C++ runtime type information (RTTI).

* ``nullptr``: C++ ``NULL``.


`Preprocessor`
--------------

Preprocessor macro's (``#define``) are not available on the Python side,
because there is no type information available for them.
They are, however, often used for constant data (e.g. flags or numbers; note
that modern C++ recommends the use of ``const`` and ``constexpr`` instead).
Within limits, macro's representing constant data are accessible through the
``macro`` helper function.
Example::

    >>> import cppyy
    >>> cppyy.cppdef('#define HELLO "Hello, World!"')
    True
    >>> cppyy.macro("HELLO")
    'Hello, World!'
    >>> 

