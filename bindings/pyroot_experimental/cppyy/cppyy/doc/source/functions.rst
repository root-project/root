.. _functions:


Functions
=========

C++ functions are first-class objects in Python and can be used wherever
Python functions can be used, including for dynamically constructing
classes.

The C++ code used for the examples below can be found
:doc:`here <cppyy_features_header>`, and it is assumed that that code is
loaded at the start of any session.
Download it, save it under the name ``features.h``, and load it:

  .. code-block:: python

    >>> import cppyy
    >>> cppyy.include('features.h')
    >>>


`Free functions`
----------------

All bound C++ code starts off from the global C++ namespace, represented in
Python by ``gbl``.
This namespace, as any other namespace, is treated as a module after it has
been loaded.
Thus, we can directly import C++ functions that live underneath it.

  .. code-block:: python

    >>> from cppyy.gbl import global_function, Namespace
    >>> global_function == Namespace.global_function
    False
    >>>

C++ supports overloading, whereas Python supports "duck typing", so C++
overloads have to be selected dynamically:

  .. code-block:: python

    >>> global_function(1.)        # selects 'double' overload
    2.718281828459045
    >>> global_function(1)         # selects 'int' overload
    42
    >>>

C++ does a static dispatch at compile time based on the argument types.
The dispatch is a selection among overloads (incl. templates) visible at that
point in the translation unit.
Bound C++ in Python does a dynamic dispatch: it considers all overloads
visible _globally_ at that point in the execution.
Because the dispatch is fundamentally different (albeit in line with the
expectation of the respective languages), differences can occur.
Especially if overloads live in different header files and are only an
implicit conversion apart.

If the overload selection fails in a specific case, the ``__overload__``
function can be called directly with a signature:

  .. code-block:: python

     >>> global_function.__overload__('double')(1)   # int implicitly converted
     2.718281828459045
     >>>


`\*args and \*\*kwds`
---------------------

C++ default arguments work as expected, but python keywords are not (yet)
supported.
(It is technically possible to support keywords, but for the C++ interface,
the formal argument names have no meaning and are not considered part of the
API, hence it is not a good idea to use keywords.)
Example:

  .. code-block:: python

    >>> from cppyy.gbl import Concrete
    >>> c = Concrete()       # uses default argument
    >>> c.m_int
    42
    >>> c = Concrete(13)     # uses provided argument
    >>> c.m_int
    13
    >>> args = (27,)
    >>> c = Concrete(*args)  # argument pack
    >>> c.m_int
    27
    >>>


`Callbacks`
-----------

Python callables (functions/lambdas/instances) can be passed to C++ through
function pointers and/or ``std::function``.
This involves creation of a temporary wrapper, which has the same life time as
the Python callable it wraps, so the callable needs to be kept alive on the
Python side if the C++ side stores the callback.
Example:

  .. code-block:: python

    >>> from cppyy.gbl import call_int_int
    >>> print(call_int_int.__doc__)
    int ::call_int_int(int(*)(int,int) f, int i1, int i2)
    >>> def add(a, b):
    ...    return a+b
    ...
    >>> call_int_int(add, 3, 7)
    7
    >>> call_int_int(lambda x, y: x*y, 3, 7)
    21
    >>>

