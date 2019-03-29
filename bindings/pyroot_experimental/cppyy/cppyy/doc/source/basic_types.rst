.. _basic_types:


Basic Types
===========

C++ has a far richer set of builtin types than Python.
Most Python code can remain relatively agnostic to that, and ``cppyy``
provides automatic conversions as appropriate.
On the other hand, Python builtin types such as lists and maps are far
richer than any builtin types in C++.
These are mapped to their Standard Template Library equivalents instead.

The C++ code used for the examples below can be found
:doc:`here <cppyy_features_header>`, and it is assumed that that code is
loaded at the start of any session.
Download it, save it under the name ``features.h``, and load it:

  .. code-block:: python

    >>> import cppyy
    >>> cppyy.include('features.h')
    >>>


`Builtins`
""""""""""

Most builtin data types map onto the expected equivalent Python types, with
the caveats that there may be size differences, different precision or
rounding.
For example, a C++ ``float`` is returned as a Python ``float``, which is in
fact a C++ ``double``.
If sizes allow, conversions are automatic.
For example, a C++ ``unsigned int`` becomes a Python ``long``, but
unsigned-ness is still honored:

  .. code-block:: python

    >>> type(cppyy.gbl.gUint)
    <type 'long'>
    >>> cppyy.gbl.gUint = -1
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ValueError: cannot convert negative integer to unsigned
    >>>


`Arrays`
""""""""

Builtin arrays are supported through arrays from module ``array`` (or any
other builtin-type array that implements the Python buffer interface, such
as numpy arrays) and a low-level view type from ``cppyy`` for returns and
variable access (that implements the buffer interface as well).
Out-of-bounds checking is limited to those cases where the size is known at
compile time.
Example:

  .. code-block:: python

    >>> from cppyy.gbl import Concrete
    >>> from array import array
    >>> c = Concrete()
    >>> c.array_method(array('d', [1., 2., 3., 4.]), 4)
    1 2 3 4
    >>> c.m_data[4] # static size is 4, so out of bounds
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    IndexError: buffer index out of range
    >>>


`Pointers`
""""""""""

When the C++ code takes a pointer or reference type to a specific builtin
type (such as an ``unsigned int`` for example), then types need to match
exactly.
``cppyy`` supports the types provided by the standard modules ``ctypes`` and
``array`` for those cases.
Example of using a reference to builtin:

  .. code-block:: python

    >>> from ctypes import c_uint
    >>> u = c_uint(0)
    >>> c.uint_ref_assign(u, 42)
    >>> u.value
    42
    >>>

For objects, a pointer to an object and an object are represented the same
way, with the necessary (de)referencing applied automatically.
Pointer variables are also bound by reference, so that updates on either the
C++ or Python side are reflected on the other side as well.


`Enums`
"""""""

Both named and anonymous enums are supported.
The type of an enum is implementation dependent and may even be different for
different enums on the same compiler.
Typically, however, the types are ``int`` or ``unsigned int``, which
translates to Python's ``int`` or ``long`` on Python2 or class ``int`` on
Python3:

  .. code-block:: python

    >>> from cppyy.gbl import kApple, kBanana, kCitrus
    >>> cppyy.gbl.kApple
    78
    >>>
