.. _strings:


Strings/Unicode
===============

Both Python and C++ have core types to represent text and these are expected
to be freely interchangeable.
``cppyy`` makes it easy to do just that for the most common cases, while
allowing customization where necessary to cover the full range of diverse use
cases (such as different codecs).
In addition to these core types, there is a range of other character types,
from ``const char*`` and ``std::wstring`` to ``bytes``, that see much less
use, but are also fully supported.


`std::string`
"""""""""""""

The C++ core type ``std::string`` is considered the equivalent of Python's
``str``, even as purely implementation-wise, it is more akin to ``bytes``:
as a practical matter, a C++ programmer would use ``std::string`` where a
Python developer would use ``str`` (and vice versa), not ``bytes``.

A Python ``str`` is unicode, however, whereas an ``std::string`` is character
based, thus conversions require encoding or decoding.
To allow for different encodings, ``cppyy`` defers implicit conversions
between the two types until forced, at which point it will default to seeing
``std::string`` as ASCII based and ``str`` to use the UTF-8 codec.
To support this, the bound ``std::string`` has been pythonized to allow it to
be a drop-in for a range of uses as appropriate within the local context.

In particular, it is sometimes necessary (e.g. for function arguments that
take a non-const reference or a pointer to non-const ``std::string``
variables), to use an actual ``std::string`` instance to allow in-place
modifications.
The pythonizations then allow their use where ``str`` is expected.
For example:

  .. code-block:: python

    >>> cppyy.cppexec("std::string gs;")
    True
    >>> cppyy.gbl.gs = "hello"
    >>> type(cppyy.gbl.gs)   # C++ std::string type
    <class cppyy.gbl.std.string at 0x7fbb02a89880>
    >>> d = {"hello": 42}    # dict filled with str
    >>> d[cppyy.gbl.gs]      # drop-in use of std::string -> str
    42
    >>>

To handle codecs other than UTF-8, the ``std::string`` pythonization adds a
``decode`` method, with the same signature as the equivalent method of
``bytes``.
If it is known that a specific C++ function always returns an ``std::string``
representing unicode with a codec other than UTF-8, it can in turn be
explicitly pythonized to do the conversion with that codec.


`std::string_view`
""""""""""""""""""

It is possible to construct a (char-based) ``std::string_view`` from a Python
``str``, but it requires the unicode object to be encoded and by default,
UTF-8 is chosen.
This will give the expected result if all characters in the ``str`` are from
the ASCII set, but otherwise it is recommend to encode on the Python side and
pass the resulting ``bytes`` object instead.


`std::wstring`
""""""""""""""

C++'s "wide" string, ``std::wstring``, is based on ``wchar_t``, a character
type that is not particularly portable as it can be 2 or 4 bytes in size,
depending on the platform.
cppyy supports ``std::wstring`` directly, using the ``wchar_t`` array
conversions provided by Python's C-API.


`const char*`
"""""""""""""

The C representation of text, ``const char*``, is problematic for two
reasons: it does not express ownership; and its length is implicit, namely up
to the first occurrence of ``'\0'``.
The first can, up to an extent, be ameliorated: there are a range of cases
where ownership can be inferred.
In particular, if the C string is set from a Python ``str``, it is the latter
that owns the memory and the bound proxy of the former that in turn owns the
(unconverted) ``str`` instance.
However, if the ``const char*``'s memory is allocated in C/C++, memory
management is by necessity fully manual.
Length, on the other hand, can only be known in the case of a fixed array.
However even then, the more common case is to use the fixed array as a
buffer, with the actual string still only extending up to the ``'\0'`` char,
so that is assumed.
(C++'s ``std::string`` suffers from none of these issues and should always be
preferred when you have a choice.)


`char*`
"""""""

The C representation of a character array, ``char*``, has all the problems of
``const char*``, but in addition is often used as "data array of 8-bit int".


`character types`
"""""""""""""""""

cppyy directly supports the following character types, both as single
variables and in array form: ``char``, ``signed char``, ``unsigned char``,
``wchar_t``, ``char16_t``, and ``char32_t``.

