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

Function argument type conversions follow the expected rules, with implicit
conversions allowed, including between Python builtin types and STL types,
but it is rather more efficient to make conversions explicit.


`Free functions`
----------------

All bound C++ code starts off from the global C++ namespace, represented in
Python by ``gbl``.
This namespace, as any other namespace, is treated as a module after it has
been loaded.
Thus, we can directly import C++ functions from it and other namespaces that
themselves may contain more functions.
All lookups on namespaces are done lazily, thus if loading more headers bring
in more functions (incl. new overloads), these become available dynamically.

  .. code-block:: python

    >>> from cppyy.gbl import global_function, Namespace
    >>> global_function == Namespace.global_function
    False
    >>> from cppyy.gbl.Namespace import global_function
    >>> global_function == Namespace.global_function
    True
    >>> from cppyy.gbl import global_function
    >>>

Free functions can be bound to a class, following the same rules as apply to
Python functions: unless marked as static, they will turn into member
functions when bound to an instance, but act as static functions when called
through the class.
Consider this example:

  .. code-block:: python

    >>> from cppyy.gbl import Concrete, call_abstract_method
    >>> c = Concrete()
    >>> Concrete.callit = call_abstract_method
    >>> Concrete.callit(c)
    called Concrete::abstract_method
    >>> c.callit()
    called Concrete::abstract_method
    >>> Concrete.callit = staticmethod(call_abstract_method)
    >>> c.callit()
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: void ::call_abstract_method(Abstract* a) =>
        TypeError: takes at least 1 arguments (0 given)
    >>> c.callit(c)
    called Concrete::abstract_method
    >>>


`Static methods`
----------------

Class static functions are treated the same way as free functions, except
that they are accessible either through the class or through an instance,
just like Python's ``staticmethod``.


`Instance methods`
------------------

For class methods, see the :ref:`methods section <sec-methods-label>` under
the :doc:`classes heading<classes>`.


`Lambda's`
----------

C++ lambda functions are supported by first binding to a ``std::function``,
then providing a proxy to that on the Python side.
Example::

    >>> cppyy.cppdef("""\
    ... auto create_lambda(int a) {
    ...     return [a](int b) { return a+b; };
    ... }""")
    True
    >>> l = cppyy.gbl.create_lambda(4)
    >>> type(l)
    <class cppyy.gbl.std.function<int(int)> at 0x11505b830>
    >>> l(2)
    6
    >>> 


`Operators`
-----------

Globally defined operators are found lazily (ie. can resolve after the class
definition by loading the global definition or by defining them interactively)
and are mapped onto a Python equivalent when possible.
See the :ref:`operators section <sec-operators-label>` under the
:doc:`classes heading<classes>` for more details.


`Templates`
-----------

Templated functions (and class methods) can either be called using square
brackets (``[]``) to provide the template arguments explicitly, or called
directly, through automatic lookup.
The template arguments may either be a string of type names (this results
in faster code, as it needs no further lookup/verification) or a list of
the actual types to use (which tends to be more convenient).

**Note**: the Python type ``float`` maps to the C++ type ``float``, even
as Python uses a C ``double`` as its internal representation.
The motivation is that doing so makes the Python code more readable (and
Python may anyway change its internal representation in the future).
The same has been true for Python ``int``, which used to be a C ``long``
internally.

Examples, using multiply from :doc:`features.h <cppyy_features_header>`:

  .. code-block:: python

    >>> mul = cppyy.gbl.multiply
    >>> mul(1, 2)
    2
    >>> mul(1., 5)
    5.0
    >>> mul[int](1, 1)
    1
    >>> mul[int, int](1, 1)
    1
    >>> mul[int, int, float](1, 1)
    1.0
    >>> mul[int, int](1, 'a')
     TypeError: Template method resolution failed:
     none of the 6 overloaded methods succeeded. Full details:
     int ::multiply(int a, int b) =>
       TypeError: could not convert argument 2 (int/long conversion expects an integer object)
     ...
     Failed to instantiate "multiply(int,std::string)"
    >>> mul['double, double, double'](1., 5)
    5.0
    >>>


`Overloading`
-------------

C++ supports overloading, whereas Python supports "duck typing", thus C++
overloads have to be selected dynamically in response to the available
"ducks."
This may lead to additional lookups or template instantiations.
However, pre-existing methods (incl. auto-instantiated methods) are always
preferred over new template instantiations:

  .. code-block:: python

    >>> global_function(1.)        # selects 'double' overload
    2.718281828459045
    >>> global_function(1)         # selects 'int' overload
    42
    >>>

C++ does a static dispatch at compile time based on the argument types.
The dispatch is a selection among overloads (incl. templates) visible at the
current parse location in the *translation unit*.
Bound C++ in Python does a dynamic dispatch: it considers all overloads
visible *globally* at the time of execution.
These two approaches, even if completely in line with the expectations of the
respective languages, are fundamentally different and there can thus be
discrepancies in overload selection.
For example, if overloads live in different header files and are only an
implicit conversion apart; or if types that have no direct equivalent in
Python, such as e.g. ``unsigned short``, are used.

It is implicitly assumed that the Python code is correct as-written and there
are no warnings or errors for overloads that C++ would consider ambiguous,
but only if every possible overload fails.
For example, the following overload would be ambiguous in C++ (the value
provided is an integer, but can not be passed through a 4-byte ``int`` type),
but instead ``cppyy`` silently accepts promotion to ``double``:

  .. code-block:: python

    >>> cppyy.cppdef(r"""\
    ...   void process_data(double) { std::cerr << "processing double\n"; }
    ...   void process_data(int32_t) { std::cerr << "processing int\n"; }""")
    True
    >>> cppyy.gbl.process_data(2**32)  # too large for int32_t type
    processing double
    >>>

There are two rounds to run-time overload resolution.
The first round considers all overloads in sorted order, with promotion but
no implicit conversion allowed.
The sorting is based on priority scores of each overload.
Higher priority is given to overloads with argument types that can be
promoted or align better with Python types.
E.g. ``int`` is preferred over ``double`` and ``double`` is preferred over
``float``.
If argument conversion fails for all overloads during this round *and* at
least one argument converter has indicated that it can do implicit
conversion, a second round is tried where implicit conversion, including
instantiation of temporaries, is allowed.
The implicit creation of temporaries, although convenient, can be costly in
terms of run-time performance.

During some template calls, implicit conversion is not allowed, giving
preference to new instantiations (as is the case in C++).
If, however, a previously instantiated overload is available and would match
with promotion, it is preferred over a (costly) new instantiation, unless a
template overload is explicitly selected using template arguments.
For example:

  .. code-block:: python

    >>> cppyy.cppdef(r"""\
    ...   template<typename T>
    ...   T process_T(T t) { return t; }""")
    True
    >>> type(cppyy.gbl.process_T(1.0))
    <class 'float'>
    >>> type(cppyy.gbl.process_T(1))        # selects available "double" overload
    <class 'float'>
    >>> type(cppyy.gbl.process_T[int](1))   # explicit selection of "int" overload
    <class 'int'>
    >>>

The template parameters used for instantiation can depend on the argument
values.
For example, if the type of an argument is Python ``int``, but its value is
too large for a 4-byte C++ ``int``, the template may be instantiated with,
for example, an ``int64_t`` instead (if available on the platform).
Since Python does not have unsigned types, the instantiation mechanism
strongly prefers signed types.
However, if an argument value is too large to fit in a signed integer type,
but would fit in an unsigned type, then that will be used.

If it is important that a specific overload is selected, then use the
``__overload__`` method to match a specific function signature.
An optional boolean second parameter can be used to restrict the selected
method to be const (if ``True``) or non-const (if ``False``).
The return value of which is a first-class callable object, that can be
stored to by-pass the overload resolution:

  .. code-block:: python

    >>> gf_double = global_function.__overload__('double')
    >>> gf_double(1)        # int implicitly promoted
    2.718281828459045
    >>>

The ``__overload__`` method only does a lookup; it performs no (implicit)
conversions and the types in the signature to match should be the fully
resolved ones (no typedefs).
To see all overloads available for selection, use ``help()`` on the function
or look at its ``__doc__`` string:

  .. code-block:: python

    >>> print(global_function.__doc__)
    int ::global_function(int)
    double ::global_function(double)
    >>>

For convenience, the ``:any:`` signature allows matching any overload, for
example to reduce a method to its ``const`` overload only, use:

  .. code-block:: python

     MyClass.some_method = MyClass.some_method.__overload__(':any:', True)


`Overloads and exceptions`
--------------------------

Python error reporting is done using exceptions.
Failed argument conversion during overload resolution can lead to different
types of exceptions coming from respective attempted overloads.
The final error report issued if all overloads fail, is a summary of the
individual errors, but by Python language requirements it has to have a
single exception type.
If all the exception types match, that type is used, but if there is an
amalgam of types, the exception type chosen will be ``TypeError``.
For example, attempting to pass a too large value through ``uint8_t`` will
uniquely raise a ``ValueError``

  .. code-block:: python

    >>> cppyy.cppdef("void somefunc(uint8_t) {}")
    True
    >>> cppyy.gbl.somefunc(2**16)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ValueError: void ::somefunc(uint8_t) =>
        ValueError: could not convert argument 1 (integer to character: value 65536 not in range [0,255])
    >>>

But if other overloads are present that fail in a different way, the error
report will be a ``TypeError``:

  .. code-block:: python

    >>> cppyy.cppdef(r"""
    ...   void somefunc(uint8_t) {}
    ...   void somefunc(std::string) {}""")
    True
    >>> cppyy.gbl.somefunc(2**16)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: none of the 2 overloaded methods succeeded. Full details:
      void ::somefunc(std::string) =>
        TypeError: could not convert argument 1
      void ::somefunc(uint8_t) =>
        ValueError: could not convert argument 1 (integer to character: value 65536 not in range [0,255])
    >>>

Since C++ exceptions are converted to Python ones, there is an interplay
possible between the two as part of overload resolution and ``cppyy``
allows C++ exceptions as such, enabling detailed type disambiguation and
input validation.
(The original use case was for filling database fields, requiring an exact
field label and data type match.)

If, however, all methods fail and there is only one C++ exception (the other
exceptions originating from argument conversion, never succeeding to call
into C++), this C++ exception will be preferentially reported and will have
the original C++ type.


`Return values`
---------------

Most return types are readily amenable to automatic memory management: builtin
returns, by-value returns, (const-)reference returns to internal data, smart
pointers, etc.
The important exception is pointer returns.
 
A function that returns a pointer to an object over which Python should claim
ownership, should have its ``__creates__`` flag set through its
:doc:`pythonization <pythonizations>`.
Well-written APIs will have clear clues in their naming convention about the
ownership rules.
For example, functions called ``New...``, ``Clone...``, etc.  can be expected
to return freshly allocated objects.
A basic name-matching in the pythonization then makes it simple to mark all
these functions as creators.

The return values are :ref:`auto-casted <sec-auto-casting-label>`.


`\*args and \*\*kwds`
---------------------

C++ default arguments work as expected.
Keywords, however, are a Python language feature that does not exist in C++.
Many C++ function declarations do have formal arguments, but these are not
part of the C++ interface (the argument names are repeated in the definition,
making the names in the declaration irrelevant: they do not even need to be
provided).
Thus, although ``cppyy`` will map keyword argument names to formal argument
names from the C++ declaration, use of this feature is not recommended unless
you have a guarantee that the names in C++ the interface are maintained.
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
    >>> c = Concrete(n=17)
    >>> c.m_int
    17
    >>> kwds = {'n' : 18}
    >>> c = Concrete(**kwds)
    >>> c.m_int
    18
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

Python functions can be used to instantiate C++ templates, assuming the
type information of the arguments and return types can be inferred.
If this can not be done directly from the template arguments, then it can
be provided through Python annotations, by explicitly adding the
``__annotations__`` special data member (e.g. for older versions of Python
that do not support annotations), or by the function having been bound by
``cppyy`` in the first place.
For example:

  .. code-block:: python

    >>> import cppyy
    >>> cppyy.cppdef("""\
    ... template<typename R, typename... U, typename... A>
    ... R callT(R(*f)(U...), A&&... a) {
    ...    return f(a...);
    ... }""")
    True
    >>> def f(a: 'int') -> 'double':
    ...     return 3.1415*a
    ...
    >>> cppyy.gbl.callT(f, 2)
    6.283
    >>> def f(a: 'int', b: 'int') -> 'int':
    ...     return 3*a*b
    ...
    >>> cppyy.gbl.callT(f, 6, 7)
    126
    >>>


`extern "C"`
------------

Functions with C linkage are supported and are simply represented as
overloads of a single function.
Such functions are allowed both globally as well as in namespaces.
