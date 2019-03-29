.. _classes:


Classes
=======

Both Python and C++ support object-oriented code through classes and thus
it is logical to expose C++ classes as Python ones, including the full
inheritance hierarchy.

The C++ code used for the examples below can be found
:doc:`here <cppyy_features_header>`, and it is assumed that that code is
loaded at the start of any session.
Download it, save it under the name ``features.h``, and load it:

  .. code-block:: python

    >>> import cppyy
    >>> cppyy.include('features.h')
    >>>


`Basics`
--------

All bound C++ code starts off from the global C++ namespace, represented in
Python by ``gbl``.
This namespace, as any other namespace, is treated as a module after it has
been loaded.
Thus, we can import C++ classes that live underneath it:

  .. code-block:: python

    >>> from cppyy.gbl import Concrete
    >>> Concrete
    <class cppyy.gbl.Concrete at 0x2058e30>
    >>>

Placing classes in the same structure as imposed by C++ guarantees identity,
even if multiple Python modules bind the same class.
There is, however, no necessity to expose that structure to end-users: when
developing a Python package that exposes C++ classes through ``cppyy``,
consider ``cppyy.gbl`` an "internal" module, and expose the classes in any
structure you see fit.
The C++ names will continue to follow the C++ structure, however, as is needed
for e.g. pickling:

  .. code-block:: python

    >>> from cppyy.gbl import Namespace
    >>> Concrete == Namespace.Concrete
    False
    >>> n = Namespace.Concrete.NestedClass()
    >>> type(n)
    <class cppyy.gbl.Namespace.Concrete.NestedClass at 0x22114c0>
    >>> type(n).__name__
    NestedClass
    >>> type(n).__module__
    cppyy.gbl.Namespace.Concrete
    >>> type(n).__cppname__
    Namespace::Concrete::NestedClass
    >>>


`Inheritance`
-------------

The output of help shows the inheritance hierarchy, constructors, public
methods, and public data.
For example, ``Concrete`` inherits from ``Abstract`` and it has
a constructor that takes an ``int`` argument, with a default value of 42.
Consider:

  .. code-block:: python

    >>> from cppyy.gbl import Abstract
    >>> issubclass(Concrete, Abstract)
    True
    >>> a = Abstract()
    Traceback (most recent call last):
      File "<console>", line 1, in <module>
    TypeError: cannot instantiate abstract class 'Abstract'
    >>> c = Concrete()
    >>> isinstance(c, Concrete)
    True
    >>> isinstance(c, Abstract)
    True
    >>> d = Concrete(13)
    >>>

Just like in C++, interface classes that define pure virtual methods, such
as ``Abstract`` does, can not be instantiated, but their concrete
implementations can.
As the output of ``help`` showed, the ``Concrete`` constructor takes
an integer argument, that by default is 42.


`Cross-inheritance`
-------------------

Python classes that derive from C++ classes can override virtual methods as
long as those methods are declared on class instantiation (adding methods to
the Python class after the fact will not provide overrides on the C++ side,
only on the Python side).
Example:

  .. code-block:: python

    >>> from cppyy.gbl import Abstract, call_abstract_method
    >>> class PyConcrete(Abstract):
    ...     def abstract_method(self):
    ...         print("Hello, Python World!\n")
    ...     def concrete_method(self):
    ...         pass
    ...
    >>> pc = PyConcrete()
    >>> call_abstract_method(pc)
    Hello, Python World!
    >>> 

Note that it is not necessary to provide a constructor (``__init__``), but
if you do, you *must* call the base class constructor through the ``super``
mechanism.


`Typedefs`
----------

Typedefs are simple python references to the actual classes to which
they refer.

  .. code-block:: python

    >>> from cppyy.gbl import Concrete_t
    >>> Concrete is Concrete_t
    True
    >>>


`Data members`
--------------

The ``Concrete`` instances have a public data member ``m_int`` that
is treated as a Python property, albeit a typed one:

  .. code-block:: python

    >>> c.m_int, d.m_int
    (42, 13)
    >>> c.m_int = 3.14   # a float does not fit in an int
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: int/long conversion expects an integer object
    >>> c.m_int = int(3.14)
    >>> c.m_int, d.m_int
    (3, 13)
    >>>

Note that private and protected data members are not accessible and C++
const-ness is respected:

  .. code-block:: python

    >>> c.m_const_int = 71    # declared 'const int' in class definition
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: assignment to const data not allowed
    >>>

Static C++ data members act like Python class-level data members.
They are also represented by property objects and both read and write access
behave as expected:

  .. code-block:: python

     >>> Concrete.s_int       # access through class
     321
     >>> c.s_int = 123        # access through instance
     >>> Concrete.s_int
     123


`Methods`
---------

C++ methods are represented as Python ones: these are first-class objects and
can be bound to an instance.
If a method is virtual in C++, the proper concrete method is called, whether
or not the concrete class is bound.
Similarly, if all classes are bound, the normal Python rules apply:

  .. code-block:: python

    >>> c.abstract_method()
    called Concrete::abstract_method
    >>> c.concrete_method()
    called Concrete::concrete_method
    >>> m = c.abstract_method
    >>> m()
    called Concrete::abstract_method
    >>>


`Templates`
-----------

Templated classes are instantiated using square brackets.
(For backwards compatibility reasons, parentheses work as well.)
The instantiation of a templated class yields a class, which can then
be used to create instances.

Templated classes need not pre-exist in the bound code, just their
declaration needs to be available.
This is true for e.g. all of STL:

  .. code-block:: python

    >>> cppyy.gbl.std.vector                # template metatype
    <cppyy.Template 'std::vector' object at 0x7fffed2674d0>
    >>> cppyy.gbl.std.vector(int)           # instantiates template -> class
    <class cppyy.gbl.std.vector<int> at 0x1532190>
    cppyy.gbl.std.vector[int]()             # instantiates class -> object
    <cppyy.gbl.std.vector<int> object at 0x2341ec0>
    >>>

The template arguments may be actual types or their names as a string,
whichever is more convenient.
Thus, the following are equivalent:

  .. code-block:: python

     >>> from cppyy.gbl.std import vector
     >>> type1 = vector[Concrete]
     >>> type2 = vector['Concrete']
     >>> type1 == type2
     True
     >>>

