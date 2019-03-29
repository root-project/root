.. cppyy documentation master file, created by
   sphinx-quickstart on Wed Jul 12 14:35:45 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

cppyy: Automatic Python-C++ bindings
====================================

cppyy is an automatic Python-C++ bindings generator, for calling C++ from
Python and Python from C++, designed for large scale programs in high
performance computing that use modern C++.
Design and performance are described in this `PyHPC paper`_, albeit that the
CPython/cppyy performance has been vastly improved since.

cppyy is based on `Cling`_, the C++ interpreter, to match Python's dynamism
and interactivity.
Consider this session, showing dynamic, interactive, mixing of C++ and Python
features (more examples are in the `tutorial`_):

.. code-block:: python

   >>> import cppyy
   >>> cppyy.cppdef("""
   ... class MyClass {
   ... public:
   ...     MyClass(int i) : m_data(i) {}
   ...     virtual ~MyClass() {}
   ...     virtual int add_int(int i) { return m_data + i; }
   ...     int m_data;
   ... };""")
   True
   >>> from cppyy.gbl import MyClass
   >>> m = MyClass(42)
   >>> cppyy.cppdef("""
   ... void say_hello(MyClass* m) {
   ...     std::cout << "Hello, the number is: " << m->m_data << std::endl;
   ... }""")
   True
   >>> MyClass.say_hello = cppyy.gbl.say_hello
   >>> m.say_hello()
   Hello, the number is: 42
   >>> m.m_data = 13
   >>> m.say_hello()
   Hello, the number is: 13
   >>> class PyMyClass(MyClass):
   ...     def add_int(self, i):  # python side override (CPython only)
   ...         return self.m_data + 2*i
   ...
   >>> cppyy.cppdef("int callback(MyClass* m, int i) { return m->add_int(i); }")
   True
   >>> cppyy.gbl.callback(m, 2)             # calls C++ add_int
   15
   >>> cppyy.gbl.callback(PyMyClass(1), 2)  # calls Python-side override
   5
   >>>

With a modern C++ compiler having its back, cppyy is future-proof.
Consider the following session using ``boost::any``, a capsule-type that
allows for heterogeneous containers in C++.
The `Boost`_ library is well known for its no holds barred use of modern C++
and heavy use of templates:

.. code-block:: python

   >>> import cppyy
   >>> cppyy.include('boost/any.hpp')
   >>> from cppyy.gbl import std, boost
   >>> val = boost.any()                    # the capsule 
   >>> val.__assign__(std.vector[int]())    # assign it a std::vector<int>
   <cppyy.gbl.boost.any object at 0xf6a8a0>
   >>> val.type() == cppyy.typeid(std.vector[int])    # verify type
   True
   >>> extract = boost.any_cast[int](std.move(val))   # wrong cast
   Traceback (most recent call last):
     File "<stdin>", line 1, in <module>
   Exception: int boost::any_cast(boost::any&& operand) =>
       boost::bad_any_cast: failed conversion using boost::any_cast (C++ exception)
   >>> extract = boost.any_cast[std.vector[int]](val) # correct cast
   >>> type(extract) is std.vector[int]
   True
   >>> extract += xrange(100)
   >>> len(extract)
   100
   >>> val.__assign__(std.move(extract))    # move forced
   <cppyy.gbl.boost.any object at 0xf6a8a0>
   >>> len(extract)                         # now empty (or invalid)
   0
   >>> extract = boost.any_cast[std.vector[int]](val)
   >>> list(extract)
   [0, 1, 2, 3, 4, 5, 6, ..., 97, 98, 99]
   >>>

And yes, there is no reason to use Boost from Python (in fact, this example
calls out for :doc:`pythonizations <pythonizations>`), but it shows that
cppyy seamlessly supports many advanced C++ features.

cppyy is available for both `CPython`_ (v2 and v3) and `PyPy`_, reaching
C++-like performance with the latter.
It makes judicious use of precompiled headers, dynamic loading, and lazy
instantiation, to support C++ programs consisting of millions of lines of
code and many thousands of classes.
cppyy minimizes dependencies to allow its use in distributed, heterogeneous,
development environments.

.. _Cling: https://root.cern.ch/cling
.. _tutorial: https://bitbucket.org/wlav/cppyy/src/master/doc/tutorial/CppyyTutorial.ipynb?viewer=nbviewer&fileviewer=notebook-viewer%3Anbviewer
.. _`PyHPC paper`: http://wlav.web.cern.ch/wlav/Cppyy_LavrijsenDutta_PyHPC16.pdf
.. _`Boost`: http://www.boost.org/
.. _`CPython`: http://python.org
.. _`PyPy`: http://pypy.org


.. only: not latex

   Contents:

.. toctree::
   :caption: Getting Started
   :maxdepth: 1

   installation
   changelog

.. toctree::
   :caption: Features
   :maxdepth: 1

   basic_types
   classes
   functions
   type_conversions
   python
   features

.. toctree::
   :caption: Redistribution
   :maxdepth: 1

   pythonizations
   dictionaries
   bindings_generation

.. toctree::
   :caption: Developers
   :maxdepth: 1

   packages
   repositories


Comments and bugs
-----------------

Please report bugs or requests for improvement on the `issue tracker`_.

.. _`issue tracker`: https://bitbucket.org/wlav/cppyy/issues
