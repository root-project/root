.. _features:

Miscellaneous
=============

.. toctree::
   :hidden:

   cppyy_features_header

This is a collection of a few more features listed that do not have a proper
place yet in the rest of the documentation.

The C++ code used for the examples below can be found
:doc:`here <cppyy_features_header>`, and it is assumed that that code is
loaded at the start of any session.
Download it, save it under the name ``features.h``, and load it:

  .. code-block:: python

    >>> import cppyy
    >>> cppyy.include('features.h')
    >>>


`Odds and ends`
---------------

* **memory**: C++ instances created by calling their constructor from python
  are owned by python.
  You can check/change the ownership with the __python_owns__ flag that every
  bound instance carries.
  Example:

  .. code-block:: python

    >>> from cppyy.gbl import Concrete
    >>> c = Concrete()
    >>> c.__python_owns__         # True: object created in Python
    True
    >>>

* **namespaces**: Are represented as python classes.
  Namespaces are more open-ended than classes, so sometimes initial access may
  result in updates as data and functions are looked up and constructed
  lazily.
  Thus the result of ``dir()`` on a namespace shows the classes available,
  even if they may not have been created yet.
  It does not show classes that could potentially be loaded by the class
  loader.
  Once created, namespaces are registered as modules, to allow importing from
  them.
  Namespace currently do not work with the class loader.
  Fixing these bootstrap problems is on the TODO list.
  The global namespace is ``cppyy.gbl``.

* **NULL**: Is represented as ``cppyy.nullptr``.
  In C++11, the keyword ``nullptr`` is used to represent ``NULL``.
  For clarity of intent, it is recommended to use this instead of ``None``
  (or the integer ``0``, which can serve in some cases), as ``None`` is better
  understood as ``void`` in C++.

* **static methods**: Are represented as python's ``staticmethod`` objects
  and can be called both from the class as well as from instances.

* **templated functions**: Automatically participate in overloading and are
  used in the same way as other global functions.

* **templated methods**: For now, require an explicit selection of the
  template parameters.
  This will be changed to allow them to participate in overloads as expected.

* **unary operators**: Are supported if a python equivalent exists, and if the
  operator is defined in the C++ class.
