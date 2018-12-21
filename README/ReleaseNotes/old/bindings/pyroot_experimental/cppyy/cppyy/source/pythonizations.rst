.. _pythonizations:

Pythonizations
==============

Automatic bindings generation mostly gets the job done, but unless a C++
library was designed with expressiveness and interactivity in mind, using it
will feel stilted.
Thus, if you are not the end-user of a set of bindings, it is beneficial to
implement *pythonizations*.
Some of these are already provided by default, e.g. for STL containers.
Consider the following code, iterating over an STL map, using naked bindings
(i.e. "the C++ way"):

.. code-block:: python

   >>> from cppyy.gbl import std
   >>> m = std.map[int, int]()
   >>> for i in range(10):
   ...     m[i] = i*2
   ...
   >>> b = m.begin()
   >>> while b != m.end():
   ...     print(b.__deref__().second, end=' ')
   ...     b.__preinc__()
   ...
   0 2 4 6 8 10 12 14 16 18 
   >>>   

Yes, that is perfectly functional, but it is also very clunky.
Contrast this to the (automatic) pythonization:

.. code-block:: python

   >>> for key, value in m:
   ...    print(value, end=' ')
   ...
   0 2 4 6 8 10 12 14 16 18
   >>>

Such a pythonization can be written completely in python using the bound C++
methods, with no intermediate language necessary.
Since it is written on abstract features, there is also only one such
pythonization that works for all STL map instantiations.


Installing callbacks
--------------------


