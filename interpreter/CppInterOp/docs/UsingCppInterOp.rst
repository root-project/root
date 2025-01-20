Using CppInterop
----------------

C++ Language Interoperability Layer
===================================

Loading Dynamic shared library
==============================

The CppInterop comes with using it is a dynamic shared library, which resides 
in the build/lib/ after building CppInterOp following the instructions in 
:doc:`Installation and usage <InstallationAndUsage>` .

.. code-block:: bash

    libInterop = ctypes.CDLL("./libclangCppInterOp.so")
    
The above method of usage is for Python on Linux; for C, we can include the headers of 
the library. Including this library in our program enables the user to use 
the abilities of CppInterOp. CppInterOp helps programmers with multiple 
verifications such as isClass, isBoolean, isStruct, and many more in different 
languages. With the interop layer, we can access the scopes, namespaces of 
classes and members that are being used. The interoperability layer helps us 
with the instantiation of templates, diagnostic interaction, creation of 
objects, and many more things.

Using LLVM as external library
==============================

In CppInterOp, we are leveraging Clang as a library for interoperability purposes.
To use Clang, we need to pass the Clang configuration to the CMake build system,
so that the build system recognizes the configuration and enables usage of Clang
and LLVM.
We can consider clang-repl as a state manager, where CppInterOp allows you to
query the state from the state manager. Thereafter, cppyy uses this to create
Python objects for C++.
This section briefly describes all the key **features** offered by 
CppInterop. If you are just getting started with CppInterop, then this is the 
best place to start.

Incremental Adoption
====================
CppInterOp can be adopted incrementally. While the rest of the framework is the 
same, a small part of CppInterOp can be utilized. More components may be 
adopted over time.

Minimalist by design
====================
While the library includes some tricky code, it is designed to be simple and
robust (simple function calls, no inheritance, etc.). The goal is to make it as
close to the compiler API as possible, and each routine should do just one thing.
that it was designed for.

How cppyy leverages CppInterOp
===============================

cppyy is a run-time Python-C++ bindings generator for calling C++ from Python
and Python from C++. Interestingly, it uses C++ interactively by using the
compiler as a service. This is made possible by the CppInterOp library.
Following are some of the ways cppyy leverages CppInterOp for better
performance and usability.

1. **CppInterOp enables interoperability with C++ code**: CppInterOp provides a
   minimalist and robust interface for language interoperability on the fly,
   which helps CPPYY generate dynamic Python-C++ bindings by using a C++
   interpreter (e.g., Clang-REPL/Cling) and LLVM.

2. **Reducing dependencies**: Reducing domain-specific dependencies of cppyy
   (e.g., on the Cling interpreter and the ROOT framework) to enable more
   generalized usage.

3. **LLVM Integration**: CppInterOp is designed to be used as a part of the
   LLVM toolchain (as part of Clang-REPL) that can then be used as a runtime
   compiler for CPPYY. This simplifies the codebase of CPPYY and enhances its
   performance.

 4. **Making C++ More Social**: CppInterOp and cppyy help data scientists that
    are working with legacy C++ code experiment with simpler, more interactive
    languages, while also interacting with larger communities.

**CppInterOp enables interoperability with C++ code**

cppyy is a major use case for CppInterOp. cppyy is an automatic run-time
bindings generator for Python and C++, and supports a wide range of C++
features, including template instantiation. It operates on demand and generates
only what is necessary. It requires a compiler (Cling or Clang-REPL) that can
be available during program runtime.

**Reducing Dependencies**

Recent work done on cppyy has been focused on reducing dependencies on
domain-specific infrastructure (e.g., the ROOT framework). Using an independent
library such as CppInterOp helps accomplish that, while also improving the code
consistency in cppyy.

The CppInterOp library can be configured to use the newly developed Clang-Repl
backend available in LLVM upstream (or to use the Cling legacy backend, for
compatibility with High Energy Physics applications).

Only a small set of APIs are needed to connect to the interpreter (Clang-Repl/
Cling), since other APIs are already available in the standard compiler. This
is one of the reasons that led to the creation of CppInterOp (a library of
helper functions), that can help extract out things that are unnecessary for
for core cppyy functionality.

The cppyy API surface is now incomparably smaller and simpler than what it used
to be.

**LLVM Integration**

Once CppInterOp is integrated with LLVM's Clang-REPL component (that can then
be used as a runtime compiler for cppyy), it will further enhance cppyy's
performance in the following ways:


- *Simpler codebase:* The removal of string parsing logic will lead to a
  simpler code base.

- *Built into the LLVM toolchain:* The CppInterOp depends only on the LLVM
  toolchain (as part of Clang-REPL).

- *Better C++ Support:* Finer-grained control over template instantiation is
  available through CppInterOp.

- *Fewer Lines of Code:* A lot of dependencies and workarounds will be
  removed, reducing the lines of code required to execute cppyy.

- *Well tested interoperability Layer:* The CppInterOp interfaces have full
  unit test coverage.

**Making C++ More Social**

cppyy is the first use case demonstrating how CppInterOp can enable C++ to be
more easily interoperable with other languages. This helps many data scientists
that are working with legacy C++ code and would like to use simpler, more
interactive languages.

The goal of these enhancements is to eventually land these interoperability
tools (including CppInterOp) to broader communities like LLVM and Clang, to
enable C++ to interact with other languages besides Python.

Example: Template Instantiation
-------------------------------

The developmental cppyy version can run basic examples such as the one
here. Features such as standalone functions and basic classes are also
supported.

C++ code (Tmpl.h)

::

   template <typename T>
   struct Tmpl {
     T m_num;
     T add (T n) {
       return m_num + n;
   }
   };

Python Interpreter

::

   >>> import cppyy
   >>> cppyy.include("Tmpl.h")
   >>> tmpl = Tmpl[int]()
   >>> tmpl.m_num = 4
   >>> print(tmpl.add(5))
   9
   >>> tmpl = Tmpl[float]()
   >>> tmpl.m_num = 3.0
   >>> print(tmpl.add(4.0))
   7.0

Where does the cppyy code reside?
---------------------------------

Following are the main components where cppyy logic (with Compiler Research
Organization’s customizations started by `sudo-panda`_) resides:

-  `cppyy <https://github.com/compiler-research/cppyy>`_
-  `cppyy-backend <https://github.com/compiler-research/cppyy-backend>`_
-  `CPyCppyy <https://github.com/compiler-research/CPyCppyy>`_

..

   Note: These are forks of the `upstream cppyy`_ repos created by `wlav`_.

CppInterOp is a separate library that helps these packages communicate with C++
code.

-  `CppInterOp <https://github.com/compiler-research/CppInterOp/tree/main>`_

How cppyy components interact with each other
---------------------------------------------

cppyy is made up of the following packages: 

- A frontend: cppyy, 

- A backend: cppyy-backend, and 

- An extension: CPyCppyy.

Besides these, the ``CppInterOp`` library serves as an additional layer on top
of Cling/Clang-REPL that helps these packages in communicating with C++ code.

**1. cppyy-backend**

The `cppyy-backend`_ package forms a layer over ``cppyy``, for example,
modifying some functionality to provide the functions required for
``CPyCppyy``. 

  `CPyCppyy`_ is a CPython extension module built on top of the same backend
  API as PyPy/_cppyy. It thus requires the installation of the cppyy-backend
  for use, which will pull in Cling. 

``cppyy-backend`` also adds some `utilities`_ to help with repackaging and
redistribution.

For example, ``cppyy-backend`` initializes the interpreter (using the
``clingwrapper::ApplicationStarter`` function), adds the required ``include``
paths, and adds the headers required for cppyy to work. It also adds some
checks and combines two or more functions to help CPyCppyy work.

These changes help ensure that any change in ``cppyy`` doesn’t directly
affect ``CPyCppyy``, and the API for ``CPyCppyy`` remains unchanged.

**2. CPyCppyy**

The ``CPyCppyy`` package uses the functionality provided by ``cppyy-backend``
and provides Python objects for C++ entities. ``CPyCppyy`` uses separate proxy
classes for each type of object. It also includes helper classes, for example,
``Converters.cxx`` helps convert Python type objects to C++ type objects, while
``Executors.cxx`` is used to execute a function and convert its return value to
a Python object, so that it can be used inside Python.

**3. cppyy**

The cppyy package provides the front-end for Python. It is `included in code`_
(using ``import cppyy``) to import cppyy in Python. It initializes things on
the backend side, provides helper functions (e.g., ``cppdef()``, ``cppexec()``,
etc.) that the user can utilize, and it calls the relevant backend functions
required to initialize cppyy.


Further Reading
---------------

-  `High-performance Python-C++ bindings with PyPy and
   Cling <http://cern.ch/wlav/Cppyy_LavrijsenDutta_PyHPC16.pdf>`_

-  `Efficient and Accurate Automatic Python Bindings with cppyy &
   Cling <https://arxiv.org/abs/2304.02712>`_

-  cppyy documentation:
   `cppyy.readthedocs.io <http://cppyy.readthedocs.io/>`_.

-  Notebook-based tutorial: `Cppyy
   Tutorial <https://github.com/wlav/cppyy/blob/master/doc/tutorial/CppyyTutorial.ipynb>`_.

-  `C++ Language Interoperability
   Layer <https://compiler-research.org/libinterop/>`_

**Credits:**

-  `Wim Lavrijsen <https://github.com/wlav>`_ (Lawrence Berkeley National Lab.)
   for his original work in cppyy and mentorship towards student contributors.

-  `Vassil Vasilev <https://github.com/vgvassilev>`_ (Princeton University)
   for his mentorship towards Compiler Research Org's student contributors.

-  `Baidyanath Kundu <https://github.com/sudo-panda>`_ (Princeton University)
   for his research work on cppyy and Numba with `Compiler Research Organization`_ 
   (as discussed in this document).
   
- `Aaron Jomy <https://github.com/maximusron>`_ (Princeton University) for
  continuing this research work with `Compiler Research Organization`_.

In case you haven't already installed CppInterop, please do so before proceeding
with the Installation And Usage Guide.
:doc:`Installation and usage <InstallationAndUsage>`

.. _Compiler Research Organization: https://compiler-research.org/

.. _upstream cppyy: https://github.com/wlav/cppyy

.. _wlav: https://github.com/wlav

.. _utilities: https://cppyy.readthedocs.io/en/latest/utilities.html

.. _included in code: https://cppyy.readthedocs.io/en/latest/starting.html

.. _sudo-panda: https://github.com/sudo-panda

.. _cppyy: https://cppyy.readthedocs.io/en/latest/index.html

.. _CppInterOp: https://github.com/compiler-research/CppInterOp

.. _ROOT meta: https://github.com/root-project/root/tree/master/core/meta

.. _enhancements in cppyy: https://arxiv.org/abs/2304.02712

.. _CPyCppyy: https://github.com/wlav/CPyCppyy

.. _cppyy-backend: https://github.com/wlav/cppyy-backend
