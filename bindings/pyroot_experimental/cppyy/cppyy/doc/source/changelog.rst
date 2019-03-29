.. _changelog:

Changelog
=========

For convenience, this changelog keeps tracks of changes with version numbers
of the main cppyy package, but many of the actual changes are in the lower
level packages, which have their own releases.
See :doc:`packages <packages>`, for details on the package structure.
PyPy support lags CPython support.


MASTER : 1.4.8
--------------

* Properly resolve overloaded functions with using of templates from bases
* Get templated constructor info from decl instead of name comparison


2019-04-04 : 1.4.7
------------------

* Enable initializer_list conversion on Windows as well
* Improved mapping of operator() for indexing (e.g. for matrices)
* Implicit conversion no longer uses global state to prevent recursion
* Improved overload reordering
* Fixes for templated constructors in namespaces


2019-04-02 : 1.4.6
------------------

* More transparent use of smart pointers such as shared_ptr
* Expose versioned std namespace through using on Mac
* Improved error handling and interface checking in cross-inheritance
* Argument of (const/non-const) ref types support in callbacks/cross-inheritance
* Do template argument resolution in order: reference, pointer, value
* Fix for return type deduction of resolved but uninstantiated templates
* Fix wrapper generation for defaulted arguments of private types
* Several linker fixes on 64b Windows


2019-03-25 : 1.4.5
------------------

* Allow templated free functions to be attached as methods to classes
* Allow cross-derivation from templated classes
* More support for 'using' declarations (methods and inner namespaces)
* Fix overload resolution for std::set::rbegin()/rend() operator ==
* Fixes for bugs #61, #67
* Several pointer truncation fixes fo 64b Windows
* Linker and lookup fixes for Windows


2019-03-20 : 1.4.4
------------------

* Support for 'using' of namespaces
* Improved support for alias templates
* Faster template lookup
* Have rootcling/genreflex respect compile-time flags (except for --std if
  overridden by CLING_EXTRA_FLAGS)
* Utility to build dictionarys on Windows (32/64)
* Name mangling fixes in Cling for JITed global/static variables on Windows
* Several pointer truncation fixes for 64b Windows


2019-03-10 : 1.4.3
------------------

* Cross-inheritance from abstract C++ base classes
* Preserve 'const' when overriding virtual functions
* Support for by-ref (using ctypes) for function callbacks
* Identity of nested typedef'd classes matches actual
* Expose function pointer variables as std::function's
* More descriptive printout of global functions
* Ensure that standard pch is up-to-date and that it is removed on
  uninstall
* Remove standard pch from wheels on all platforms
* Add -cxxflags option to rootcling
* Install clang resource directory on Windows
