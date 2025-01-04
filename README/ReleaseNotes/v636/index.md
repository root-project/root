% ROOT Version 6.36 Release Notes
% 2025-05
<a name="TopOfPage"></a>

## Introduction

ROOT version 6.36.00 is scheduled for release at the end of May 2025.

For more information, see:

[http://root.cern](http://root.cern)

The following people have contributed to this new version:

## Deprecation and Removal

* The RooFit legacy interfaces that were deprecated in ROOT 6.34 and scheduled for removal in ROOT 6.36 are removed. See the RooFit section in the 6.34 release notes for a full list.
* The `TPython::Eval()` function that was deprecated in ROOT 6.34 and scheduled for removal in ROOT 6.36 is removed.
* The `RooDataSet` constructors to construct a dataset from a part of an existing dataset are deprecated and will be removed in ROOT 6.38. This is to avoid interface duplication. Please use `RooAbsData::reduce()` instead, or if you need to change the weight column, use the universal constructor with the `Import()`, `Cut()`, and `WeightVar()` arguments.
* The ROOT splash screen was removed for Linux and macOS
* Proof support has been completely removed form RooFit and RooStats, after it was already not working anymore for several releases

## Python Interface

### Changed ownership policy for non-`const` pointer member function parameters

If you have a member function taking a raw pointer, like `MyClass::foo(T *obj)`,
PyROOT was so far assuming that calling this method on `my_instance`
transfers the ownership of `obj` to `my_instance`.

However, this resulted in many memory leaks, since many functions with such a
signature actually don't take ownership of the object.

To avoid such memory leaks, PyROOT now doesn't make this guess anymore as of
ROOT 6.32. Because of this change, some double-deletes or dangling references
might creep up in your scripts. These need to be fixedby properly managing
object lifetime with Python references.

You can fix the dangling references problem for example via:

  1. Assigning the object to a python variable
  2. Creating an owning collection that keeps the objects alive
  3. Writing a pythonization for the member function that does the ownership
     transfer if needed

The double-delete problems can be fixed via:

  1. Drop the ownership on the Python side with `ROOT.SetOwnership(obj, False)`
  3. Writing a pythonization for the member function that drops the ownership on the Python side as above

This affects for example the `TList::Add(TObject *obj)` member function, which
will not transfer ownership from PyROOT to the TList anymore. The new policy
fixes a memory leak, but at the same time it is not possible anymore to create
the contained elements in place:

```python
# A TList is by default a non-owning container
my_list = ROOT.TList()


# This is working, but resulted in memory leak prior to ROOT 6.32:
obj_1 = ROOT.TObjString("obj_1")
my_list.Add(obj_1)


# This is not allowed anymore, as the temporary would be
# deleted immediately leaving a dangling pointer:
my_list.Add(ROOT.TObjString("obj_2"))

# Python reference count to contained object is now zero,
# TList contains dangling pointer!
```

**Note:** You can change back to the old policy by calling
`ROOT.SetMemoryPolicy(ROOT.kMemoryHeuristics)` after importing ROOT, but this
should be only used for debugging purposes and this function might be removed
in the future!

## RDataFrame

## RooFit

## IO

## RDataFrame

## Tutorials and Code Examples

## Core 

## Histograms

## Math

## Graphics

## Geometry

## Montecarlo

## JavaScript ROOT

## Class Reference Guide

## Build, Configuration and Testing Infrastructure


