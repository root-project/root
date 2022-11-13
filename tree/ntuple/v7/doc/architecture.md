RNTuple Code Architecture
=========================

> This document is meant for ROOT developers. It provides background information on the RNTuple code design and behavior.

> This document is currently a stub.


Semantics of reading a non-trivial objects
==========================================

Reading an object with RNTuple should be seen as _overwriting_ its persistent data members.
Given a properly constructed and valid object, the object must ensure that it stays valid when overwriting its persistent data members.
However, the object should not rely on its transient state to remain unchanged during reading:
it may be destructed and constructed again when it is read as part of a collection (see below).

An object that is being read from disk may have been constructed by `RField::GenerateValue()`.
In this case, RNTuple owns the object and it will be destructed by `RField::DestroyValue()`.

When reading collections of type `T` (`std::vector<T>`, `ROOT::RVec<T>`, ...), RNTuple uses `RField::GenerateValue()` to construct elements of the inner type `T`.
As the size of a collection changes from event to event, this has the following effect on its elements
  - If the collection shrinks, cut-off elements are destructed
  - If the collection grows, new elements are constructed before reading them
  - If the array buffer of the collection is reallocated (may happen for both shrinking and growing depending on the collection), all elements are destructed first in the old buffer
  and the new number of elements is constructed in the new buffer

So unless the collection buffer needs to be reallocated, RNTuple tries to avoid unnecessary destruction/construction but instead overwrites existing objects.
Note that RNTuple currently does not copy or move existing objects when the collection buffer is reallocated.
