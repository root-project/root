# SoA I/O in RNTuple

RNTuple has a mechanism to represent a collection of a class, i.e. an "array of struct" (AoS),
as "struct of array" (SoA) in memory.
Note that because the data of an AoS in an RNTuple are stored in columnar layout, 
the on-disk layout allows for the effecient transformation into a SoA in-memory layout.

RNTuple provides SoA I/O through the `RSoAField`.
The `RSoAField` stores "RNTuple SoA types", classes with `RVec` data types that meet certain properties (see below).
There is some degree of freedom on the in-memory layout of a SoA type 
but every SoA type has one, well-defined on-disk representation 
based on the corresponding "underlying record type" (see below).

Being a regular class, a SoA type could be stored just through the normal RNTuple I/O.
Storing them using "SoA I/O", however, has the following advantages:
  
  - The length of the vector members is not duplicated on disk.
  - The same on-disk representation can be used to populate different, compatible SoA in-memory layouts, 
    picked at runtime
  - As a runtime decision, the data can also be read into an AoS layout (e.g., a `vector` of the underlying record type)

## RNTuple SoA Types

An RNTuple SoA type can only exist in combination with its "underlying record type".
The underlying record type defines the AoS layout to which the SoA type corresponds.
During writing, the underlying record type may not be used directly by the user but its dictionary must exists.

Concretely, an RNTuple SoA type is a user-defined class that has exactly one associated underlying record type,
with the following constraints:

  - The RNTuple SoA type must meet all the conditions for doing RNTuple I/O (see binary format specification).
  - Likewise, its underlying record type must be a user-defined class that meets the conditions for RNTuple I/O.
  - SoA type `A` is allowed to inherit from a SoA type `B` whose underlying record type is `X` 
    if and only if the underlying record type of `A` inherits from `X`. 
    SoA types must only inherit from other SoA types.
  - For every persistent member of type `T` in the underlying record type, 
    there must be a member with the same name in the SoA type.
    The data type of that data member in the SoA class must be either `RVec<T>` or 
    a SoA type that has `T` as an underlying record type (nested SoA type).
    The SoA type must have no additional persistent data members.
  - The SoA type and its underlying record type must have the same class version number.

These conditions are checked at runtime when an `RSoAField` is created.
Equal vector lengths are ensured by construction when reading from disk and checked when writing to disk.

The underlying record type of a SoA type can be specified in the dictionary or, at runtime, as a class attribute.

Emulated reading (see Architecture.md) reads SoA types as `std::vector<underlying record type>`.

### Example

For the underlying record type(s)

```
struct Properties {
   int fId;
   int fColor;
};

struct Point {
   float fX;
   float fY;
   Properties fProperties;
};
```

a possible SoA layout is

```
struct PointSoA {
   ROOT::RVec<float> fX;
   ROOT::RVec<float> fY;
   ROOT::RVec<Properties> fProperties;
}
```

Another possible SoA layout is

```
struct PropertiesSoA {
   ROOT::RVec<int> fId;
   ROOT::RVec<int> fColor;
};

struct PointSoA {
   PropertiesSoA fProperties;
   ROOT::RVec<float> fY;
   ROOT::RVec<float> fX;
}
```

### Choice of `ROOT::RVec`

For the SoA vectors, the `ROOT::RVec` type is used because it can own or adopt memory.
As a result, optimized code can prepare a memory region and initialize an `RVec` with that region and the right length
in order to directly read into adopted memory.
SoA fields can also be read without any additional logic, in which case the `RVec`s own their memory.

## Schema Evolution of SoA types

Schema evolution of SoA types is identical to normal user-defined classes except for the following caveats.

For added members, reading will set the corresponding vector(s) to the collection length 
and default-initialize the vector elements.
This is different to added members of normal classes, for which reading is a no-op.

For I/O customization rules, there is no check if the rules of the underlying record type are consistent 
with the rules of the SoA types.
A rule mismatch means that data values are different depending on whether data is read into an SoA or the AoS layout.
When reading through the `RSoAField`, the rules of the SoA type apply.
When reading data as a collection of underlying record type, the rules of the underlying record type apply.
