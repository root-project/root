# Schema Evolution

Schema evolution is the capability of the ROOT I/O to read data
into in-memory models that are different but compatible to the on-disk schema.

Schema evolution allows for data models to evolve over time
such that old data can be read into current models ("backward compatibility")
and old software can read newer data models ("forward compatibility").
For instance, data model authors may over time add and reorder class members, change data types
(e.g. `std::vector<float>` --> `ROOT::RVec<double>`), rename classes, etc.

ROOT applies automatic schema evolution rules for common, safe and unambiguous cases.
Users can complement the automatic rules by manual schema evolution ("I/O customization rules")
where custom code snippets implement the transformation logic.
In case neither automatic nor any of the provided I/O customization rules suffice
to transform the on-disk schema into the in-memory model, ROOT will error out and refrain from reading data.

This document describes schema evolution support implemented in RNTuple.
For the most part, schema evolution works identical across the different ROOT I/O systems (TFile, TTree, RNTuple).
The exceptions are listed in the last section of this document.

## Automatic schema evolution

ROOT applies a number of rules to read data transparently into in-memory models
that are not an exact match to the on-disk schema.
The automatic rules apply recursively to compound types (classes, tuples, collections, etc.);
the outer types are evolved before the inner types.

Automatic schema evolution rules transform native _types_ as well as the _shape_ of user-defined classes
as listed in the following, exhaustive tables.

### Class shape transformations

User-defined classes can automatically evolve their layout in the following ways.
Note that users should increase the class version number when the layout changes.

| Layout Change                           | Comment                                              |
| --------------------------------------- | ---------------------------------------------------- |
| Remove member                           | Match by member name                                 |
| Add member                              | Match by member name, new member default-initialized |
| Reorder members                         | Match by member name                                 |
| Remove all base classes                 |                                                      |
| Add base class(es) where they were none | New base class members default initialized           |

Reordering and incremental addition or removal of base classes is currently unsupported
but may be supported in future RNTuple versions.

The class shape evolution also applies to untyped records.
Note that untyped records cannot have base classes.

### Type transformations

ROOT transparently reads into in-memory types that are different from but compatible to the on-disk type.
In the following tables, `T'` denotes a type that is compatible to `T`.
This includes user-defined types that are related via a renaming rule.

#### Plain fields

| In-memory type              | Compatible on-disk types    | Comment                 |
| --------------------------- | --------------------------- | ------------------------|
| `bool`                      | `char`                      |                         |
|                             | `std::[u]int[8,16,32,64]_t` |                         |
|                             | enum                        |                         |
|-----------------------------|-----------------------------|-------------------------|
| `char`                      | `bool`                      |                         |
|                             | `std::[u]int[8,16,32,64]_t` | with bounds check       |
|                             | enum                        | with bounds check       |
|-----------------------------|-----------------------------|-------------------------|
| `std::[u]int[8,16,32,64]_t` | `bool`                      |                         |
|                             | `char`                      |                         |
|                             | `std::[u]int[8,16,32,64]_t` | with bounds check       |
|                             | enum                        | with bounds check       |
|-----------------------------|-----------------------------|-------------------------|
| enum                        | enum of different type      | with bounds check       |
|                             |                             | on underlying integer   |
|-----------------------------|-----------------------------|-------------------------|
| float                       | double                      | with fp class check[^1] |
|-----------------------------|-----------------------------|-------------------------|
| double                      | float                       |                         |
|-----------------------------|-----------------------------|-------------------------|
| `std::atomic<T>`            | `T'`                        |                         |

[^1]: The floating point class check ensures that the on-disk value and the in-memory value are of the same nature
(NaN, +/-inf, zero, underflow, or normal value).


#### Variable-length collections

The different variable-length collections have the same on-disk representation
and thus evolve naturally into one another.
However, only those transformations that are guarantueed to work at runtime will be performed.
For instance, a set can always be read as a vector but a vector does not necessarily fulfil the set property.

| In-memory type                   | Compatible on-disk types             | Comment                               |
| -------------------------------- | ------------------------------------ | ------------------------------------- |
| `std::vector<T>`                 | `ROOT::RVec<T'>`                     |                                       |
|                                  | `std::array<T', N>`                  |                                       |
|                                  | `std::[unordered_][multi]set<T'>`    |                                       |
|                                  | `std::[unordered_][multi]map<K',V'>` | only `T` = `std::[pair,tuple]<K,V>`   |
|                                  | `std::optional<T'>`                  |                                       |
|                                  | `std::unique_ptr<T'>`                |                                       |
|                                  | User-defined collection of `T'`      |                                       |
|                                  | Untyped collection of `T'`           |                                       |
|----------------------------------|--------------------------------------|---------------------------------------|
| `ROOT::RVec<T>`                  | `std::vector<T'>`                    | with size check                       |
|                                  | `std::array<T', N>`                  | with size check                       |
|                                  | `std::[unordered_][multi]set<T'>`    | with size check                       |
|                                  | `std::[unordered_][multi]map<K',V'>` | only `T` = `std::[pair,tuple]<K,V>`,  |
|                                  |                                      | with size check                       |
|                                  | `std::optional<T'>`                  |                                       |
|                                  | `std::unique_ptr<T'>`                |                                       |
|                                  | User-defined collection of `T'`      | with size check                       |
|                                  | Untyped collectionof `T'`            | with size check                       |
|----------------------------------|--------------------------------------|---------------------------------------|
| `std::[unordered_]set<T>`        | `std::[unordered_]set<T'>`           |                                       |
|                                  | `std::[unordered_]map<K',V'>`        | only `T` = `std::[pair,tuple]<K,V>`   |
|----------------------------------|--------------------------------------|---------------------------------------|
| `std::[unordered_]multiset<T>`   | `ROOT::RVec<T'>`                     |                                       |
|                                  | `std::vector<T'>`                    |                                       |
|                                  | `std::array<T', N>`                  |                                       |
|                                  | `std::[unordered_][multi]set<T'>`    |                                       |
|                                  | `std::[unordered_][multi]map<K',V'>` | only `T` = `std::[pair,tuple]<K,V>`   |
|                                  | User-defined collection of `T'`      |                                       |
|                                  | Untyped collection of `T'`           |                                       |
|----------------------------------|--------------------------------------|---------------------------------------|
| `std::[unordered_]map<K,V>`      | `std::[unordered_]map<K',V'>`        |                                       |
|----------------------------------|--------------------------------------|---------------------------------------|
| `std::[unordered_]multimap<K,V>` | `ROOT::RVec<T>`                      | only `T` = `std::[pair,tuple]<K,V>`   |
|                                  | `std::vector<T>`                     | only `T` = `std::[pair,tuple]<K,V>`   |
|                                  | `std::array<T, N>`                   | only `T` = `std::[pair,tuple]<K,V>`   |
|                                  | `std::[unordered_][multi]set<T>`     | only `T` = `std::[pair,tuple]<K,V>`   |
|                                  | `std::[unordered_][multi]map<K',V'>` |                                       |
|                                  | User-defined collection of `T`       | only `T` = `std::[pair,tuple]<K,V>`   |
|                                  | Untyped collection of `T`            | only `T` = `std::[pair,tuple]<K,V>`   |

#### Fixed-size collections

There is no special automatic evolution for fixed-length collections (`std::array<...>`, `std::bitset<...>`).
The length of the array must not change and there is no automatic transformation from variable-length to
fixed-length collections.
C style arrays and `std::array<...>` of the same type and length can be used interchangibly.

#### Nullable fields

| In-memory type       | Compatible on-disk types |
| -------------------- | ------------------------ |
| `std::optional<T>`   | `std::unique_ptr<T'>`    |
|                      | `T'`                     |
|----------------------|--------------------------|
| `std::unique_ptr<T>` | `std::optional<T'>`      |
|                      | `T'`                     |

#### Records

| In-memory type              | Compatible on-disk types               |
| --------------------------- | -------------------------------------- |
| `std::pair<T,U>`            | `std::tuple<T',U'>`                    |
|-----------------------------|----------------------------------------|
| `std::tuple<T,U>`           | `std::pair<T',U'>`                     |
|-----------------------------|----------------------------------------|
| Untyped record              | User-defined class of compatible shape |

Note that for emulated classes, the in-memory untyped record is constructed from on-disk information.

#### Additional rules

All on-disk types `std::atomic<T'>` can be read into a `T` in-memory model.

If a class property changes from using an RNTuple streamer field to a using regular RNTuple class field,
existing files with on-disk streamer fields will continue to read as streamer fields.
This can be seen as "schema evolution out of streamer fields".

## Manual schema evolution (I/O customization rules)

ROOT I/O customization rules allow for custom code handling the transformation
from the on-disk schema to the in-memory model.
Customization rules are part of the class dictionary.
For the exact syntax of customization rules, please refer to the [ROOT manual](https://root.cern/manual/io/#dealing-with-changes-in-class-layouts-schema-evolution).

Generally, customization rules consist of
  - A target class.
  - Target members of the target class, i.e. those class members whose value is set by the rule.
    Target members must be direct members, i.e. not part of a base class.
  - A source class (possibly having a different class name than the target class)
    together with class versions or class checksums
    that describe all the possible on-disk class versions the rule applies to.
    Note that the class checksum can be retrieved, e.g., from `TClass::GetCheckSum()`.
  - Source members of the source class; the given source members will be read as the given type.
    The source member will undergo schema evolution before being passed to the rule's function.
    Source members can also be from a base class.
    Note that there is no way to specify a base class member that has the same name as a member in the derived class.
  - The custom code snippet; the code snippet has access to the (whole) target object and to the given source members.

For illustration purposes, here is a concrete example of a customization rule
```
#pragma read \
  targetClass = "Coordinates"\
  target = "fPhi,fR" \
  sourceClass = "Coordinates" \
  version = "[3]" \
  source = "float fX; float fY" \
  include = "cmath" \
  code = "{ fR = sqrt(onfile.fX * onfile.fX + onfile.fY * onfile.fY); fPhi = atan2(onfile.fY, onfile.fX); }"
```

At runtime, for any given target member there must be at most be one applicable rule.
A source member can be read into any type compatible to its on-disk type
but any given source member can only be read into one type for a given target class
(i.e. multiple rules for the same target/source class must not use different types for the same source member).

There are two special types of rules
  1. Pure class rename rules consisting only of source and target class
  2. Whole-object rules that have no target members

Class rename rules (pure or not) are not transitive
(if in-memory `A` can read from on-disk `B` and in-memory `B` can read from no-disk `C`,
in-memory `A` can not automatically read from on-disk `C`).

Note that customization rules operate on partially read objects.
Customization rules are executed after all members not subject to customization rules have been read from disk.
Whole-object rules are executed after other rules.
Otherwise, the scheduling of rules is unspecified.

## Interplay between automatic and manual schema evolution

The target members of I/O customization rules are exempt from automatic schema evolution
(applies to the corresponding field of the target member and all its subfields).
Otherwise, automatic and manual schema evolution work side by side.
For instance, a renamed class is still subject to automatic schema evolution.

The source member of a customization rule is subject to the same automatic and manual schema evolution rules
as if it was normally read, e.g. in an `RNTupleView`.

## Schema evolution differences between RNTuple and Classic I/O

In contrast to RNTuple, TTree and TFile apply also the following automatic schema evolution rules
  - Conversion between floating point and integer types
  - Conversion from `unique_ptr<T>` --> `T'`
  - Complete conversion matrix of all collection types
  - Insertion and removal of intermediate classes
  - Move of a member between base class and derived class
  - Reordering of base classes

