# RNTuple merging process

This is an informal document acting as a reference and high-level explanation of merging of RNTuple as implemented in the `RNTupleMerger` class.
Please note that the RNTupleMerger is currently experimental and the content of this document is subject to change.

## Glossary

* **Incremental Merging**: merging without re-reading or re-writing the payload of an existing RNTuple
* "Levels" of merging:
    * **L0 Merge** (*copyless* merge): merge a page without copying it (needs filesystem support)
    * **L1 Merge** (*fast* merge, or *copy* merge): merge a page by just copying it (no recompressing, no resealing)
    * **L2 Merge** (*recompress* merge): merge a page by recompressing it (no resealing - which implies no re-encoding)
    * **L3 Merge** (*reseal* merge): merge a page by resealing/repacking it (implies decompressing/recompressing, may change the encoding)
    * **L4 Merge** (*slow* merge): merge a page by re-reading and re-writing all values, potentially changing the pages' boundaries and clustering (implies resealing and recompressing)

Currently there is no guarantee for the user about which mode will be used to generate the merged RNTuple.
At the moment, this is how it works:
- if the compression of the target column match that of the source column, L1 is used;
- otherwise, L2 is used.

L0, L3 and L4 are currently never used.

**NOTE**: prior to ROOT 6.42, if two columns had the same compression but different encoding they would undergo L3 merging (implying a recompression and resealing);
from 6.42 onwards the RNTupleMerger will instead attach a new column to the parent field as a new representation and L1-merge them.

## Goal
The goal of the RNTuple merging process is producing one output RNTuple from *N* input RNTuples that can be used as if it were produced directly in the merged state. This means that:

* R1: All fields in the output RNTuple are accessible and have a type compatible<sup>1</sup> with the original fields of the input RNTuples.
* R2: The values of those fields are a concatenation of the original fields. If the first input RNTuple had *M* entries, the first *M* entries of the output RNTuple map to those entries; entry *M+1* maps to the first entry of the second input RNTuple, and so on.

<sup>1</sup>: currently "compatible" means "identical". This may be extended in the future to include fields that have convertible types.

At a lower level, we require that:

* R3: the output RNTuple has the **same non-extended schema description** as the **first input RNTuple**;
* R4: the output RNTuple has an **extended schema description** that is:
    *  **strictly equal** to the first input (`Filter` and `Strict` mode), or
    *  **a superset** of the first input (`Union` mode)

Consequences of R3 and R4:

* all columns that were deferred in the first RNTuple remain deferred in the output RNTuple and have the same FirstElementIdx.
* all projected fields in the input RNTuples remain projected in the output and have the same source field.
 
The following properties are currently true but they are subject to change:

* P1: all output pages have the **same compression** (which may be different from the input pages' compression);
* P2: the output clusters are **the same as the input clusters**;
* P3: the output RNTuple **always has 1 cluster group**

Note that these properties influence and are influenced by the level of merging used. 
E.g. P1 is currently true because we only support L1 merging of pages with identical compressions. This is a limitation that we intend to lift at some point (both for L1 and L0 if we ever support it).
P2 and P3 would not necessarily be true with L4 support (which might be desirable in some cases, e.g. to group pages into smaller/larger clusters).

Also note that the output pages coming from matching columns of a field may use mixed encodings.

Therefore we *will* want to drop at least some of these properties at some point, in order to improve the capabilities of the Merger.

## High-level description
The merging process requires at least 1 input, in the form of an `RPageSource`.

The first input is attached in `EDescriptorDeserializeMode::kForWriting` mode, which doesn't collate the extended header with the non-extended header. Since we use the first input's descriptor as the output schema (barring late model extensions, see later), opening in `kForWriting` mode allows us to write the output to disk while preserving the non-extended schema description as per requirement R3. A consequence of this choice is that the merger never produces (new) deferred columns in the output RNTuple's header.

In `Union` mode only, we allow any subsequent input RNTuple to define new fields that don't appear in the first input. These fields, after being validated, are late model extended into the output model and will thus appear in the output RNTuple's extended header on disk. This means that all columns that were not part of the first input's schema become deferred columns in the output RNTuple (unless the first source had 0 entries).

## Descriptor compatibility and validation
Whenever a new input is processed, we compare its descriptor with the output descriptor to verify that merging is possible.

The comparison function does 4 main things:
- collect all "extra destination fields" (i.e. fields that exist in the output but not in this input RNTuple)
- collect all "extra source fields" from the input RNTuple
- collect and validate all common fields
- collect all columns that need to be extended with additional representations.

If the merging mode is set to **Filter** we require the "extra destination fields" list to be empty.
If the merging mode is set to **Strict** we require both the "extra destination fields" and "extra source fields" lists to be empty.
If the merging mode is set to **Union**, the "extra source fields" list is used to late model extend the destination model.

As for common fields, they are matched by name and validated as follows:
- any field that is projected in the destination must be also projected in the source and must be projected to the same field;
- any field that is not projected in the destination must also not be projected in the source;
- the field types names must be **identical** (*this could probably be relaxed in the future to allow for different but compatible types - see requirement R1*)
- the type checksums, if present, must be identical. Note that if a field has a type checksum and the other doesn't, we consider this valid (*is this sound?*);
- the type versions must be identical;
- the fields' structural roles must be identical;
- the column representations must match<sup>1</sup>, as follows:
    - the source and destination fields must have the same number of columns;
    - the types of each column must either be identical or one must be the split/unsplit version of the other;
    - the bits on storage of both columns must be identical;
    - the value range of both columns must be identical;
    - the representation index of the each source column must be 0 (i.e. we currently don't support multiple columns representations while merging);
- if the fields have subfields, the number of subfields must be identical, and each source subfield is recursively validated against its destination counterpart via all the rules described in this list.


<sup>1</sup>: these restrictions will likely not be required for L4 merging.

## Column representation extension
In all merging modes, we allow new column representations to be attached to the source fields. This is done to allow for L1 merging of columns with different encodings, which would otherwise require recompressing.
These new column representations are added to the output RNTuple's footer and become part of its Schema Extension section. Note that in general these columns will be added as deferred *and* suppressed.

**Technical note**: this is *not* done via the regular late model extension API, but uses internal functionality.

We add new (physical) column representations in the following cases:

- when one or more columns of a field has a different type than its matching counterpart in the destination RNTuple;
- when one or more columns of a field has the same type but different metadata than its matching counterpart in the destination RNTuple (e.g. in case of a Real32Quant column, different bit width or value range).

Whenever we extend a physical column that is referred to by one or more alias columns in some projected fields, we also add a corresponding new alias column in those fields.

#### Example
Suppose we merge source RNTuples **S1** and **S2**, each with the following fields:

1. `foo` of type `int`
1. `fooProj` projecting onto field `foo`

Suppose that S1 is compressed and thus its `foo` field is represented by a column of type `kSplitInt32`, whereas S2 is uncompressed and its `foo` field is represented by a column `kInt32`.
When merging S1 and S2 we collate those two representations under the same field `foo`, so that it will now have representatives: `{kSplitInt32, kInt32}`.
At the same time, we add a second alias column to the field `fooProj`, which will now have its first column aliasing the `kSplitInt32` column (column 0 of field `foo`) and its second one aliasing the `kInt32` one (column 1 of field `foo`).

