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

(*NOTE: currently there is no guarantee for the user about which mode will be used to generate the merged RNTuple. At the moment, L0 and L4 are never used; L1 is used when possible, otherwise L3 is used. Note that we currently don't use L2 because in general when recompressing we might need to change encoding. Improvements on this front are possible and in principle we should be able to use L2 when the encoding doesn't need to change.*)

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
* P2: all pages in the same output column have the **same encoding** (which may be different from the inputs' encoding);
* P3: the output clusters are **the same as the input clusters**;
* P4: the output RNTuple **always has 1 cluster group**

Note that these properties influence and are influenced by the level of merging used.
E.g. P1 and P2 are currently true because we only support L1 merging of pages with identical compressions. This is a limitation that we intend to lift at some point (both for L1 and L0 if we ever support it).
P3 and P4 would not necessarily be true with L4 support (which might be desirable in some cases, e.g. to group pages into smaller/larger clusters).

Therefore we *will* want to drop these properties at some point, in order to improve the capabilities of the Merger.

## High-level description
The merging process requires at least 1 input, in the form of an `RPageSource`.

The first input is attached in `EDescriptorDeserializeMode::kForWriting` mode, which doesn't collate the extended header with the non-extended header. Since we use the first input's descriptor as the output schema (barring late model extensions, see later), opening in `kForWriting` mode allows us to write the output to disk while preserving the non-extended schema description as per requirement R3. A consequence of this choice is that the merger never produces (new) deferred columns in the output RNTuple's header.

In `Union` mode only, we allow any subsequent input RNTuple to define new fields that don't appear in the first input. These fields, after being validated, are late model extended into the output model and will thus appear in the output RNTuple's extended header on disk. This means that all columns that were not part of the first input's schema become deferred columns in the output RNTuple (unless the first source had 0 entries).

## Descriptor compatibility and validation
Whenever a new input is processed, we compare its descriptor with the output descriptor to verify that merging is possible.

The comparison function does 3 main things:
- collect all "extra destination fields" (i.e. fields that exist in the output but not in this input RNTuple)
- collect all "extra source fields" from the input RNTuple
- collect and validate all common fields.

If the Merging Mode is set to **Filter** we require the "extra destination fields" list to be empty.
If the Merging Mode is set to **Strict** we require both the "extra destination fields" and "extra source fields" lists to be empty.
If the Merging Mode is set to **Union**, the "extra source fields" list is used to late model extend the destination model.

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
