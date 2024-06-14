# RNTuple Reference Specifications 0.2.3.0

**Note:** This is work in progress. The RNTuple specification is not yet finalized.

## Versioning Notes

The RNTuple binary format version is inspired by semantic versioning.
It uses the following scheme: EPOCH.MAJOR.MINOR.PATCH

_Epoch_: an increment of the epoch indicates backwards-incompatible changes.
The RNTuple pre-release has epoch 0.
The fist public release will get epoch 1.
There is currently no further epoch foreseen.

_Major_: an increment of the major version indicates forward-incompatible changes.
A forward-incompatible change is known to break reading in previous software versions that do not support that feature.
The use of new, forward-incompatible features must be indicated in the feature flag in the header (see below).
For the RNTuple pre-release (epoch == 0), the major version is the release candidate number.

_Minor_: an increment of the minor version indicates new, optional format features.
Such optional features, although unknown to previous software versions,
won't prevent those software versions from properly reading the file.
Old readers will safely ignore these features.

_Patch_: an increment of the patch version indicates backported features from newer format versions.
The backported features may correspond to a major or a minor release.

Except for the epoch, the versioning is for reporting only.
Readers should use the feature flag in the header to determine whether they support reading the file.

## Introduction

The RNTuple binary format describes the serialized, on-disk representation of an RNTuple data set.
The data on disk is organized in **pages** (typically 10-100kB in size)
and several **envelopes** that contain information about the data such as header and footer.
The RNTuple format specifies the binary layout of the pages and the envelopes.

Pages and envelopes are meant to be embedded in a data container
such as a ROOT file or a set of objects in an object store.
Envelopes can reference other envelopes and pages by means of a **locator** or an **envelope link**;
for a file embedding, the locator consists of an offset and a size.
The RNTuple format does _not_ establish a specific order of pages and envelopes.

For the ROOT file embedding, pages and envelopes are stored in "invisible", non-indexed **RBlob** keys.
The RNTuple format does _not_ establish a semantic mapping from objects to keys or vice versa.
For example, one key may hold a single page or a number of pages of the same cluster.
The only relevant means of finding objects is the locator information, consisting of an offset and a size.

Every embedding must define an **anchor** that contains the format version supported by the writer,
and envelope links (location, compressed and uncompressed size) of the header and footer envelopes.
For the ROOT file embedding, the **ROOT::Experimental::RNTuple** object acts as an anchor.

### Anchor schema

The current (class version 5) **ROOT::Experimental::RNTuple** object has the following schema:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|        Version Epoch          |         Version Major         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|        Version Minor          |         Version Patch         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
+                         Seek Header                           +
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
+                        Nbytes Header                          +
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
+                         Len Header                            +
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
+                         Seek Footer                           +
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
+                        Nbytes Footer                          +
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
+                         Len Footer                            +
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
+                        Max Key Size                           +
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

```

When serialized to disk, a 64 bit checksum is appended to the anchor, calculated as the XXH3 hash of
all the (serialized) fields of the anchor object.

Note that, since the anchor is serialized as a "classic" TFile key, all integers in the anchor, as well
as the checksum, are encoded in **big-endian**, unlike the RNTuple payload which is encoded in little-endian.

The anchor may evolve in future versions only by appending new fields to the existing schema, but
fields will not be removed, renamed or reordered.


## Compression Block

RNTuple envelopes and pages are wrapped in compression blocks.
In order to deserialize a page or an envelope, its compressed and uncompressed size needs to be known.

If the compressed size == uncompressed size, the data is stored unmodified in uncompressed form.
Otherwise, data is represented as a series of compressed chunks.
Each chunk is prepended with the following 9 bytes header.

```
Byte
0      1      2      3      4      5      6      7      8      9
+------+------+------+------+------+------+------+------+------+...
|     Algorithm      |  Compressed size   |  Uncompressed size | <COMPRESSED DATA>
+------+------+------+------+------+------+------+------+------+...
```

_Algorithm_: Identifies the compression algorithm used to compress the data. This can take one of the following values

| Algorithm                | Meaning                                      |
|--------------------------|----------------------------------------------|
| 'Z' 'L' '\x08'           | zlib                                         |
| 'C' 'S' '\x08'           | Old Jean-loup Gailly's deflation algorithm   |
| 'X' 'Z' '\x00'           | LZMA                                         |
| 'L' '4' <VERSION_MAJOR>  | LZ4; third byte encodes major version number |
| 'Z' 'S' '\x01'           | Zstd                                         |

_Compressed size_: An unsigned, little-endian integer that indicates the compressed size of the data that follows the header.

_Uncompressed size_: An unsigned, little-endian integer that indicates the uncompressed size of the data that follows.
The maximum representable value is $(2^{24})-1$, i.e. 16777215, and thus each compressed chunk can represent up to 16 MiB of uncompressed data.
If the original data is larger than this value, more compressed chunks will follow.

## Basic Types

Data stored in envelopes is encoded using the following type system.
Note that this type system is independent (and different) from the regular ROOT serialization.

_Integer_: Integers are encoded in two's complement, little-endian format.
They can be signed or unsigned and have lengths up to 64bit.

_String_: A string is stored as a 32bit unsigned integer indicating the length of the string
followed by the characters.
Strings are ASCII encoded; every character is a signed 8bit integer.

_Compression settings_: A 32bit integer containing both a compression algorithm and the compression level.
The compression settings are encoded according to this formula: $settings = algorithm * 100 + level$.
See Compression.[h/cxx] for details and available algorithms.

The meta-data envelope defines additional basic types (see below).


### Feature Flags

Feature flags are 64bit integers where every bit represents a certain forward-incompatible feature that is used
in the binary format of the RNTuple at hand (see Versioning Notes).
The most significant bit is used to indicate that there are more than 63 features to specify.
That means that readers need to continue reading feature flags as long as their signed integer value is negative.

Readers should gracefully abort reading when they encounter unknown bits set.

The following feature bits are defined:

| Bit                                | Feature              |
------------------------------------ |----------------------|
| 137 (0x09 of the 3rd feature int)  | Reserved for testing |


## Frames

RNTuple envelopes can store records and lists of basic types and other records or lists by means of **frames**.

A frame has the following format
```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
+                             Size                            +-+
|                                                             |T|
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|              Number of Items (for list frames)                |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         FRAME PAYLOAD                         |
|                              ...                              |
```

_Size_: The absolute value gives the size in bytes of the frame and the payload.

_T(ype)_: Can be either 0 for a **record frame** or 1 for a **list frame**.
The type should be interpreted as the sign bit of the size, i.e. negative sizes indicate list frames.

_Number of items_: Only used for list frames to indicate the length of the list in the frame payload.

File format readers should use the size provided in the frame to seek to the data that follows a frame
instead of summing up the sizes of the elements in the frame.
This approach ensures that frames can be extended in future file format versions
without breaking the deserialization of older readers.


## Locators and Envelope Links

A locator is a generalized way to specify a certain byte range on the storage medium.
For disk-based storage, the locator is just byte offset and byte size.
For other storage systems, the locator contains enough information to retrieve the referenced block, e.g. in object stores, the locator can specify a certain object ID.
The locator has the following format

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                             Size                            |T|
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
+                             Offset                            +
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

_Size_: If `T` is zero, the number of bytes to read, i.e. the compressed size of the referenced block.
Otherwise the 16 least-significant bits, i.e bits 0:15, specify the size of the locator itself (see below).

_T(ype)_: Zero for a simple on-disk or in-file locator, 1 otherwise.
Can be interpreted as the sign bit of the size, i.e. negative sizes indicate non-disk locators.
In this case, the locator should be interpreted like a frame, i.e. size indicates the _size of the locator itself_.

_Offset_:
For on-disk / in-file locators, the 64bit byte offset of the referenced byte range counted from the start of the file.

For non-disk locators, i.e. `T` == 1, the locator format is as follows

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|              Size             |   Reserved    |     Type    |T|
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                        LOCATOR PAYLOAD                        |
|                              ...                              |
```

In this case, the last 8 bits of the size should be interpreted as a locator type.
To determine the locator type, the absolute value of the 8bit integer should be taken.
The type can take one of the following values

| Type | Meaning      | Payload format     |
|------|--------------|--------------------|
| 0x01 | URI string   | [ASCII characters] |
| 0x02 | DAOS locator | Object64           |

The range 0x03 - 0x7f is currently unused. Additional types can be registered in the future.
For URI locators, the locator contains the ASCII characters of the URI following the size and the type.
Each locator type follows a given format for the payload (see Section "Well-known payload formats" below).

_Reserved_ is an 8bit field that can be used by the storage backend corresponding to the type in order to store additional information about the locator.

An envelope link consists of a 64bit unsigned integer that specifies the uncompressed size of the envelope
followed by a locator.

### Well-known Payload Formats

This section describes the well-known payload formats used in non-disk locators.
Note that locators having a different value for _Type_ may share a given payload format (see the table above).

- _Object64_: Targets object storage systems in which 64bit suffice to locate a specific object. The payload has the following format
```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                          Content size                         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
+                            Location                           +
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

_Content size_: the number of bytes to read, i.e. the compressed size of the referenced block.

_Location_: 64bit object address; its specific use depends on the object store.
In particular, it might contain a partial address that can be qualified using some other information depending on the storage backend, e.g. a URL might be generated based on this value.


## Envelopes

An Envelope is a data block containing information that describes the RNTuple data.
The following envelope types exist

| Type              |  ID  | Contents                                                          |
|-------------------|------|-------------------------------------------------------------------|
| Header            | 0x01 | RNTuple schema: field and column types                            |
| Footer            | 0x02 | Description of clusters, location of user meta-data               |
| Page list         | 0x03 | Location of data pages                                            |
| User meta-data    | 0x04 | Key-value pairs of additional information about the data          |

Envelopes have the following format

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|        Envelope Type ID       |                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+        Envelope Length        +
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
                         ENVELOPE PAYLOAD
                               ...
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
+                            XxHash-3                           +
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

_Envelope type ID_: As specified in the table above,
encoded in the least significant 16 bits of the first 64bit integer

_Envelope length: Uncompressed size of the envelope,
encoded in the 48 most significant bits of the first 64bit integer

_XxHash-3_: Checksum of the envelope and the payload bytes together

Note that the compressed size (and also the length) of envelopes is given by the RNTuple anchor (header, footer)
or by a locator that references the envelope.


### Header Envelope

The header consists of the following elements:

 - Feature flag
 - String: name of the ntuple
 - String: description of the ntuple
 - String: identifier of the library or program that writes the data
 - List frame: list of field record frames
 - List frame: list of column record frames
 - List frame: list of alias column record frames
 - List frame: list of extra type information

The last four list frames containing information about fields and columns are collectively referred to as _schema description_.

#### Field Description

Every field record frame of the list of fields has the following contents

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         Field Version                         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                          Type Version                         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
+                        Parent Field ID                        +
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|        Structural Role        |             Flags             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

The field version and type version are used for schema evolution.

If `flags=0x0001` (_repetitive field_) is set, the field represents a fixed-size array.
In this case, an additional 64bit integer follows immediately that specifies the size of the array.
Typically, another (sub) field with `Parent Field ID` equal to the ID of this field
is expected to be found, representing the array content
(see Section "Mapping of C++ Types to Fields and Columns").

The block of integers is followed by a list of strings:

- String: field name
- String: type name
- String: type alias
- String: field description

The order of fields matters: every field gets an implicit field ID
which is equal the zero-based index of the field in the serialized list;
subfields are ordered from smaller IDs to larger IDs.
Top-level fields have their own field ID set as parent ID.

The flags field can have one of the following bits set

| Bit      | Meaning                                                                    |
|----------|----------------------------------------------------------------------------|
| 0x01     | Repetitive field, i.e. for every entry $n$ copies of the field are stored  |

The structural role of the field can have on of the following values

| Value    | Structural role                                                          |
|----------|--------------------------------------------------------------------------|
| 0x00     | Leaf field in the schema tree                                            |
| 0x01     | The field is the mother of a collection (e.g., a vector)                 |
| 0x02     | The field is the mother of a record (e.g., a struct)                     |
| 0x03     | The field is the mother of a variant                                     |
| 0x04     | The field represents an unsplit object serialized with the ROOT streamer |


#### Column Description

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|              Type             |        Bits on Storage        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
+                            Field ID                           +
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                             Flags                             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

The order of columns matter: every column gets an implicit column ID
which is equal to the zero-based index of the column in the serialized list.
Multiple columns attached to the same field should be attached from smaller to larger IDs.

The column type and bits on storage integers can have one of the following values

| Type | Bits | Name         | Contents                                                                      |
|------|------|--------------|-------------------------------------------------------------------------------|
| 0x01 |   64 | Index64      | Mother columns of (nested) collections, counting is relative to the cluster   |
| 0x02 |   32 | Index32      | Mother columns of (nested) collections, counting is relative to the cluster   |
| 0x03 |   96 | Switch       | Tuple of a kIndex64 value followed by a 32 bits dispatch tag to a column ID   |
| 0x04 |    8 | Byte         | An uninterpreted byte, e.g. part of a blob                                    |
| 0x05 |    8 | Char         | ASCII character                                                               |
| 0x06 |    1 | Bit          | Boolean value                                                                 |
| 0x07 |   64 | Real64       | IEEE-754 double precision float                                               |
| 0x08 |   32 | Real32       | IEEE-754 single precision float                                               |
| 0x09 |   16 | Real16       | IEEE-754 half precision float                                                 |
| 0x16 |   64 | Int64        | Two's complement, little-endian 8 byte signed integer                         |
| 0x0A |   64 | UInt64       | Little-endian 8 byte unsigned integer                                         |
| 0x17 |   32 | Int32        | Two's complement, little-endian 4 byte signed integer                         |
| 0x0B |   32 | UInt32       | Little-endian 4 byte unsigned integer                                         |
| 0x18 |   16 | Int16        | Two's complement, little-endian 2 byte signed integer                         |
| 0x0C |   16 | UInt16       | Little-endian 2 byte unsigned integer                                         |
| 0x19 |    8 | Int8         | Two's complement, 1 byte signed integer                                       |
| 0x0D |    8 | UInt8        | 1 byte unsigned integer                                                       |
| 0x0E |   64 | SplitIndex64 | Like Index64 but pages are stored in split + delta encoding                   |
| 0x0F |   32 | SplitIndex32 | Like Index32 but pages are stored in split + delta encoding                   |
| 0x10 |   64 | SplitReal64  | Like Real64 but in split encoding                                             |
| 0x11 |   32 | SplitReal32  | Like Real32 but in split encoding                                             |
| 0x12 |   16 | SplitReal16  | Like Real16 but in split encoding                                             |
| 0x1A |   64 | SplitInt64   | Like Int64 but in split + zigzag encoding                                     |
| 0x13 |   64 | SplitUInt64  | Like UInt64 but in split encoding                                             |
| 0x1B |   64 | SplitInt32   | Like Int32 but in split + zigzag encoding                                     |
| 0x14 |   32 | SplitUInt32  | Like UInt32 but in split encoding                                             |
| 0x1C |   16 | SplitInt16   | Like Int16 but in split + zigzag encoding                                     |
| 0x15 |   16 | SplitUInt16  | Like UInt16 but in split encoding                                             |

The "split encoding" columns apply a byte transformation encoding to all pages of that column
and in addition, depending on the column type, delta or zigzag encoding:

Split (only)
: Rearranges the bytes of elements: All the first bytes first, then all the second bytes, etc.

Delta + split
: The first element is stored unmodified, all other elements store the delta to the previous element.
  Followed by split encoding.

Zigzag + split
: Used on signed integers only; it maps $x$ to $2x$ if $x$ is positive and to $-(2x+1)$ if $x$ is negative.
  Followed by split encoding.

**Note**: these encodings always happen within each page, thus decoding should be done page-wise,
not cluster-wise.

Future versions of the file format may introduce additional column types
without changing the minimum version of the header.
Old readers need to ignore these columns and fields constructed from such columns.
Old readers can, however, figure out the number of elements stored in such unknown columns.

The flags field can have one of the following bits set

| Bit      | Meaning                                                      |
|----------|--------------------------------------------------------------|
| 0x01     | Elements in the column are sorted (monotonically increasing) |
| 0x02     | Elements in the column are sorted (monotonically decreasing) |
| 0x04     | Elements have only non-negative values                       |
| 0x08     | Index of first element in the column is not zero             |

If flag 0x08 (deferred column) is set, the index of the first element in this column is not zero, which happens if the column is added at a later point during write.
In this case, an additional 64bit integer containing the first element index follows the flags field.
Compliant implementations should yield synthetic data pages made up of 0x00 bytes when trying to read back elements in the range $[0, firstElementIndex-1]$.
This results in zero-initialized values in the aforementioned range for fields of any supported C++ type, including `std::variant<Ts...>` and collections such as `std::vector<T>`.
The leading zero pages of deferred columns are _not_ part of the page list, i.e. they have no page locator.
In practice, deferred columns only appear in the schema extension record frame (see Section Footer Envelope).

#### Alias columns

An alias column has the following format

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
+                      Physical Column ID                       +
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                           Field ID                            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```
Alias columns do not have associated data pages.  Instead, their data comes from another column referred to below as "physical column".
The first 32bit integer references the physical column ID.
The second 32bit integer references the associated "projected" field.
A projected field is a field using alias columns to present available data by an alternative C++ type.
The ID of the alias column itself is given implicitly by the serialization order.
In particular, alias columns have larger IDs than physical columns.
In the footer and page list envelopes, only physical column IDs must be referenced.


#### Extra type information

Certain field types may come with additional information required, e.g., for schema evolution.
The type information record frame has the following contents

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
+                       Content Identifier                      +
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                        Type Version From                      |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         Type Version To                       |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

followed by a string containing the type name.

The combination of type version from/to, type name, and content identifier should be unique in the list.
However, not every type needs to provide additional type information.

The following kinds of content are supported:

| Content identifier  | Meaning of content                                  |
|---------------------|-----------------------------------------------------|
| 0x00                | Serialized ROOT streamer info; see notes            |

The serialized ROOT streamer info is not bound to a specific type.
It is the combined streamer information from all the unsplit fields.
Writers set version from/to to zero and use an empty type name.
Readers should ignore the type-specific information.
The format of the content is a ROOT streamed TList of TStreamerInfo objects.

### Footer Envelope

The footer envelope has the following structure:

- Feature flags
- Header checksum (XxHash-3 64bit)
- Schema extension record frame
- List frame of column group record frames
- List frame of cluster group record frames
- List frame of meta-data block envelope links

The header checksum can be used to cross-check that header and footer belong together.
The meaning of the feature flags is the same as for the header.
The header flags do not need to be repeated.
Readers should combine (logical `or` of the bits) the feature flags from header and footer for the complete set of flags.

#### Schema Extension Record Frame

The schema extension record frame contains an additional schema description that is incremental with respect to the schema contained in the header (see Section Header Envelope). Specifically, it is a record frame with the following four fields (identical to the last four fields in Header Envelope):

 - List frame: list of field record frames
 - List frame: list of column record frames
 - List frame: list of alias column record frames
 - List frame: list of extra type information


In general, a schema extension is optional and thus this record frame might be empty.
The interpretation of the information contained therein should be identical as if it was found directly at the end of the header.
This is necessary when fields have been added during writing.

#### Column Group Record Frame
The column group record frame is used to set IDs for certain subsets of column IDs.
Column groups are only used when there are sharded clusters.
Otherwise, the enclosing list frame in the footer envelope is empty and all clusters span all columns.
The purpose of column groups is to prevent repetition of column ID ranges in cluster summaries.

The column group record frame consists of a list frame of 32bit integer items.
Every item denotes a column ID that is part of this particular column group.
The ID of the column group is given implicitly by the order of column groups.

The frame hierarchy is as follows

    - Column group outer list frame
    |
    |---- Column group 1 record frame
    |     |---- List frame of column IDs
    |     |     |---- Column ID 1 [32bit integer]
    |     |     |---- Column ID 2 [32bit integer]
    |     |     | ...
    |
    |---- Column group 2 record frame
    | ...


#### Cluster Group Record Frame

The cluster group record frame references the page list envelopes for groups of clusters.
A cluster group record frame starts with

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
+                      Minimum Entry Number                     +
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
+                          Entry Span                           +
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                       Number of clusters                      |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```
Followed by the page list envelope link.

To compute the minimum entry number, take first entry number from all clusters in the cluster group,
and take the minimum among these numbers.
The entry span is the number of entries that are (partially for sharded clusters) covered by this cluster group.
The entry range allows for finding the right page list for random access requests to entries.
The number of clusters information allows for using consistent cluster IDs even if cluster groups are accessed non-sequentially.

### Page List Envelope

The page list envelope contains cluster summaries and page locations.
It has the following structure

  - Header checksum (XxHash-3 64bit)
  - List frame of cluster summary record frames
  - Nested list frame of page locations

#### Cluster Summary Record Frame
The cluster summary record frame contains the entry range of a cluster:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
+                       First Entry Number                      +
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                       Number of Entries                       |
+                                                       +-+-+-+-+
|                                                       | Flags |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

If flag 0x01 (sharded cluster) is set,
an additional 32bit integer containing the column group ID follows the flags field.
If flags is zero, the cluster stores the event range of _all_ the original columns
_including_ the columns from extension headers.

The order of the cluster summaries defines the cluster IDs,
starting from the first cluster ID of the cluster group that corresponds to the page list.

#### Page Locations

The page locations are stored in a nested list frame as follows.
A top-most list frame where every item corresponds to a cluster.
The order of items corresponds to the cluster IDs as defined by the cluster groups and cluster summaries.

Every item of the top-most list frame consists of an outer list frame where every item corresponds to a column.
Every item of the outer list frame is an inner list frame
whose items correspond to the pages of the column in the cluster.
The inner list is followed by a 64bit unsigned integer element offset and the 32bit compression settings (see Section "Basic Types").
Note that the size of the inner list frame includes the element offset and compression settings.
The order of the outer items must match the order of the columns as specified in the cluster summary and column groups.
For a complete cluster (covering all original columns), the order is given by the column IDs (small to large).

The order of the inner items must match the order of pages' resp. elements.
Every inner item (that describes a page) has the following structure:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                     Number of Elements                      |C|
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

Followed by a locator for the page.
_C(hecksum)_: If set, an XxHash-3 64bit checksum of the uncompressed page data is stored just after the page.
This bit should be interpreted as the sign bit of the size, i.e. negative values indicate pages with checksums.

Depending on the number of pages per column per cluster, every page induces
a total of 28-36 Bytes of data to be stored in the page list envelope.
For typical page sizes, that should be < 1 per mille.

Note that we do not need to store the uncompressed size of the page
because the uncompressed size is given by the number of elements in the page and the element size.
We do need, however, the per-column and per-cluster element offset in order to read a certain event range
without inspecting the meta-data of all the previous clusters.

The hierarchical structure of the frames in the page list envelope is as follows:

    # this is `List frame of cluster group record frames` mentioned above
    - Top-most cluster list frame (one item for each cluster in this RNTuple)
    |
    |---- Cluster 1 column list frame (outer list frame, one item for each column in this RNTuple)
    |     |---- Column 1 page list frame (inner list frame, one item for each page in this column)
    |     |     |---- Page 1 description (inner item)
    |     |     |---- Page 2 description (inner item)
    |     |     | ...
    |     |---- Column 1 element offset (UInt64)
    |     |---- Column 1 compression settings (UInt32)
    |     |---- Column 2 page list frame
    |     | ...
    |
    |---- Cluster 2 column list frame
    | ...

In order to save space, the page descriptions (inner items) are _not_ in a record frame.
If at a later point more information per page is needed,
the page list envelope can be extended by addtional list and record frames.

### User Meta-data Envelope

User-defined meta-data can be attached to an ntuple.
These meta-data are key-value pairs.
The key is a string.
The value can be of type integer, double, string, or a list thereof.

Keys are scoped with the different namespace parts separated by a dot (`.`).
The `ROOT.` namespace prefix is reserved for the ROOT internal meta-data.
Meta-data are versioned: the same key can appear multiple times with different values.
This is interpreted as different versions of the meta-data.

The meta-data envelope consists of a single collection frame with an item for every key-value pair.
Every key-value pair is a record frame with the following contents:

- Type: 32bit integer
- String: key

Followed by the value.
The format of the value depends on the type, which can be one of the following list

| Type |  Contents                                |
|------|------------------------------------------|
| 0x01 | 64bit integer                            |
| 0x02 | bool (stored as 8bit integer)            |
| 0x03 | IEEE-754 double precision floating point |
| 0x04 | String                                   |

If the most significant bit of the type is set (i.e., the type has a negative value),
the value is a list of the type given by the absolute value of the type field.
The list is stored as a list frame.

Future versions of the file format may introduce additional meta-data types
without setting a feature flag.
Old readers need to ignore these key-value pairs.

Key versioning starts with zero.
The version is given by the order of serialization within a meta-data envelope
and by the order of meta-data envelope links in the footer.

## Mapping of C++ Types to Fields and Columns

This section is a comprehensive list of the C++ types with RNTuple I/O support.
Within the supported type system complex types can be freely composed,
e.g. `std::vector<MyEvent>` or `std::vector<std::vector<float>>`.

### Fundamental Types

The following fundamental types are stored as `leaf` fields with a single column each.
Type can potentially be stored in multiple possible column types.
The possible combinations are marked as `W` in the following table
Additionally, some types allow for reading from certain column types but not to write into them.
Such cases are marked as `R` in the table.

|               |                                                  Fundamental C++ Type                                                   ||
| Column Type   | bool | std::byte | char | int8_t | uint8_t | in16_t | uin16_t | int32_t | uint32_t | int64_t | uint64_t | float | double |
|---------------|:----:|:---------:|:----:|:------:|:-------:|:------:|:-------:|:-------:|:--------:|:-------:|:--------:|:-----:|:------:|
| Bit           |  W*  |           |      |        |         |        |         |         |          |         |          |       |        |
| Byte          |      |     W*    |      |        |         |        |         |         |          |         |          |       |        |
| Char          |      |           |  W*  |        |         |        |         |         |          |         |          |       |        |
| Int8          |      |           |      |   W*   |    R    |        |         |         |          |         |          |       |        |
| UInt8         |      |           |      |   R    |    W*   |        |         |         |          |         |          |       |        |
| (Split)Int16  |      |           |      |        |         |   W*   |    R    |         |          |         |          |       |        |
| (Split)UInt16 |      |           |      |        |         |   R    |    W*   |         |          |         |          |       |        |
| (Split)Int32  |      |           |      |        |         |        |         |    W*   |    R     |    R    |          |       |        |
| (Split)UInt32 |      |           |      |        |         |        |         |    R    |    W*    |    R    |          |       |        |
| (Split)Int64  |      |           |      |        |         |        |         |         |          |    W*   |    R     |       |        |
| (Split)UInt64 |      |           |      |        |         |        |         |         |          |    R    |    W*    |       |        |
| Real16        |      |           |      |        |         |        |         |         |          |         |          |   W   |   W    |
| (Split)Real32 |      |           |      |        |         |        |         |         |          |         |          |   W*  |   W    |
| (Split)Real64 |      |           |      |        |         |        |         |         |          |         |          |       |   W*   |

Possibly available `const` and `volatile` qualifiers of the C++ types are ignored for serialization.
The default column for serialization is denoted with an asterix.
If the ntuple is stored uncompressed, the default changes from split encoding to non-split encoding where applicable.

### Low-precision Floating Points

The ROOT type `Double32_t` is stored on disk as a `double` field with a `SplitReal32` column representation.
The field's type alias is set to `Double32_t`.

### STL Types and Collections

The following STL and collection types are supported.
Generally, collections have a mother column of type (Split)Index32 or (Split)Index64.
The mother column stores the offsets of the next collection entries relative to the cluster.
For instance, an `std::vector<float>` with the values `{1.0}`, `{}`, `{1.0, 2.0}`
for the first 3 entries results in an index column `[1, 1, 3]`
and a value column `[1.0, 1.0, 2.0]`.

#### std::string

A string is stored as a single field with two columns.
The first (principle) column is of type SplitIndex32.
The second column is of type Char.

#### std::vector\<T\> and ROOT::RVec\<T\>

STL vector and ROOT's RVec have identical on-disk representations.
They are stored as two fields:
  - Collection mother field whose principal column is of type `(Split)Index[64|32]`.
  - Child field of type `T`, which must by a type with RNTuple I/O support.
    The name of the child field is `_0`.

For RVecs, ROOT will always store the fully qualified type name `ROOT::VecOps::RVec<T>`.
Implementations should also be able to parse the shorter alias `ROOT::Vec<T>`.

#### std::array<T, N> and array type of the form T[N]

Fixed-sized arrays are stored as two fields:
  - A repetitive field of type `std::array<T, N>` with no attached columns. The array size `N` is stored in the field meta-data.
  - Child field of type `T` named `_0`, which must be a type with RNTuple I/O support.

Note that T can itself be an array type, which includes support for multidimensional C-style arrays.

#### std::variant<T1, T2, ..., Tn>

Variants are stored in $n+1$ fields:
  - Variant mother field with one column of type Switch; the dispatch tag points to the principal column of the active type
  - Child fields of types `T1`, ..., `Tn`; their names are `_0`, `_1`, ...

The dispatch tag ranges from 1 to $n$.
A value of 0 indicates that the variant is in the invalid state, i.e., it does not hold any of the valid alternatives.
Variants must not have more than 125 subfields.
This follows common compiler implementation limits.

#### std::pair<T1, T2>

A pair is stored using an empty mother field with two subfields, one of type `T1` and one of type `T2`. `T1` and `T2` must be types with RNTuple I/O support.
The child fileds are named `_0` and `_1`.

#### std::tuple<T1, T2, ..., Tn>

A tuple is stored using an empty mother field with $n$ subfields of type `T1`, `T2`, ..., `Tn`. All types must have RNTuple I/O support.
The child fields are named `_0`, `_1`, ...

#### std::bitset\<N\>

A bitset is stored as a repetitive leaf field with an attached `Bit` column.
The bitset size `N` is stored as repetition parameter in the field meta-data.
Within the repetition blocks, bits are stored in little-endian order, i.e. the least significant bits come first.

#### std::unique_ptr\<T\>, std::optional\<T\>

A unique pointer and an optional type have the same on disk representation.
They are represented as a collection of `T`s of zero or one elements.
A collection mother field has a single subfield named `_0` for `T`, where `T` must have RNTuple I/O support.
Note that RNTuple does not support polymorphism, so the type `T` is expected to be `T` and not a child class of `T`.

By default, the mother field has a principal column of type `(Split)Index[64|32]`.
This is called sparse representation.
The alternative, dense representation uses a `Bit` column to mask non-existing instances of the subfield.
In this second case, a default-constructed `T` (or, if applicable, a `T` constructed by the ROOT I/O constructor) is stored on disk for the non-existing instances.

#### std::set\<T\> and std::unordered_set\<T\>

While STL (unordered) sets by definition are associative containers (i.e., elements are referenced by their keys, which in the case for sets are equal to the values), on disk they are represented as indexed collections.
This means that they have the same on-disk representation as `std::vector<T>`, using two fields:
  - Collection mother field whose principal column is of type `(Split)Index[64|32]`.
  - Child field of type `T`, which must by a type with RNTuple I/O support.
    The name of the child field is `_0`.

#### std::map\<K, V\> and std::unordered_map\<K, V\>

An (unordered) map is stored using a collection mother field, whose principal column is of type `(Split)Index[64|32]` and a child field of type `std::pair<K, V>` named `_0`.

### std::atomic\<T\>

Atomic types are stored as a leaf field with a single subfield named `_0`.
The mother field has no attached columns.
The subfield corresponds to the the inner type `T`.

### User-defined enums

User-defined enums are stored as a leaf field with a single subfield named `_0`.
The mother field has no attached columns.
The subfield corresponds to the integer type the underlies the enum.
Unscoped and scoped enums are supported as long as the enum has a dictionary.

### User-defined classes

User-defined classes might behave either as a record or as a collection of elements of a given type.
The behavior depends on whether the class has an associated collection proxy.

#### Regular class / struct

User defined C++ classes are supported with the following limitations
  - The class must have a dictionary
  - All persistent members and base classes must be themselves types with RNTuple I/O support
  - Transient members must be marked, e.g. by a `//!` comment
  - The class must not be in the `std` namespace
  - The class must be empty or splittable (e.g., the class must not provide a custom streamer)
  - There is no support for polymorphism,
    i.e. a field of class `A` cannot store class `B` that derives from `A`
  - Virtual inheritance is unsupported

User classes are stored as a record mother field with no attached columns.
Direct base classes and persistent members are stored as subfields with their respective types.
The field name of member subfields is identical to the C++ field name.
The field name of base class subfields are numbered and preceeded by a colon (`:`), i.e. `:_0`, `:_1`, ...

#### Classes with an associated collection proxy

User classes that specify a collection proxy behave as collections of a given value type.
Associative collections are not currently supported.

The on-disk representation is similar to a `std::vector<T>` where `T` is the value type; specifically, it is stored as two fields:
  - Collection mother field whose principal column is of type `(Split)Index[64|32]`.
  - Child field of type `T`, which must by a type with RNTuple I/O support.
    The name of the child field is `_0`.

### ROOT::Experimental::RNTupleCardinality<SizeT>

A field whose type is `ROOT::Experimental::RNTupleCardinality<SizeT>` is associated to a single column of type (Split)Index32 or (Split)Index64.
This field presents the offsets in the index column as lengths that correspond to the cardinality of the pointed-to collection.

The value for the $i$-th element is computed by subtracting the $(i-1)$-th value from the $i$-th value in the index column.
If $i == 0$, i.e. it falls on the start of a cluster, the $(i-1)$-th value in the index column is assumed to be 0, e.g. given the index column values `[1, 1, 3]`, the values yielded by `RNTupleCardinality` shall be `[1, 0, 2]`.

The `SizeT` template parameter defines the in-memory integer type of the collection size.
The valid types are `std::uint32_t` and `std::uint64_t`.

### Unsplit types

A field with the structural role 0x05 ("unsplit") represents an object serialized by the ROOT streamer in unsplit mode.
It can have any type supported by TClass (even types that are not available in the native RNTuple type system).
The first (principal) column is of type [Split]Index[32|64].
The second column is of type Byte.

## Limits

This section summarizes key design limits of RNTuple data sets.
The limits refer to a single RNTuple and do not consider combinations/joins such as "friends" and "chains".

| Limit                                          | Value                        | Reason / Comment                                     |
|------------------------------------------------|------------------------------|------------------------------------------------------|
| Maximum volume                                 | 10 PB (theoretically more)   | Assuming 10k cluster groups of 10k clusters of 100MB |
| Maximum number of elements, entries            | 2^64                         | Using default (Split)Index64, otherwise 2^32         |
| Maximum cluster & entry size                   | 8TB (depends on pagination)  | Assuming limit of 4B pages of 4kB each               |
| Maximum page size                              | 2B elements, 256MB-2GB       | #elements * element size, 2GB limit from locator     |
| Maximum element size                           | 8kB                          | 16bit for number of bits per element                 |
| Maximum number of column types                 | 64k                          | 16bit for column type                                |
| Maximum envelope size                          | 2^48B (~280TB)               | Envelope header encoding                             |
| Maximum frame size                             | 2^62B, 4B items (list frame) | Frame preamble encoding                              |
| Maximum field / type version                   | 4B                           | Field meta-data encoding                             |
| Maximum number of fields, columns              | 4B (foreseen: <10M)          | 32bit column / field IDs, list frame limit           |
| Maximum number of cluster groups               | 4B (foreseen: <10k)          | List frame limits                                    |
| Maximum number of clusters per group           | 4B (foreseen: <10k)          | List frame limits, cluster group summary encoding    |
| Maximum number of pages per cluster per column | 4B                           | List frame limits                                    |
| Maximum number of entries per cluster          | 2^60                         | Cluster summary encoding                             |
| Maximum string length (meta-data)              | 4GB                          | String encoding                                      |

## Glossary

TODO: This glossary is not yet complete

### Column

A column is a storage backed vector of a number of **elements** of a simple type.
Column elements have a fixed bit-length depending on the column type.

### Page

A page is segment of a column.
Columns are partitioned in pages.
A page is a unit of compression.
Typical page sizes are of the order of 10-100kB.

### Cluster

A cluster is a set of pages from a fixed set of columns that contain all the data that belongs to a certain entry range.
The data set is partitioned in clusters.
Typically, a cluster comprises pages from all the available columns.
If only a subset of the available columns are covered, it is called a **sharded cluster**.
A typical cluster size is 50MB - 500MB.

### Indications of size

In this document, the `length` of something (e.g., a page) refers to its size in bytes in memory, uncompressed.
The `size` of something refers to the size in bytes on disk, possibly compressed.

## Notes on Backward and Forward Compatibility

TODO(jblomer)
- Ignore unknown column types
- Backwards compatiblity promised back to version 1
- Envelope compression algorithm(s) fixed (zstd or none?)
- Feature flags
- Skipping of unknown information (frames, envelopes)
- Writer version and minimum version
- Feature flag skipping
