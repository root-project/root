# RNTuple Reference Specifications (WIP)

**Note:** This is work in progress. The RNTuple specification is not yet finalized.

This document describes version 1 of the RNTuple binary format.

The RNTuple binary format describes the serialized, on-disk representation of an RNTuple data set.
The data on disk is organized in **pages** (typically 10-100kB in size)
and several **envelopes** that contain information about the data such as header and footer.
The RNTuple format specifies the binary layout of the pages and the envelopes.

Pages and envelopes are meant to be embedded in a data container
such as a ROOT file or a set of objects in an object store.
Envelopes can reference other envelopes and pages by means of a **locator** or an **envelope link**;
for a file embedding, the locator consists of an offset and a size.
The RNTuple format does _not_ establish a specific order of pages and envelopes.

Every emedding must define an **anchor** that contains the envelope links (location, compressed and uncompressed size)
of the header envelope and the footer envelope.
For the ROOT file embedding, the **ROOT::Experimental::RNTuple** object acts as an anchor.


## Compression Block

RNTuple envelopes and pages are wrapped into compression blocks.
In order to deserialize a page or an envelope, its compressed and ucompressed size needs to be known.

TODO(jblomer): reference or describe the compression block format.
  - Compressed size == uncompressed size --> uncompressed
  - Otherwise: connected compressed chunks with the 9 byte header


## Basic Types

Data stored in envelopes is encoded using the following type system.
Note that this type system is independent (and different) from the regular ROOT serialization.

Integer
: Integers are encoded in two's complement, little-endian format.
They can be signed or unsigned and have lengths up to 64bit.

String
: A string is stored as a 32bit unsigned integer indicating the length of the string
followed by the characters.
Strings are ASCII encoded; every character is a signed 8bit integer.

The meta-data envelope defines additional basic types (see below).


### Feature Flags

Feature flags are 64bit integers where every bit represents a certain feature that is used
in the binary format of the RNTuple at hand.
The most significant bit is used to indicate that there are more than 63 features to specify.
That means that readers need to continue reading feature flags as long as their integer value is negative.


### Frames

RNTuple envelopes can store records and lists of basic types and other records or lists by means of **frames**.
The frame has the following format

 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                             Size                            |T|
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|        Number of Items (for collections frames)       |Reserv.|
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         FRAME PAYLOAD                         |
|                              ...                              |

Size (uint 31bit)
: The size in bytes of the frame and the payload

T(ype)
: Can be either 0 for a **record frame** or 1 for a **collection frame**.
The type can be interpreted as the sign bit of the size, i.e. negative sizes indicate collection frames.

Reserved, Number of items (uint 28bit)
: Only used for collection frames to indicate the length of the list in the frame payload.
The reseved bits might be used in a future format versions.

File format readers should use the size provided in the frame to seek to the data that follows a frame
instead of summing up the sizes of the elements in the frame.
This approach ensures that frames can be extended in future file format versions
without breaking the deserialization of older readers.


### Locators and Envelope Links

A locator is a generalized way to specify a certain byte range on the storage medium.
For disk-based storage, the locator is just byte offset and byte size.
For object stores, the locator can specify a certain object ID.
The locator as the following format

 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                             Size                            |T|
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
+                             Offset                            +
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

Size:
If type is zero, the number of bytes to read, i.e. the compressed size of the referenced block.
Otherwise the size of the locator itself.

T(ype):
Zero for an on-disk or in-file locator, 1 otherwise.
Can be interpreted as the sign bit of the size, i.e. negative sizes indicate non-disk locators.
In this case, the locator should be interpreted like a frame, i.e. size indicates the _frame size_.

Offset:
For on-disk / in-file locators, the 64bit byte offset of the referenced byte range.

An envelope link consists of a locator followed by a 32bit unsigned integer
that specifies the uncompressed size of the envelope.


## Envelopes

Envelopes are continuous data blocks containing information that describe the RNTuple data.
The following envelope types exist

| Type                  | Contents                                                          |
|-----------------------|-------------------------------------------------------------------|
| Header                | RNTuple schema: field and column types                            |
| Footer                | Description of clusters, location of auxiliary meta-data          |
| Page list             | Location of data pages                                            |
| Auxiliary meta-data   | Key-value pairs of additional information about the data          |
| Check point (?)       | Minimal footer at X MB boundaries for recovery of crashed writes  |

Envelopes have the following format

 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|        Envelope Version       |        Minimum Version        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
                         ENVELOPE PAYLOAD
                               ...
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                             CRC32                             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

Envelope version
: Envelope types are versioned independently from each other.
The envelope version is the version that the writer used to build the envelope.

Minimum version
: A reader must support at least this version in order to extract meaningful data from the envelope.
If the envelope version is larger than the minimum version, there might be additional data in the envelope
that older readers can safely ignore.

CRC32
: Checksum of the envelope and the payload

Note that the size of envelopes is given by the RNTuple anchor (header, footer)
or by a locator that references the envelope.


### Header Envelope

The header consists of the following elements:

 - Feature flag
 - String: name of the ntuple
 - String: description of the ntuple
 - Collection frame: list of fields
 - Collection frame: list of columns
 - Collection frame: list of alias columns

#### Field Description

Every element of the list of fields has the following contents

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

If the flag 0x01 (_repetitive field_) is set, the field represents a fixed sized array.
In this case, an additional 64bit integer specifies the size of the array.

The block of integers is followed by a list of strings:

- String: field name
- String: type name
- String: type alias
- String: field description

The order of fields matters: every field gets an implicit field ID
which is equal the zero-based index of the field in the serialized list;
subfields are ordered from smaller IDs to larger IDs.

The flags field can have one of the following bits set

| Bit      | Meaning                                                                       |
|----------|-------------------------------------------------------------------------------|
| 0x01     | Repetitive field, i.e. for every entry $n$ copies of the field are stored     |
| 0x02     | Alias field, i.e. among the columns referring to this field are alias columns |

The structural role of the field can have on of the following values

| Value    | Structural role                                  |
|----------|--------------------------------------------------|
| 0x00     | Leaf field in the schema tree                    |
| 0x01     | The field is the mother of a collection (vector) |
| 0x02     | The field is the mother of a record (struct)     |
| 0x03     | The field is the mother of a variant (union)     |
| 0x04     | The field is a reference (pointer), TODO         |


#### Column Description

 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|              Type             |        Bits on Storage        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
+                            Field ID                           +
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                             Flags                             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

The order of columns matter: every column gets an implicit column ID
which is equal to the zero-based index of the column in the serialized list.
Multiple columns attached to the same field should be attached from smaller to larger IDs.

The column type and bits on storage integers can have one of the following values

| Type | Bits | Name         | Contents                                                                      |
|------|------|--------------|-------------------------------------------------------------------------------|
| 0x01 |   64 | Index64      | Mother columns of (nested) collections, counting is relative to the cluster   |
| 0x02 |   64 | Switch       | Lower 44 bits like kIndex64, higher 20 bits are a dispatch tag to a column ID |
| 0x03 |    8 | Byte         | An uninterpreted byte, e.g. part of a blob                                    |
| 0x04 |    8 | Char         | ASCII character                                                               |
| 0x05 |    1 | Bit          | Boolean value                                                                 |
| 0x06 |   64 | Real64       | IEEE-754 double precision float                                               |
| 0x07 |   32 | Real32       | IEEE-754 single precision float                                               |
| 0x08 |   16 | Real16       | IEEE-754 half precision float                                                 |
| 0x09 |   64 | Int64        | Two's complement, little-endian 8 byte integer                                |
| 0x0A |   32 | Int32        | Two's complement, little-endian 4 byte integer                                |
| 0x0B |   16 | Int16        | Two's complement, little-endian 2 byte integer                                |
| 0x0C |    8 | Int8         | Two's complement, 1 byte integer                                              |
| 0x0D |   64 | SplitIndex64 | Like Index64 but pages are stored in split + delta encoding                   |
| 0x0E |   64 | SplitReal64  | Like Real64 but in split encoding                                             |
| 0x0F |   32 | SplitReal32  | Like Real32 but in split encoding                                             |
| 0x10 |   16 | SplitReal16  | Like Real16 but in split encoding                                             |
| 0x11 |   64 | SplitInt64   | Like Int64 but in split encoding                                              |
| 0x12 |   32 | SplitInt32   | Like Int32 but in split encoding                                              |
| 0x13 |   16 | SplitInt16   | Like Int16 but in split encoding                                              |

Future versions of the file format may introduce addtional column types
without changing the minimum version of the header.
Old readers need to ignore these columns and fields constructed from such columns.
Old readers can, however, figure out the number of elements stored in such unknown columns.

The flags field can have one of the following bits set

| Bit      | Meaning                                                      |
|----------|--------------------------------------------------------------|
| 0x01     | Elements in the column are sorted (monotonically increasing) |
| 0x02     | Elements in the column are sorted (monotonically decreasing) |
| 0x04     | Elements have only positive values                           |


#### Alias columns

An alias column is just a 32bit integer that references the physical column ID.
The ID of the alias column itself is given implicitly by the serialization order.
In particular, alias columns have larger IDs than physical columns.
In the footer and page list envelopes, only physical column IDs can be referenced.

### Footer Envelope

- Feature flags
- Header checksum (CRC32)
- Collection frame of extension header envelope links
- Collection frame of meta-data block envelope links
- Collection frame of cluster summaries



### Auxiliary Meta-data envelope

TODO(jblomer)

### Checkpoint Envelope

TODO(jblomer)

## Limits

TODO(jblomer)

maximum size of frame, envelope
max number of fields, columns, clusters: 200M (due to frame limits)
  Due to switch column: 1M fields, columns
max cluster size: 16TB (switch column)
max file size / data set size
Maximum element size: 8k (better 4?)

## Notes on Backward and Forward Compatibility

TODO(jblomer)
- Ignore unknown column types
- Backwards compatiblity promised back to version 1
- Envelope compression algorithm(s) fixed
- Feature flags
- Skipping of unknown information (frames, envelopes)
- Writer version and minimum version
- Feature flag skipping

# Questions:
  - Better big endian?
  - Field ID and column ID: 32 bit?
  - Take ROOT::Experimental::RNTuple out of experimental?
  - Which locator types should we specify now? 64bit object ID, URL?
  - Reference fields
  - Column types complete? Should we have 128bit floats, ints? 8bit floats?
