# RNTuple Reference Specifications (WIP)

**Note:** This is work in progress. The RNTuple specification is not yet finalized.

This document describes version 1 of the RNTuple binary format.

The RNTuple binary format describes the serialized, on-disk representation of an RNTuple data set.
The data on disk is organized in **pages** (typically 10-100kB in size) and several **envelope** blocks that contain meta-data,
such as header and footer.
The RNTuple format specifies the binary layout of the pages and the envelopes.

Pages and envelopes are meant to be embedded in a data container
such as a ROOT file or a set of objects in an object store.
Envelopes can reference other envelopes and pages by means of a **locator**;
for a file embedding, the locator consists of an offset and a size.
The RNTuple format does _not_ establish a specific order of pages and envelopes.

Every emedding must define an **anchor** that specifies the location and the uncompressed and compressed size
of the header envelope and the footer envelope.
For the ROOT file embedding, the **ROOT::Experimental::RNTuple** object acts as an anchor.


## Compression Block

RNTuple envelopes and pages are wrapped into compression blocks.
In order to read a page or an envelope, its compressed and ucompressed size needs to be known.

TODO(jblomer): reference or describe the compression block format.


## Basic Types

Data stored in envelopes is encoded using the following type system.
Note that this type system is independent and different from the regular ROOT serialization.

Integer
: Integers are encoded in two's complement, little-endian format.
They can be signed or unsigned and have lengths up to 64bit.

String
: A string is stored as a 32bit unsigned integer indicating the length of the string
followed by the characters.
Strings are ASCII encoded; every character is an 8bit signed integer.

The meta-data envelope defines additional basic types (see below).


### Frames

RNTuple envelopes can store records and lists of basic types and other records or lists by means of **frames**.
The frame has the following format

 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                             Size                            |T|
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|            Number of Items (for collections frames)           |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         FRAME PAYLOAD                         |
|                              ...                              |

Size
: The size in bytes of the frame and the payload

T(ype)
: Can be either 0 for a **record frame** or 1 for a **collection frame**.
The type can be interpreted as the sign bit of the size, i.e. negative sizes indicate collection frames.

Number of items
: Only used for collection frames to indicate the length of the list in the frame payload.
Omitted

File format readers should use the size provided in the frame to seek to the data that follows a frame
instead of summing up the sizes of the elements in the frame.
This approach ensures that frames can be extended in future file format versions
without breaking the deserialization of older readers.


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

Note that the size of envelopes is given by the RNTuple anchor (header, footer)
or by a locator that references the envelope.


### Header Envelope



### Footer Envelope

### Auxiliary Meta-data envelope

TODO(jblomer)

### Checkpoint Envelope

TODO(jblomer)

## Limits

TODO(jblomer)

maximum size of frame, envelope
max number of fields, columns, clusters
max file size / data set size

## Notes on Backward and Forward Compatibility

TODO(jblomer)
- Ignore unknown column types
- Backwards compatiblity promised back to version 1
- Envelope compression algorithm(s) fixed
- Feature flags
- Skipping of unknown information (frames, envelopes)
- Writer version and minimum version
- Feature flag skipping
