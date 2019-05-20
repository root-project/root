RForest Introduction
====================

RForest is the experimental evolution of TTree columnar data storage. RForest introduces new interfaces that aim to be
more robust.  In particular, the new interfaces are type-safe through the use of templates, and the ownership is
well-defined through the use of smart pointers.  For instance

    tree->Branch("px", &Category, "px/F");

becomes

    auto px = model->MakeField<float>("px");
    // px is std::shared_ptr<float>

The physical layout changes slightly from big endian to little endian so that it matches the in-memory layout on
most modern architectures. Combined with a clear separation of offset/index data and payload data for collections,
uncompressed RForest data can be directly mapped to memory without further copies.


Goals
-----

RForest shall investigate improvements of the TTree I/O in the following ways

1. More speed
   * Improve mapping to vectorized and parallel hardware
   * For types known at compile / JIT time: generate optimized code
   * Optimized for simple types (float, int, and vectors of them)
   * Better memory control: work with a fixed budget of pre-defined I/O buffers
   * Naturally thread-safe and asynchronous interfaces

2. More robust interfaces
   * Compile-time type safety by default
   * Decomposition into layers: logical layer, primitives layer, storage layer
   * Separation of data model and live data
   * Self-contained I/O code to support creation of a standalone I/O library


Concepts
--------

At the **logical layer**, the user defines a data model using the RNTupleModel class.
The data model is a collection of serializable C++ types with associated names, similar to branches in a TTree.
The data model can contain (nested) collection, e.g., a type can be `std::vector<std::vector<float>>`.

Each serializable type is represented by a **field**, concretely by a templated version of RField,
e.g. `RField<double>`. A field can generate or adopt an associated **value**, which represents a memory location
storing a value of the given C++ type.  These distinguished memory locations are the destinations and sources for the
deserialization and serialization.

The (de-)serialization is a mapping from the C++ type to the more simple **column** type system.  A column contains
an arbitrary number of fixed-sized elements of a well-defined set of types: integers and floats of different
bit sizes.  A C++ type may be mapped to multiple columns.  For instance, an `std::vector<float>` maps to two columns,
an offset column indicating the size of the vector per entry, and a payload column with the float data.

Columns are partitioned into **pages** (roughly: TTree baskets) of a few kB -- a few tens of kB each.
The **physical layer** (only) needs to provide the means to store and retrieve pages.  The physical layer is
decoupled from the high-level C++ logic.  The physical layer implements an abstract page storage interface,
so that dedicated implementations for key-value stores and other storage systems are conceivable.
At this point, the only provided backend stores the pages in ROOT files.

Forests are further grouped into **clusters**, which are, like TTree clusters, self-contained blocks of
consecutive entries.  Clusters provide a unit of writing and will provide the means for parallel writing of data
in a future version of RForest.
