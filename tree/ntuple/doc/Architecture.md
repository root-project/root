# RNTuple Code Architecture

> This document is meant for ROOT developers. It provides background information on the RNTuple code design and behavior.

> The RNTuple code uses the nomenclature from the [RNTuple format specification](https://github.com/root-project/root/blob/master/tree/ntuple/doc/BinaryFormatSpecification.md) (e.g. "field", "column", "anchor", etc.).

## General Principles

The RNTuple classes provide the functionality to read, write, and describe RNTuple datasets.
The core classes, such as `RNTupleReader` and `RNTupleWriter`, are part of the RNTuple library.
Additional tools, such as the `RNTupleImporter` and the `RNTupleInspector`, are part of the RNTupleUtils library,
which depends on the RNTuple library.

The RNTuple classes are organized in layers:
the storage layer, the primitives layer, the logical layer and the event iteration layer.
Most classes in the storage layer and the primitives layer are in the `ROOT::Internal` namespace (non-public interfaces),
with the notable exception of the descriptor classes (`RNTupleDescriptor`, `RFieldDescriptor`, etc.).
Most classes in the upper layers provide public interfaces.

| Layer      | Description                                                         | Example of classes                                          |
|------------|---------------------------------------------------------------------|-------------------------------------------------------------|
| Storage    | Read and write pages (physical: file, object store; virtual: e.g. buffered) | RPage{Source,Sink}, RNTupleDescriptor, RClusterPool         |
| Primitives | Storage-backed columns of simple types                              | RColumn, RColumnElement, RPage                              |
| Logical    | Mapping of C++ types onto columns                                   | RField, RNTupleModel, REntry                                |
| Iteration  | Reading and writing events / properties                             | RNTuple{Reader,Writer}, RNTupleView, RNTupleDS (RDataFrame) |
| Tooling    | Higher-level, RNTuple related utility classes                       | RNTupleMerger, RNTupleImporter, RNTupleInspector            |

The RNTuple classes are, unless explicitly stated otherwise, conditionally thread safe.

The read and write APIs provide templated, compile-time type-safe APIs,
APIs where the type at hand is passed as string and which are runtime type-safe,
and type-unsafe APIs using void pointers.

On I/O errors and invalid input, RNTuple classes throw an `RException`.

## Walkthrough: Reading Data

```c++
auto file = std::make_unique<TFile>("data.root");
auto ntuple = std::unique_ptr<RNTuple>(file->Get<RNTuple>("ntpl"));

// Option 1: entire row
// The reader creates a page source; the page source creates a model from the on-disk information
auto reader = RNTupleReader::Open(*ntuple);
// Populate the objects that are used in the model's default entry
reader->LoadEntry(0);
std::shared_ptr<float> pt = reader->GetModel().GetDefaultEntry().GetPtr<float>("pt");

// Option 2: imposed model
auto model = RNTupleModel::Create();
auto pt = model->MakeField<float>("pt");
// The reader checks the passed model for compatibility; only the subset of fields defined in the model is read
auto reader = RNTupleReader::Open(std::move(model), *ntuple);
reader->LoadEntry(0);

// Option 3: through views
// Each view will only trigger reading of the related field, without reading other fields at the same entry number.
auto reader = RNTupleReader::Open(*ntuple);
auto viewPt = reader->GetView<float>("pt");
// Load the pt from the first entry
auto pt = viewPt(0);
```

In the above cases, RNTuple creates the objects being read into.
It is also possible to bind already existing objects.
This is shown below for entries and works similarly for views.

```c++
// A bare entry is an entry that has initially no bindings (all top-level fields need to be bound by the caller)
auto entry = reader->GetModel().CreateBareEntry();
auto ptToken = entry->GetToken("pt");

// Option 1: type safe, shared ownership
std::shared_ptr<float> ptTypedSharedPtr;
entry->BindValue(ptToken, ptTypedSharedPtr);

// Option 2: type unsafe, shared ownership
std::shared_ptr<void> ptVoidSharedPtr;
entry->BindValue(ptToken, ptVoidSharedPtr);

// Option 3: type unsafe, application owns the object
void *ptVoidPtr;
entry->BindRawPtr(ptToken, ptVoidPtr);

// Option 4: switch back from application-provided object to RNTuple-created object
entry->EmplaceNewValue(ptToken);

// For all options: use an explicit entry
reader->LoadEntry(0, *entry);
```


## Walkthrough: Writing Data

```c++
auto model = RNTupleModel::Create();
// Add a field to the model and return the shared pointer for that field in the model's default entry.
auto ptrPt = model->MakeField<float>("pt");

auto file = std::make_unique<TFile>("data.root", "APPEND");
// The writer creates a page sink and connects the model's fields to it
auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);
*ptrPt = 1.0;
// Append the model's default entry
writer->Fill();
// Commit the dataset by destructing the writer
writer.reset();
```

The points on object type-safety and ownership apply in the same way as for reading data.

Creation of the RNTuple model can use runtime type information:

```c++
auto model = RNTupleModel::Create();
model->AddField(RFieldBase::Create("pt", "float").Unwrap());
```


## Main Classes

### RNTuple
The RNTuple class contains the information of the RNTuple anchor in a ROOT file (see specification).
It has a dictionary and is streamed through the standard ROOT I/O.
An RNTuple object represents an RNTuple dataset but it is not the dataset itself.
It can be used like a token to open the actual RNTuple dataset with, e.g., RDF or an RNTupleReader,
and it provides the `Merge(...)` interface for the `TFileMerger`.

### RPageSource / Sink
The page source and sink can read and write pages and clusters from and to a storage backend.
There are concrete class implementations for an RNTuple stored in a ROOT file (local or remote), and for an RNTuple stored in a DAOS object store.
There is a virtual page sink for buffered writes, which also groups pages of the same column before flushing them to disk.

Page sources and sinks do not operate entry-based but based on pages/indices of columns.
For instance, there is no API in the page sink to write an entry, but only to write pages of columns.
The higher-level APIs, e.g. `RField`, `REntry`, `RNTupleWriter`, take care of presenting the available data as entries where necessary.

The page source also gives access to an `RNTupleDescriptor` through a read/write lock guard.
The `RNTupleDescriptor` owned by the page source changes only when new cluster metadata are loaded.
The header and the cluster group summary information is stable throughout its lifetime (cf. format specification).

### R{NTuple,Field,Column,Cluster,...}Descriptor
The descriptor classes provide read-only access to the on-disk metadata of an RNTuple.
The metadata includes the schema (fields and columns), information about clusters and the page locations.
The descriptor classes are closely related to the format specification.

For normal read and write tasks, access to the descriptor is not necessary.
One notable exception is bulk reading, where the descriptor can be used to determine entry boundaries of clusters.
The descriptors are used internally, e.g. to build an RNTupleModel from the on-disk information.
The descriptors are also useful for inspection purposes.

The descriptor classes contain a copy of the metadata; they are not linked to an open page source.
A descriptor can be used after its originating page source has been deleted.

### RField<T>
The RField<T> classes are central in RNTuple:
they link the in-memory representation of data types to their on-disk representation.
All field classes inherit from `RFieldBase`.

Every type with RNTuple I/O supported has a corresponding RField<T> template specialization.
Complex types are composed of fields and sub fields.
E.g., a `struct` is represented by a parent field for the `struct` itself and a subfield for every member of the `struct`.
Fields of complex types have type-erased versions in addition to the templated ones (e.g., `RVectorField`, `RClassField`).
In this way, fields can be constructed even if the type information is only available at runtime.
To this end, `RFieldBase::Create()` creates an `RField` object from a type string.

On the "in-memory" side, fields can construct and destroy objects of their wrapped type
(cf. `CreateValue()`, `CreateObject()`, `GetDeleter()` methods).
Existing objects in memory can be bound to fields (cf. `BindValue()` method).

On the "on-disk" side, fields know about the possible column representations of their wrapped type.
Upon connecting a field to a page source or page sink,
fields create `RColumn` objects and register them with the page source/sink.
When reading and writing data, the field maps the in-memory information of an object to read/write calls on its columns.
For instance, when writing a `std::vector<float>`,
the field writes to an index column (storing information about the size of the vector).
Its subfield writes the actual values to a float column.

During its lifetime, a field undergoes the following possible state transitions:
```
[*] --> Unconnected --> ConnectedToSink ----
             |      |                      |
             |      --> ConnectedToSource ---> [*]
             |                             |
             -------------------------------
```

The RField class hierarchy is fixed and not meant to be extended by user classes.

### RFieldBase::RValue
The `RValue` class makes the connection between an object in memory and the corresponding field used for I/O.
It contains a shared pointer of the object, i.e. RNTuple and the application share ownership of objects.
The object in an RValue can either be created by an RNTuple field (cf. `RField<T>::CreateValue()` method)
or provided by the application (cf. `RField<T>::BindValue()` method).
Raw pointers can be passed with the understanding that the raw pointer is owned by the application and are kept alive during I/O operations.

`RValue` objects can only be created from fields, and they are linked to their originating field.

### RNTupleModel
The RNTupleModel represents a data schema as a tree of fields.
The model owns its fields.
A model undergoes the following possible state transitions:
```
[*] ---> Building / --> Frozen --
         Updating         |     |
             ^            |     |
             |-------------     v
             ----------------->[*]
```
During the building/updating phase, new top-level fields are added to the model.
Frozen models can create entries.
Every model has a unique model ID, which is used to identify the entries created from this model.
Unless a model is created as "bare model", it owns a default entry that is used by default by the `RNTupleReader` and the `RNTupleWriter`.

A model can add _projected fields_.
Projected fields map existing physical fields to a different type.
For instance, a `std::vector<Event>` can be projected onto a `std::vector<float>` for a float member of `Event`.
Projected fields are stored as header metadata.

Fields can be added to a model after the writing process has started (cf. `RNTupleWriter::CreateModelUpdater()`).
This is called _late model extension_.
Addition of fields invalidates previously created entries.
The values of deferred fields for the already written entries is set to the default constructed type of the field.

### REntry
The REntry represents a row/entry in an RNTuple.
It contains a list of `RValue` objects that correspond to the top-level fields of the originating model.
The entry gives access to the shared pointers corresponding to the top-level fields.
It also provides functionality to bind application-provided pointers.

An REntry can be passed to `RNTupleWriter::Fill()` and `RNTupleReader::LoadEntry()`.
Otherwise, the reader/writer uses the default entry of its model.

An entry can safely outlive its originating model.
New objects cannot anymore be created (`EmplaceNewValue` will throw an exception), but the entry is still properly destructed.

### RNTupleWriter, RNTupleParallelWriter
The RNTupleWriter is the primary interface to create an RNTuple.
The writer takes ownership of a given model.
The writer can either add an RNTuple to an existing ROOT file (`RNTupleWriter::Append()`) or create a new ROOT file with an RNTuple (`RNTupleWriter::Recreate()`).
Once created, entries are added to an RNTuple either serially (`RNTupleWriter::Fill()`) or in concurrently in multiple threads with the `RNTupleParallelWriter`.
Once committed (e.g. by releasing the RNTupleWriter), the RNTuple is immutable and cannot be amended.
An RNTuple that is currently being written cannot be read.

### RNTupleReader
The RNTupleReader is the primary interface to read and inspect an RNTuple.
An RNTupleReader owns a model: either a model created from the on-disk information or an imposed, user-provided model.
The user-provided model can be limited to a subset of fields.
Data is populated to an explicit `REntry` or the model's default entry through `RNTupleReader::LoadEntry()`.

The reader can create `RNTupleView` objects for the independent reading of individual fields.
The reader can create `RBulkValues` objects for bulk reading of individual fields.

Additionally, the reader provides access to a cached copy of the descriptor.
It can display individual entries (`RNTupleReader::Show()`) and summary information (`RNTupleReader::PrintInfo()`).

### RNTupleView<T>
RNTuple views provide read access to individual fields.
Views are created from an RNTupleReader.
Views are templated; for simple types (e.g., `float`, `int`), views provide read-only access directly to an RNTuple page in memory.
Complex types and void views require additional memory copies to populate an object in memory from the column data.

A view can iterate over the entry range, over the field range, and over the range of a collection within an entry.
For instance, for a field `std::vector<float> pt`, a view can iterate over all `pt` values of all entries, or over the `pt` values of a particular entry.

A view can safely outlive its originating reader.
Once the reader is deconstructed, any attempt to read data will throw an exception, but the view is still properly destructed.

Views that originate from the same reader _cannot_ be used concurrently by different threads.

## Internal Classes

### RNTupleDS
The `RNTupleDS` class is an internal class that provides an RNTuple data source for RDataFrame.
It is part of the `ROOTDataFrame` library.
The RNTuple data source supports chains with a constructor that takes a list of input files.
The RNTuple data source also supports multi-threaded dataframes, parallelized on the file and cluster level.

The data source exposes inner fields of complex collections.
For instance, if the data model contains a vector of `Event` classes, where each `Event` has `pt` and `eta` floats,
the dataframe can use the event vector itself (`Event` column) as well as the `float` columns `Event.pt` and `Event.eta`.

### RClusterPool
The RClusterPool is an internal class owned be a page source.
The cluster pool maintains an I/O thread that asynchronously prefetches the next few clusters.
Through `RPageSource::SetEntryRange()`, the cluster pool is instructed to not read beyond the given limit.
This is used in the RNTuple data source when multiple threads work on different clusters of the same file.

### RMiniFile
The RMiniFile is an internal class used to read and write RNTuple data in a ROOT file.
It provides a minimal subset of the `TFile` functionality.
Its purpose is to reduce the coupling between RNTuple and the ROOT I/O library.

For writing data, the RMiniFile can either use a proper `TFile` (descendant) or a C file stream (only for new ROOT files with a single RNTuple).
For reading, the `RMiniFile` always uses an `RRawFile`.

### RRawFile
The RRawFile internal abstract class provides an interface to read byte ranges from a file, including vector reads.
Concrete implementations exist for local files, XRootD and HTTP (the latter two through the ROOT plugin mechanism).
The local file implementation on Linux uses uring for vector reads, if available.
`RRawFileTFile` wraps an existing `TFile` and provides access to the full set of implementations, e.g. `TMemFile`.

## Tooling

### RNTupleMerger
The `RNTupleMerger` is an internal class and part of the core RNTuple library.
It concatenates RNTuple data from several sources into a combined sink.
It implements "fast merging", i.e. copy-based merging that does not decompress and recompress pages.
The RNTupler merger is used by the `TFileMerger` and thus provides RNTuple merge support in `hadd` and `TBufferMerger`.

### RNTupleImporter
The RNTupleImporter creates RNTuple data sets from ROOT trees.
It is part of the `ROOTNTupleUtil` library.

### RNTupleInspector
The RNTupleInspector provides insights of an RNTuple, e.g. the distribution of data volume wrt. column types.
It is part of the `ROOTNTupleUtil` library.

## Ownership Model

By default, objects involved in RNTuple I/O (objects read from disk or written to disk) are passed to RNTuple as shared pointers.
Both RNTuple or the application may create the object.
Raw pointers to objects can be passed to RNTuple -- such objects are considered as owned by the application.
The caller has to ensure that the lifetime of the object lasts during the I/O operations.

An RNTuple writer that is constructed without a `TFile` object (`RNTupleWriter::Recreate()`) assumes exclusive access to the underlying file.
An RNTuple writer that uses a `TFile` for writing (`RNTupleWriter::Append()`) assumes that the `TFile` object outlives the writer's lifetime.
The serial writer assumes exclusive access to the underlying file during construction, destruction and `Fill()` as well as `CommitCluster()` and `FlushCluster()`.
For `FlushColumns()` and `FillNoFlush()`, the sequential writer assumes exclusive access only if buffered writing is turned off.
The parallel writer assumes exclusive access to the underlying file during all operations on the writer (e.g. construction and destruction) and all operations on any created fill context (e.g. `Fill()` and `FlushCluster()`).
Notable exceptions are `FlushColumns()` and `FillNoFlush()` which are guaranteed to never access the underlying `TFile` during parallel writing (which is always buffered).

A `TFile` does not take ownership of any `RNTuple` objects.

When reading data, RNTuple uses the `RMiniFile` and `RRawFile` classes to open a given storage path and find the `RNTuple` anchor.
When creating a `RNTupleReader` from an existing anchor object, RNTuple uses `RRawFile` only for files of dynamic type `TFile`, `TDavixFile`, and `TNetXNGFile`.
In either case, the `RRawFile` owns its own file descriptor and does not interfere with `TFile` objects concurrently reading the file.
For anchors from files of other dynamic type, including all other `TFile` subclasses, the file is wrapped in a `RRawFileTFile` and access is shared.

## On-Disk Encoding

### Writing Case
The following steps are taken to write RNTuple data to disk:

  1. On creation of the RNTupleWriter, the header is written to disk
  2. Upon `RNTupleWriter::Fill()`, the RField<T> class _serializes_ the object into its column representation.
     To this end, it uses the `RColumn` class to append elements to the columns page buffer (`RPage`)
  3. When a page buffer is full (cf. tuning.md), it is sent to the page sink for writing it to disk.
     Note that page boundaries do _not_ need to align with entry boundaries,
     e.g. information from a single entry can span multiple pages.
      1. The page is _packed_:
         depending on the type of the page, a light encoding is applied to facilitate compression, e.g., byte splitting (`RColumnElement`).
         Big-endian / little-endian conversion takes place here.
      2. The packed page is _compressed_ according to the user-provided compression settings (default: zstd).
         A packed and compressed page is _sealed_.
      3. The sealed page is written to the storage backend.
  4. When the target cluster size is reached (cf. tuning.md), the `Fill()` method automatically commits the cluster.
     The user can also manually commit a cluster.
  5. When the dataset is committed (e.g., on destruction of the RNTupleWriter), the page list, the footer, and the anchor are written to disk.

The header, footer, and page list are compressed.

If the buffered sink is used (default), the pages of a cluster are buffered until the cluster is committed.
On committing the cluster, all pages are sealed and sent to a _persistent sink_ in one go (vector write).
Pages are also reordered to ensure locality of pages of the same column.

#### Late model extension
For fields added to the RNTupleModel after the RNTuple schema has been created (i.e., through `RNTupleWriter::CreateModelUpdater()`), the following steps are taken:

  1. On calling `RUpdater::BeginUpdate()`, all `REntry` instances belonging to the underlying RNTupleModel are invalidated.
  2. After adding the desired additional fields, calling `RUpdater::CommitUpdate()` will add the relevant fields to the footer's [schema extension record frame](./BinaryFormatSpecification.md#schema-extensions-record-frame).
      1. The principal columns of top-level fields and record subfields will have a non-zero first element index.
         These columns are referred to as "deferred columns".
         In particular, columns in a subfield tree of collections or variants are _not_ stored as deferred columns (see next point).
      2. All other columns belonging to the added (sub)fields will be written as usual.
  3. `RNTuple(Writer|Model)::CreateEntry()` or `RNTupleModel::CreateBareEntry()` must be used to create an `REntry` matching the new model.
  4. Writing continues as described in steps 2-5 above.

### Reading Case
The reverse process is performed on reading (e.g. `RNTupleReader::LoadEntry()`, `RNTupleView` call operator).

By default, the page source uses an `RClusterPool` to asynchronously read-ahead data.
When a page of a certain cluster is required, the cluster pool reads pages of _active_ columns.
For instance, if only certain fields are used (e.g., through an imposed model), only the pages of columns connected to those fields are read.
Columns can be dynamically added (e.g. during event iteration, a new field view is created in a reader).
The cluster pool reads ahead a limited number of clusters given by the _cluster bunch size_ option (default = 1).
The read-ahead uses vector reads.
For the file backend, it additionally coalesces close read requests and uses uring reads when available.

The page source can be restricted to a certain entry range.
This allows for optimizing the page lists that are being read.
Additionally, it allows for optimizing the cluster pool to not read-ahead beyond the limits.

#### Late model extension
Reading an RNTuple with an extended model is transparent -- i.e., no additional interface calls are required.
Internally, columns that were created as part of late model extension will have synthesized zero-initialized column ranges for the clusters that were already written before the model was extended.
In addition, pages made up of 0x00 bytes are synthesized for deferred columns in the clusters that were already (partially) filled before the model was extended.

## Storage Backends

Support for storage backends is implemented through derived classes of `RPageSink` and `RPageSource`.
The `RPage{Sink,Source}File` class provides a storage backend for RNTuple data in ROOT files, local or remote.
The `RPage{Sink,Source}Daos` class provides a storage backend for RNTuple data in the DAOS object store.

Every new storage backend needs to define
  1) The RNTuple embedding: how are RNTuple data blobs stored, e.g. in keys of ROOT files, or in objects of object stores
  2) The RNTuple anchor: the initial link to the location of the header and footer (cf. format specification)
  3) A locator format: how are byte ranges addressed (e.g., through an offset in a file or an object ID)

That means that new backends are likely to have implications on the RNTuple format specification.

The page sources and sinks are ROOT internal classes.
They are not meant to be extended by users.

## Multi-Threading

The following options exist in RNTuple for multithreaded data processing.

### Implicit Multi-Threading
When `ROOT::EnableImplicitMT()` is used, RNTuple uses ROOT's task arena to compress and decompress pages.
That requires writes to be buffered and reads uses the cluster pool resp.
The RNTuple data source for RDataFrame lets RDataFrame full control of the thread pool.
That means that RDataFrame uses a separate data source for every thread, each of the data sources runs in sequential mode.

### Concurrent Readers
Multiple readers can read the same RNTuple concurrently as long as access to every individual reader is sequential.

### Parallel REntry Preparation
Multiple `REntry` object can be concurrently prepared by multiple threads.
I.e., construction and binding of the objects can happen in parallel.
The actual reading and writing of entries (`RNTupleReader::LoadEntry()`, `RNTupleWriter::Fill()`) needs to be protected by a mutex.
This is considered "mild scalability parallelization" in RNTuple.

### RNTupleParallelWriter
The parallel writer offers the most scalable parallel writing interface.
Multiple _fill contexts_ can concurrently serialize and compress data.
Every fill context prepares a set of entire clusters in the final on-disk layout.
When a fill context flushes data,
a brief serialization point handles the RNTuple metadata updates and the reservation of disk space to write into.

## Low precision float types
RNTuple supports encoding floating point types with a lower precision when writing them to disk. This encoding is specified by the
user per field and it is independent on the in-memory type used for that field (meaning both a `RField<double>` or `RField<float>` can
be mapped to e.g. a low-precision 16 bit float).

RNTuple supports the following encodings (all mutually exclusive):

- **Real16**/**SplitReal16**: IEEE-754 half precision float. Set by calling `RField::SetHalfPrecision()`;
- **Real32Trunc**: floating point with less than 32 bits of precision (truncated mantissa).
  Set by calling `RField::SetTruncated(n)`, with $10 <= n <= 31$ equal to the total number of bits used on disk.
  Note that `SetTruncated(16)` makes this effectively a `bfloat16` on disk;
- **Real32Quant**: floating point with a normalized/quantized integer representation on disk using a user-specified number of bits.
  Set by calling `RField::SetQuantized(min, max, nBits)`, where $1 <= nBits <= 32$.
  This representation will map the floating point value `min` to 0, `max` to the highest representable integer with `nBits` and any
  value in between will be a linear interpolation of the two. It is up to the user to ensure that only values between `min` and `max`
  are stored in this field. The current RNTuple implementation will throw an exception if that is not the case when writing the values to disk.

In addition to these encodings, a user may call `RField<double>::SetDouble32()` to set the column representation of a `double` field to
a 32-bit floating point value. The default behavior of `Float16_t` can be emulated by calling `RField::SetTruncated(21)` (which will truncate
a single precision float's mantissa to 12 bits).

Here is an example on how a user may dynamically decide how to quantize a floating point field to get the most precision out of a fixed bit width:
```c++
auto model = RNTupleModel::Create();
auto field = std::make_unique<RField<float>>("f");
// assuming we have an array of floats stored in `myFloats`:
auto [minV, maxV] = std::minmax_element(myFloats.begin(), myFloats.end());
constexpr auto nBits = 24;
field->SetQuantized(*minV, *maxV, nBits);
model->AddField(std::move(field));
auto f = model->GetDefaultEntry().GetPtr<float>("f");

// Now we can write our floats.
auto writer = RNTupleWriter::Recreate(std::move(model), "myNtuple", "myFile.root");
for (float val : myFloats) {
  *f = val;
  writer->Fill();
}
```

## Relationship to other ROOT components

The RNTuple classes have the following relationship to other parts of ROOT.

The RNTuple classes use core ROOT infrastructure classes, such as error handling and logging.
When necessary, RNTuple uses a `TFile` for reading and writing.
The cases of writing to a local file and reading from a local file, a file from XRootD or from HTTP, do _not_ require `TFile`.
For these cases, RNTuple depends on the `RRawFile` class and its XRootD and Davix plugins.

For user-defined classes as well as sets and maps, RNTuple uses `TClass`.
Simple types and other stdlib classes are natively supported and do not require dictionaries.
See the format specification for an exhaustive list of types supported in RNTuple.
The streamer field uses the standard ROOT streaming machinery.

Integration to RDataFrame is provided through an RNTuple data source.
A universal RDataFrame constructor can create a data frame from either a TTree or an RNTuple with the same syntax.

The RBrowser uses RNTuple classes to display RNTuple dataset information.

## Future Features

The following features are planned for after the first RNTuple production version:
  - RNTupleProcessor: advanced RNTupleReader that allows for free combination of chains and (indexed/unaligned) friends
  - Horizontal merging: persistified friends, analogous to a classical merge being a persistified chain
  - An interface for bulk writing
  - Attributes: RNTuple-specific and user-provided metadata storage, such as file provenance, scale factors, or varied columns
  - C library interface
  - S3 storage backend (page source / page sink)


# Semantics of Reading Non-Trivial Objects

Reading an object with RNTuple should be seen as _overwriting_ its persistent data members.
Given a properly constructed and valid object, the object must ensure that it stays valid when overwriting its persistent data members.
However, the object should not rely on its transient state to remain unchanged during reading:
it may be destructed and constructed again when it is read as part of a collection (see below).

An object that is being read from disk may have been constructed by `RField::CreateValue()`.
In this case, the deleter returned by `RField::GetDeleter()` releases the resources.

When reading collections of type `T` (`std::vector<T>`, `ROOT::RVec<T>`, ...), RNTuple uses `RField::CreateValue()` to construct elements of the inner type `T`.
As the size of a collection changes from event to event, this has the following effect on its elements
  - If the collection shrinks, cut-off elements are destructed
  - If the collection grows, new elements are constructed before reading them
  - If the array buffer of the collection is reallocated (may happen for both shrinking and growing depending on the collection), all elements are destructed first in the old buffer
  and the new number of elements is constructed in the new buffer

So unless the collection buffer needs to be reallocated, RNTuple tries to avoid unnecessary destruction/construction but instead overwrites existing objects.
Note that RNTuple currently does not copy or move existing objects when the collection buffer is reallocated.


# Naming Conventions

For byte arrays and collections of things, the RNTuple code uses the following variable name suffixes:
  - `XyzSize` denotes the size of Xyz in bytes on disk, i.e. after compression. Example: `fPageListSize`.
  - `XyzLength` denotes the size of Xyz in bytes in memory, i.e. uncompressed. Example: `fPageListLength`.
  - `NXyz` denotes the number of Xyz items in a collection. Example: `fNPageLists`.
