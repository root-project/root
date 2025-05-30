% ROOT Version 6.36 Release Notes
% 2025-05
<a name="TopOfPage"></a>

## Introduction

ROOT 6.36.00, released on May 26th 2025, is the first release of the ROOT 6.36 LTS cycle.
The new ROOT comes with many great features that have the potential of making you even more productive - see release notes below.
The effort made by the contributors, experiments and core developers is testified by the 202 items in the ROOT trackers that have been addressed for this release.

As previously announced, the 6.36 cycle supersedes the STS 6.34 one, which will be stopped in June.

For more information, see:

[http://root.cern](http://root.cern)

The following people have contributed to this new version:

 Samantha Abbott, UC Davis,\
 Bertrand Bellenot, CERN/EP-SFT,\
 Jakob Blomer, CERN/EP-SFT,\
 Lukas Breitwieser, CERN/EP-SFT,\
 Will Buttinger, RAL,\
 Philippe Canal, FNAL,\
 Emanuele Chiamulera, University of Padova\
 Eric Cano, CERN/EP-SFT
 Olivier Couet, CERN/EP-SFT, \
 Marta Czurylo, CERN/EP-SFT, \
 Tomas Dado, CERN/EP-ADP,\
 Monica Dessole, CERN/EP-SFT,\
 Mattias Ellert, Uppsala University,\
 Mateusz Fila, CERN EP-SFT,\
 Martin Føll, CERN/EP-SFT and University of Oslo,\
 Declan Garvey, CSIC/University of Valencia,\
 Florine de Geus, CERN/EP-SFT and University of Twente,\
 Massimiliano Galli, Princeton,\
 Jamie Gooding, TU Dortmund, \
 Jonas Hahnfeld, CERN/EP-SFT and Goethe University Frankfurt,\
 Fernando Hueso Gonzalez, CSIC/University of Valencia,\
 Stephan Hageboeck, CERN/EP-SFT,\
 Aaron Jomy, CERN/EP-SFT,\
 Marta Krawczyk, CERN/EP-AID,\
 Vit Kucera, Inha University,\
 Stephanie Kwan, Princeton,\
 Sergey Linev, GSI Darmstadt,\
 Enrico Lusiani, INFN Padova,\
 Alberto Mecca, University of Turin,\
 Lorenzo Moneta, CERN/EP-SFT,\
 Mark Owen, University of Glasgow,\
 Vincenzo Eduardo Padulano, CERN/EP-SFT,\
 Giacomo Parolini, CERN/EP-SFT,\
 Daniel Perez Astudillo, QEERI/HBKU,\
 Danilo Piparo, CERN/EP-SFT,\
 Alexandr Prozorov, CTU Prague,\
 Jonas Rembser, CERN/EP-SFT,\
 Grigori Rybkin, Paris-Saclay University,\
 Manuel Tobias Schiller, University of Glasgow,\
 Surya Somayyajula, UMass Amherst,\
 Petr Stepanov, @petrstepanov,\
 Maciej Pawel Szymanski, ANL,\
 Andres Rios, Princeton,\
 Sanjiban Sengupta, CERN/EP-SFT and Manchester University,\
 Silia Taider, CERN/EP-SFT,\
 Florian Uhlig, GSI,\
 Devajith Valaparambil Sreeramaswamy, CERN/EP-SFT,\
 Vassil Vasilev, Princeton,\
 Dongliang Zhang, University of Science and Technology of China

## Deprecation and Removal

* The RooFit legacy interfaces that were deprecated in ROOT 6.34 and scheduled for removal in ROOT 6.36 are removed. See the RooFit section in the 6.34 release notes for a full list.
* The `TPython::Eval()` function that was deprecated in ROOT 6.34 and scheduled for removal in ROOT 6.36 is removed.
* The `RooDataSet` constructors to construct a dataset from a part of an existing dataset are deprecated and will be removed in ROOT 6.38. This is to avoid interface duplication. Please use `RooAbsData::reduce()` instead, or if you need to change the weight column, use the universal constructor with the `Import()`, `Cut()`, and `WeightVar()` arguments.
* The ROOT splash screen was removed for Linux and macOS
* Proof support has been completely removed from RooFit and RooStats, after it was already not working anymore for several releases
* The build options `mysql`, `odbc`, `pgsql` have been deprecated. Please complain with root-dev@cern.ch should you still need one!
* The build options `html` and `qt5web` have been removed.
* The ClassImp macro and derived macros have no effect now and will be deprecated in ROOT 6.38.
* The default TTreeFormula constructor (without arguments) is now deleted, since it lead to an unusable and unsafe object. Instead, this implementation has been reserved now for ROOT I/O exclusively via the TRootIOCtor argument tag.
* The `RooStats::HLFactory` class was deprecated will be removed in ROOT 6.38. It provided little advantage over using the RooWorkspace directly or any of the other higher-level frameworks that exist in the RooFit ecosystem.

## Python Interface

### UHI
* ROOT histograms now comply with the [Unified Histogram Interface (UHI)](https://uhi.readthedocs.io/en/latest/index.html) specification, enhancing interoperability with other UHI-compatible libraries and standardizing histogram operations.
  The following features were added:
  * Implemented the UHI `PlottableHistogram` protocol enabling ROOT histograms to be plotted by any library supporting `PlottableHistogram` objects.
  * Introduced UHI-style indexing for access and setting bin values.
  * Introduced UHI-style slicing for selecting histogram ranges.
  * Implemented the `ROOT.uhi.loc`, `ROOT.uhi.underflow`, `ROOT.uhi.overflow`, `ROOT.uhi.rebin`, and `ROOT.uhi.sum` tags.

## Core Libraries
Existing code can now be interpreted and just-in-time compiled according to the C++23 standard for the platforms that support it. The new standard can be enabled with the CMake option `-DCMAKE_CXX_STANDARD=23`.


## RDataFrame
- When running multiple computation graphs concurrently using [`RunGraphs()`](https://root.cern/doc/master/namespaceROOT_1_1RDF.html#a526d77d018bf69462d736bbdd1a695c4),
  the pool of slot numbers that a thread can pick from is now shared across all graphs. This enables use cases where a single resource, which may be expensive to create or copy,
  is shared across tasks belonging to different computation graphs.
- RDataFrame's RResultPtr now allow direct access to the underlying shared_ptr using [GetSharedPtr()](https://root.cern.ch/doc/v636/classROOT_1_1RDF_1_1RResultPtr.html#a633d680942845d91c10a48f62918f908). See also the new tutorial [df040](https://root.cern.ch/doc/v636/df040__RResultPtr__lifetimeManagement_8C.html) for details of lifetime management of RDataFrame results.
- Support for single-threaded snapshotting to RNTuple has been added. This can be enabled through `RSnapshotOptions`. Note that snapshotting from a TTree-based RDataFrame to RNTuple is not yet supported in this release, but will be added in a future version. The current recommended way to convert from TTree to RNTuple is through the [RNTupleImporter](https://root.cern/doc/v636/classROOT_1_1Experimental_1_1RNTupleImporter.html).
- The support for reading CSV file inputs has been extended with the following capabilities:
  - Left/right trimming
  - Skipping of a given number of header/footer lines
  - Comment character to skip lines / line remainders
  - Override column names
- The `Display` operation has been extended to support different printing options. Notably, it now includes the possibility
  to display the table in Markdown or HTML formats. When using RDataFrame in an IPython-based environment, e.g. a Jupyter
  notebook, the output format will be HTML by default. This will be parsed and displayed nicely in the notebook cell output.
- The RNTuple data source has been moved out of the experimental namespace, now in namespace `ROOT::RDF`. In the transition,
  the support to read a raw RNTuple pointer was removed in order to ensure a simpler and more sustainable API.
- The `Report` action is now supported in the case of systematic variations, i.e. it is now possible to call
  `VariationsFor(report)`. This produces the usual map of varied `Report` nodes.
- New alerts have been put in place when a user requests entry ranges out of the bounds of the data source, addressing the following situations (nEntries is the total number of entries in the dataset):
  - `begin<nEntries` or `end>nEntries`: raise a warning at the end of the execution
  - `begin==nEntries`: throw an exception as soon as this is detected
- Fixed a faulty interaction between sample paths in an RDataFrame execution and the corresponding redirected URL in
  case the path is pointing to a FUSE-mounted EOS location. The map of available samples now contains both the
  FUSE-mounted location and the redirected xroot URL. For more info, see https://github.com/root-project/root/pull/17544.
- The RNTuple data source now follows the automatic conversion to `ROOT::RVec` in-memory when the corresponding on-disk
  type of the column is one of `ROOT::RVec`, `std::vector`, `std::array`. This follows the same logic used for TTree. In
  all other cases, the RNTuple data source will use the same in-memory type as the on-disk type.
- The Python usability of RDataFrame when reading a TTree-based dataset with a TChain has been improved. Specifically,
  the ownership of the TChain is now shared between the RDataFrame object(s) and the Python interpreter. The following
  Python function would have previously returned an RDataFrame with a dangling reference to the underlying dataset,
  whereas now the TChain will stay alive as long as the RDataFrame will:
  ```python
  def foo():
    ch = ROOT.TChain(...)
    return ROOT.RDataFrame(ch)
  ```
- A new option for `Snapshot` has been added to the `RSnapshotOptions`, namely the possibility to specify the output
  TTree basket size. Users can set the `fBasketSize` data member to the desired basket size.
- The RNTuple data source now supports reading the same column with different valid column types, for example reading
  a `std::vector<T>` column as `ROOT::RVec<T>` and vice-versa. An exception is thrown in case a requested type is
  incompatible with the actual column type.
- The treatment of TTree datasets in RDataFrame has been completely overhauled and is now aligned with all other data
  sources, i.e. a new `RDataSource`-derived class was introduced to manage TTree processing. These changes are completely
  transparent to the user, but improve sustainability of the code base. At the same time, the need to JIT the
  column readers for TTree dataset was removed thus slightly improving runtime performance.
- The behaviour of `Snapshot` with the TTree output in case no entries pass the selections has been changed. Now, both
  in single- and multi-thread mode, the operation will produce an output file, with a TTree inside. The TTree will contain
  the complete dataset schema (i.e. the column names) requested during the `Snapshot` call and it will have no entries.
- It is now possible to read a string column type with a `pandas.DataFrame` data source.
- It is now possible to create an RDataFrame wrapping an empty `pandas.DataFrame`, as long as all the columns of the
  `pandas.DataFrame` are of supported types.
- The behaviour of `AsNumpy` has been modified, depending on whether the input column type is a scalar or collection.
  In the case of scalar, the output numpy array will have its `dtype` of the corresponding Python type when the input
  type is a fundamental type (e.g. integer or float). Columns containing non-fundamental types will result in a numpy
  array with `dtype=object`. In the case of a collection, the output numpy array will have `dtype=object`, where each
  element is itself a numpy array representing the collection for its corresponding entry in the column. If the
  collection at a certain entry contains values of fundamental types, or if it is a regularly shaped multi-dimensional
  array of a fundamental type, then the numpy array representing the collection for that entry will have the `dtype`
  associated with the value type of the collection. If the collection at a certain entry contains values of a
  non-fundamental type, it will be represented as a numpy array with `dtype=object`.

### Distributed RDataFrame

The local and distributed APIs of RDataFrame have been made completely uniform. Using the keyword argument `executor`,
an RDataFrame can run on a distributed set of nodes. Furthermore, distributed RDataFrame (`ROOT.RDF.Distributed`) is now
out of the experimental namespace. We recommend that users switch to using the new version as soon as possible. Note the following changes between the old and the new code:

```python
# Old version - not recommended:
# This will still work in 6.36 but will raise a warning
connection = ... # your distributed Dask client or SparkContext
RDataFrame = ROOT.RDF.Experimental.Distributed.[Backend].RDataFrame
df = RDataFrame(..., [daskclient,sparkcontext] = connection)
# New version - recommended
connection = ... # your distributed Dask client or SparkContext
df = ROOT.RDataFrame(..., executor = connection)
```

In case of using the old version, with `Experimental` keyword, a warning will be raised. In ROOT 6.38, the `Experimental` name will be removed completely and using it will result in an exception.

A set of new features is now available to the distributed RDataFrame users:

* It is now much easier to inject the C++ code into the distributed RDF application. The following functions are available:
  * DistributeHeaders - include and distribute headers
  * DistributeSharedLibs - load and distribute shared libraries
  * DistributeCppCode - declare and distribute the C++ code, the C++ code is available to all dataframes in the application
  * DistributeFiles - distribute other files (not headers or shared libraries) that may be needed

* The factory function `FromSpec` is available to build the distributed RDF. Note, however, the addition and use of the metadata is not yet supported.

* Support for the following API operations was added to distributed RDataFrame:
  * `Alias`
  * `DefaultValueFor`
  * `FilterAvailable`
  * `FilterMissing`

## RooFit

### Breaking function signature changes

  * The `RooRealVar::format()` function was changed to return a `std::string` instead of a `TString *`.
    The former return type was not memory safe, since the caller had to delete the `TString`.
    This pattern was not appropriate for a modern C++ library.
    If you absolutely need the old return type, wrap the call to `format()` inside `new TString{var.format(..)}`. However, this is not recommended.

### Fix for `RooAbsReal::createHistogram()` with extended pdfs

There was a problem with [RooAbsReal::createHistogram()](https://root.cern.ch/doc/master/classRooAbsReal.html#a9451168bb4159899fe1854f591f69814) when using it to get histograms with predicted yields for extended pdfs.
The `Scale(bool)` argument was always set internally to `false` in case `createHistogram()` was called on an extended pdf. There was no way for the user to override that.
This meant that one could not get yield histograms that were correctly scaled by the bin volumes using that function.
This release changes that behavior, meaning the `Scale(bool)` command argument is now respected for extended pdfs.

### RooCrystalBall improvements

- Fitting instabilities in the RooCrystalBall class (as well as in TMath's cbShape function) with high N parameter have been mitigated by reorganizing the code implementation.

## I/O

* Honour the `Davix.GSI.CACheck` parameter also in the `ROOT::Internal::RRawFileDavix` class.
* Added support for `enum class` with a non default underlying size, for example `enum smallenum: std::int16_t`.  The default is 32 bits.  All enums, independently of their in memory size are stored on file using 32 bits to enable forward compatibility of the file; files created with a `enum class` with a non default underlying size can be read with old version of ROOT into a `enum` type of default size.
* * Note: `enum class` with an underlying size strictly greater than 32 bits are not supported since they would be truncated when stored on file.
* The version number of `TStreamerInfo` has been increase to 10 to encoded the addition of the support for `enum class` with a non default underlying size. This allows the opportunity to detect files written by old version of ROOT (`v9` and older of `TStreamerInfo`) where  `enum class` with a non default underlying size where stored incorrectly but recoverably.  Those files can be recover by using I/O customization rules that takes in consideration their size at the time of writing (this information is not recorded in the `ROOT` file).  See https://github.com/root-project/root/pull/17009#issuecomment-2522228598 for some examples.
* New attribute for I/O customization rules: `CanIgnore`.  When using this attribute the rule will be ignored if the input is missing from the schema/class-layout they apply to instead of issue a `Warning`
* The logic to retrieve branch names of a TTree has been improved, leading to O(10) speedup. For example, an RDataFrame
  wrapping a TTree with O(10K) columns could be spending multiple minutes just to retrieve the full dataset schema,
  whereas now it will take seconds.

### RNTuple

* Following the [HEP-CCE review](https://indico.fnal.gov/event/67890/contributions/307688/attachments/185815/255889/RNTuple_HEP-CCE.pdf) of the RNTuple public API,
  the following types were moved from the `ROOT::Experimental` to the `ROOT` namespace:
  * `DescriptorId_t`
  * `Detail::RFieldVisitor`
  * `ENTupleColumnType` (renamed from `EColumnType`)
  * `ENTupleStructure`
  * `NTupleSize_t`
  * `RArrayAsRVecField`
  * `RArrayField`
  * `RAtomicField`
  * `RBitsetField`
  * `RCardinalityField`
  * `RClassField`
  * `RClusterDescriptor`
  * `RClusterGroupDescriptor`
  * `RColumnDescriptor`
  * `RCreateFieldOptions`
  * `REntry`
  * `REnumField`
  * `RExtraTypeInfoDescriptor`
  * `RField`
  * `RFieldBase`
  * `RFieldDescriptor`
  * `RFieldToken` (moved from `REntry::RFieldToken`)
  * `RFieldZero`
  * `RIntegralField`
  * `RInvalidField`
  * `RMapField`
  * `RNTupleCardinality`
  * `RNTupleCollectionView`
  * `RNTupleDescriptor`
  * `RNTupleDirectAccessView`
  * `RNTupleFillStatus`
  * `RNTupleGlobalRange`
  * `RNTupleLocalRange` (renamed from `RNTupleClusterRange`)
  * `RNTupleLocator`
  * `RNTupleLocatorObject64`
  * `RNTupleModel`
  * `RNTupleReader`
  * `RNTupleReadOptions`
  * `RNTupleView`
  * `RNTupleViewBase`
  * `RNTupleWriteOptions`
  * `RNTupleWriter`
  * `RNullableField`
  * `RPairField`
  * `RProxiedCollectionField`
  * `RRecordField`
  * `RRVecField`
  * `RSetField`
  * `RSimpleField`
  * `RStreamerField`
  * `RTupleField`
  * `RVariantField`
  * `RVectorField`
* New options have been added to TFileMerger (which can be passed as whitespace-separated TStrings via `TFileMerger::SetMergeOptions`)
  * "FirstSrcCompression": when merging multiple files, instructs the class-specific merger to use the same compression as the
    first object of the destination's class as the destination's compression. Currently only recognized by the RNTuple merger;
  * "DefaultCompression": specifies that the merged output should use the class-specific default compression. Currently only
    meaningful for RNTuple, which has a default compression different from the TFile's default compression (ZSTD instead of ZLIB).
    This option is automatically set by `hadd` when no other compression option is specified;
  * "rntuple.MergingMode=(Filter|Union|Strict)": RNTuple-specific option that specifies the merging mode that should be used by
    the RNTupleMerger (see
    [RNTupleMergeOptions](https://root.cern/doc/v634/structROOT_1_1Experimental_1_1Internal_1_1RNTupleMergeOptions.html));
  * "rntuple.ErrBehavior=(Abort|Skip)": RNTuple-specific option that specifies the behavior of the RNTupleMerger on error (see link above);
  * "rntuple.ExtraVerbose": RNTuple-specific option that tells the RNTupleMerger to emit more information during the merge process.
* The option to explicitly pass a type name or `std::type_info` to `RNTupleReader::GetView<void>` has been added, in case the on-disk type of the field differs from the underlying type of the pointer to read into.
* The [RNTupleProcessor](https://root.cern/doc/v636/classROOT_1_1Experimental_1_1RNTupleProcessor.html) has been extended to support *joins* (comparable to TTree friends) and *chains of joins*. As an effect, `RNTupleReader::OpenFriends()` has been removed. Note that the RNTupleProcessor is still under active development and will remain in the Experimental namespace.

## Tutorials and Code Examples

* The ROOT Tutorials have been modernized - we organized the tutorials into new categories and added more informative descriptions to each section. Some of the tutorials were fixed, renamed or moved to the legacy folder but also new tutorials were added to showcase features not represented before. We believe that interacting with the tutorials is now much easier and more pleasant both for the new and old users. Have a look [here](https://root.cern.ch/doc/v636/group__Tutorials.html) for all the details.

* add `tutorials/visualisation/webgui/bootstrap` example showing usage of `RWebWindow` with [bootstrap](https://getbootstrap.com/) framework, including embedding of `TWebCanvas` in the widget

## Math

### RooFit / HistFactory

An important bottleneck in HistFactory was removed when many histograms are used to construct a measurement.

### Performance improvements in Minuit2 for the case of parameters with limits

For variable parameters with limits, Minuit2 applies trigonometric transformations into an internal space for the minimization.
This can be a significant bottleneck for highly-optimized function, to the point that this is also mentioned in section 1.3.1 of the [old MINUIT user's guide](https://inspirehep.net/files/c92c2ba4dac7c0a665cce687fb19b29c).
One prominent case of highly-optimized functions are RooFit likelihoods, which are heavily caching intermediate results to make numeric gradients and Hessians as efficient as possible.
This meant that e.g. for HistFactory likelihoods, a significant fraction of the gradients and Hessians evaluation time was spent in the parameter transformations.
To alleviate this bottleneck, Minuit2 is now also caching the result of the trigonometric transformations, only recomputing them if a parameters value was changed. As a result, the time spent in in parameter transformations when minimizing RooFit likelihoods and evaluating the Hessian becomes negligible.

## Graphics

* SavePrimitive methods of all graphics-related classes were revised and modernized. Objects with large arrays are stored more
efficiently. Many small bugs were found and fixed.
* extend `stressGraphics` test on more use-cases, including `TSpline`, `TScatter`, `TEfficiency`, `TProfile2D`, `TProfile3D`, `TGraph2DErrors`, `TGraph2DAsymmErrors` classes
* enable `stressGraphics` tests for web-based graphics

## JavaScript ROOT

ROOT 6.36 will use JSROOT 7.9.x release series. It includes following important changes:

* Implement 'nmatch' parameter for `TTree::Draw` to limit processed events
* Implement 'elist' parameter for `TTree::Draw` to specify entries list for processing
* Implement 'staged' algorithm for `TTree::Draw` to first select entries and then process only these entries
* Implement 'cont5' draw option for `TGraph2D` using Delaunay algorithm
* Implement 'pol' and 'arr_colz' draw option for `TH2`
* Only 'col7' draw option uses bar offset and width for color `TH2` drawing
* Interactive zooming and context menu on 'chord' `TH2` drawing
* Implement 'box1' for `TH3` with negative bins
* Introduce `settings.FilesTimeout` to configure global timeout for file reading operations
* Introduce `settings.FilesRemap` to let provide fallback address for http server, used for `root.cern`
* Introduce `settings.TreeReadBunchSize` to configure bunch read size for `TTree` processing
* Adjust histogram title drawing with native implementation
* Improve float to string conversion when 'g' is specified
* Support 'same' option for first histogram, draw directly on pad
* Display underflow/overflow bins when configured for the axis, implement 'allbins' draw option for histograms
* Support different angle coordinates in `TGraphPolargram`, handle 'N' and 'O' draw options
* Support fAxisAngle in `TGraphPolargram`, provide 'rangleNN' draw option
* Implement 'arc' draw option for `TPave`
* Provide context menus for all derived from `TPave` classes
* Let edit histograms and graphs title via context menu
* Support Poisson errors for `TH1`/`TH2`, https://root-forum.cern.ch/t/62335/
* Test fSumw2 when detect empty `TH2` bin, sync with https://github.com/root-project/root/pull/17948
* Support `TLink` and `TButton` object, used in `TInspectCanvas`
* Support `TF12` - projection of `TF2`
* Upgrade three.js r168 -> r174
* Remove support of qt5 webengine, only qt6web can be used
* Set 'user-select: none' style in drawings to exclude text selection, using `settings.UserSelect` value
* Internals - use private members and methods
* Internals - use `WeakRef` class for cross-referencing of painters
* Internals - use negative indexes in arrays and Strings
* Fix - handle `TPave` NDC position also when fInit is not set
* Fix - properly handle image sizes in svg2pdf
* Fix - drawing `TPaveText` with zero text size
* Fix - correct axis range in `TScatter` drawing
* Fix - use draw option also for graph drawing in `TTree::Draw`


## Class Reference Guide

The tutorials have been reviewed, improved and modernised. It is now more intuitive to navigate through the individual parts of the ROOT tutorials. Some tutorials were added to showcase the new features, some were modernized to modern C++ standards and some were moved to the legacy folder.

## Build, Configuration and Testing Infrastructure

### New build options for dependiencies of image processing

ROOT supports image processing via `libAfterImage`, which can be enabled with the `asimage` build option.
A few more options were added to make ROOT builds more reproducible and to avoid builtin dependencies:

  * `asimage_tiff` (default `ON`): build `libAfterImage` with TIFF support if `libtiff` is found on the system
  * `builtin_gif` (default `OFF`): don't look for `libgif` on the system and instead build it together with ROOT
  * `builtin_jpeg` (default `OFF`): same as above but for `libjpeg`
  * `builtin_png` (default `OFF`): same as above but for `libpng`

With default build option values, there is no difference in behavior compared to previous ROOT versions.
The real benefit of the new options becomes apparent in builds with `fail-on-missing=ON`, because then the build will fail if any of the dependencies is not found.

## Bugs and Issues fixed in this release

More than 200 items were addressed for this release. The full list is:

  * [[#18827](https://github.com/root-project/root/issues/18827)] - ROOT uses the wrong value for Planck's constant
  * [[#18792](https://github.com/root-project/root/issues/18792)] - Thread problem in TEnum::GetEnum
  * [[#18738](https://github.com/root-project/root/issues/18738)] - Disable the default automatic optimization of RooFit via vectorization
  * [[#18669](https://github.com/root-project/root/issues/18669)] - Incremental builds with `dev=ON` break when modifying Core header
  * [[#18667](https://github.com/root-project/root/issues/18667)] - Add setting to disable file writing in MakeModelAndMeasurementsFast
  * [[#18663](https://github.com/root-project/root/issues/18663)] - Rootssh /tmp/root.XXXXXXXXX chmod issue and consequent THttpEngine (arg?) problem
  * [[#18654](https://github.com/root-project/root/issues/18654)] - TClassEdit name normalisation fails for unordered_map, leading to a crash in df104_CSVDataSource
  * [[#18643](https://github.com/root-project/root/issues/18643)] - [C++23][macOS] Failing test `roottest-root-io-tclass-make` with C++23
  * [[#18631](https://github.com/root-project/root/issues/18631)] - Code display for rf408_RDataFrameToRooFit.C partially broken in the documentation
  * [[#18623](https://github.com/root-project/root/issues/18623)] - DaviX errors from ROOT.File.Cp
  * [[#18581](https://github.com/root-project/root/issues/18581)] - Exception when trying to build RDataFrame from pandas
  * [[#18556](https://github.com/root-project/root/issues/18556)] - Class methods not available to python in cases of transient data members of private nested class type
  * [[#18536](https://github.com/root-project/root/issues/18536)] - TClass::GetClass hoards memory when called for numerical types
  * [[#18534](https://github.com/root-project/root/issues/18534)] - Cannot use FromPandas with string columns
  * [[#18524](https://github.com/root-project/root/issues/18524)] - Memory not released after TTree GetEntry() in PyROOT
  * [[#18520](https://github.com/root-project/root/issues/18520)] - Missing lock deep inside TClassEdit::ResolveTypedef
  * [[#18519](https://github.com/root-project/root/issues/18519)] - Concurrency issue with TClassEdit::ResolveTypedef and TClass::GetListOfMethods
  * [[#18499](https://github.com/root-project/root/issues/18499)] - CI artifacts are currently useless on Linux ...
  * [[#18477](https://github.com/root-project/root/issues/18477)] - TPad::ClipPolygon illegal access vector.at(-1)
  * [[#18455](https://github.com/root-project/root/issues/18455)] - Tiny memory leak reported by valgrind in TStreamerInfoActions
  * [[#18441](https://github.com/root-project/root/issues/18441)] - import ROOT fails after a template instantiation is invoked from cppyy
  * [[#18404](https://github.com/root-project/root/issues/18404)] - [ntuple] Unable to read collection of `TObject`
  * [[#18380](https://github.com/root-project/root/issues/18380)] - [tmva][sofie] `ROperator_Shape::ShapeInference()` code can't be correct
  * [[#18374](https://github.com/root-project/root/issues/18374)] - TH2Poly bin names are drawn with an angle of 1° with option "TEXTN"
  * [[#18365](https://github.com/root-project/root/issues/18365)] - SetBranchAddress does not work with np.int16 and uint16 in PyROOT
  * [[#18363](https://github.com/root-project/root/issues/18363)] - Incorrect name normalization for template with size_t template argument
  * [[#18354](https://github.com/root-project/root/issues/18354)] - ROOT does not build some pre-requisites on macOS with latest Brew because of make v4.0...
  * [[#18343](https://github.com/root-project/root/issues/18343)] - "Definitely lost" reported by Valgrind in TCling::InspectMembers
  * [[#18329](https://github.com/root-project/root/issues/18329)] - [core] "Definitely lost" from Valgrind in TUnixSystem::UnixOpendir
  * [[#18322](https://github.com/root-project/root/issues/18322)] - [RF] "Definitely lost" reported by Valgrind in RooAbsCollection
  * [[#18312](https://github.com/root-project/root/issues/18312)] - Problem with cmake 4.0.0
  * [[#18306](https://github.com/root-project/root/issues/18306)] - macos 15.4 building bug of module map file
  * [[#18304](https://github.com/root-project/root/issues/18304)] - TBrowser not opening in root cling CLI on Fedora if RBrowser lib was deleted after install
  * [[#18301](https://github.com/root-project/root/issues/18301)] - Performance issue seen by CMS for ROOT's use of tbb::task_arena
  * [[#18298](https://github.com/root-project/root/issues/18298)] - [ntuple] RNTupleMerger: compress the generated zero pages
  * [[#18285](https://github.com/root-project/root/issues/18285)] - [FreeBSD] root build fails in master
  * [[#18251](https://github.com/root-project/root/issues/18251)] - RMiniFile.cxx harsh failure
  * [[#18236](https://github.com/root-project/root/issues/18236)] - Missing lock deep inside TClassEdit::GetNormalizedName
  * [[#18213](https://github.com/root-project/root/issues/18213)] - macos sequoia 15.4 homebrew - Failed to Open ROOT after today's update
  * [[#18195](https://github.com/root-project/root/issues/18195)] - TBufferJSON not serializing std::map
  * [[#18172](https://github.com/root-project/root/issues/18172)] - [RF] Inefficient un-registering of histograms in HistFactory
  * [[#18167](https://github.com/root-project/root/issues/18167)] - [ci] Apply Ruff Formatting to Changed Lines Only
  * [[#18146](https://github.com/root-project/root/issues/18146)] - [Python] Preserve full Python trace when raising errors in callbacks
  * [[#18128](https://github.com/root-project/root/issues/18128)] - installing with rapidyaml cmake found
  * [[#18066](https://github.com/root-project/root/issues/18066)] - Using a ternary conditional expression in TTree::Draw may mess up the histogram specification.
  * [[#18055](https://github.com/root-project/root/issues/18055)] - misterious dependency on `mkl` for conda+ROOT 6.34
  * [[#18013](https://github.com/root-project/root/issues/18013)] - [RF] RooFormula doesn't complain if you use input called `x` but don't supply it
  * [[#18002](https://github.com/root-project/root/issues/18002)] - Memory issues reported by Valgrind when cloning `gROOT->GetListOfColors()`
  * [[#17992](https://github.com/root-project/root/issues/17992)] - Spurrious auto parsing when looking up a TClass
  * [[#17969](https://github.com/root-project/root/issues/17969)] - Failure of thisroot.sh on freebsd
  * [[#17900](https://github.com/root-project/root/issues/17900)] - [ntuple] Add write API with const pointers
  * [[#17864](https://github.com/root-project/root/issues/17864)] - [Docs] TTree example is missing the critical lines
  * [[#17859](https://github.com/root-project/root/issues/17859)] - [CMake] JupyROOT sources not correctly tracked by CMake
  * [[#17848](https://github.com/root-project/root/issues/17848)] - [ntuple] Type normalization problems with types in template classes
  * [[#17843](https://github.com/root-project/root/issues/17843)] - [TTreeReader] Crash in TNotifyLink with reused TChain with friend
  * [[#17841](https://github.com/root-project/root/issues/17841)] - PyROOT multidimenstional int16_t arrays stopped working in 6.34
  * [[#17824](https://github.com/root-project/root/issues/17824)] - TDirectory::mkdir does not return the created directory
  * [[#17820](https://github.com/root-project/root/issues/17820)] - Wrong interaction between indexed TTree friend, GetEntries, GetListOfFriends
  * [[#17809](https://github.com/root-project/root/issues/17809)] - TGraph2D doc (TGraph2D::GetFunction does not work)
  * [[#17774](https://github.com/root-project/root/issues/17774)] - [ntuple] ATLAS RNTuple Writing Issue (2025-02-19)
  * [[#17757](https://github.com/root-project/root/issues/17757)] - TClass object can be generated during tear down which can lead to the user of already deleted resources.
  * [[#17754](https://github.com/root-project/root/issues/17754)] - [docu] TMinuit2 weird links
  * [[#17753](https://github.com/root-project/root/issues/17753)] - Race condition in TClass::fStreamerImpl value for emulated classes.
  * [[#17729](https://github.com/root-project/root/issues/17729)] - [Python Interface] Regression: can't properly convert a vector to 2D/3D numpy array
  * [[#17715](https://github.com/root-project/root/issues/17715)] - Disable Docs workflow for forked repositories
  * [[#17691](https://github.com/root-project/root/issues/17691)] - Crash when building dataframe from TChain
  * [[#17686](https://github.com/root-project/root/issues/17686)] - C++23 crash in DiagnosticInfo.h
  * [[#17658](https://github.com/root-project/root/issues/17658)] - Remove infinities in RooFit Crystal Ball PDF
  * [[#17648](https://github.com/root-project/root/issues/17648)] - [ntuple] Record field emulation not working with TFile
  * [[#17634](https://github.com/root-project/root/issues/17634)] - TH1 Documentation: broken links to "Further Python fitting examples"
  * [[#17570](https://github.com/root-project/root/issues/17570)] - [ntuple] Missing type name normalization for meta fields
  * [[#17564](https://github.com/root-project/root/issues/17564)] - [RF] Test failure: TestLandauEvil.CompareFixedValuesNorm
  * [[#17515](https://github.com/root-project/root/issues/17515)] - `-isysroot;` in CLING_CXX_PATH_ARGS: erroneous semicolon?
  * [[#17497](https://github.com/root-project/root/issues/17497)] - Particular combination of function overload and std::runtime_error leads to segfault
  * [[#17486](https://github.com/root-project/root/issues/17486)] - Sanity check for Vary is too restrictive
  * [[#17485](https://github.com/root-project/root/issues/17485)] - Need more checks in RDataFrame
  * [[#17472](https://github.com/root-project/root/issues/17472)] - RooEllipse not drawn in notebooks with `%jsroot on`
  * [[#17470](https://github.com/root-project/root/issues/17470)] - Unwanted black line at y == 0 in TMultiGraph with TH1::SetDefaultSumw2()
  * [[#17461](https://github.com/root-project/root/issues/17461)] - `builtin_clang=OFF` broken because of missing transitive dependencies
  * [[#17456](https://github.com/root-project/root/issues/17456)] - Test failures when compiling ROOT with gcc 15
  * [[#17454](https://github.com/root-project/root/issues/17454)] - roottest/root/io/prefetching fails when Davix is disabled
  * [[#17449](https://github.com/root-project/root/issues/17449)] - [DF] Issues with RDataSpec paths using EOS
  * [[#17444](https://github.com/root-project/root/issues/17444)] - ROOT doesn't compile with gcc-15
  * [[#17418](https://github.com/root-project/root/issues/17418)] - Add option to change default basket size in RDataFrame Snapshot
  * [[#17350](https://github.com/root-project/root/issues/17350)] - rootreadspeed should be marked as "CMAKENOEXPORT"
  * [[#17346](https://github.com/root-project/root/issues/17346)] - I/O Customization rules fails in case of changes in the inputs types.
  * [[#17321](https://github.com/root-project/root/issues/17321)] - [RF] Unused Class rule
  * [[#17320](https://github.com/root-project/root/issues/17320)] - [RF] Race when testing RooStats with its tutorials via CTest
  * [[#17305](https://github.com/root-project/root/issues/17305)] - The ONNX.Tile5D test in tmva/sofie/test/TestCustomModelsFromONNX.cxx writes array elements beyond the last element in the array.
  * [[#17295](https://github.com/root-project/root/issues/17295)] - constprefix incorrectly set in TClassEdit
  * [[#17291](https://github.com/root-project/root/issues/17291)] - [RF] Parameter ordering bug in RooFormulaArgStreamer
  * [[#17274](https://github.com/root-project/root/issues/17274)] - Veccore won't compile with clang 19
  * [[#17263](https://github.com/root-project/root/issues/17263)] - [RDF] 'DistRDF.Ranges' has no attribute 'get_clusters_and_entries'
  * [[#17223](https://github.com/root-project/root/issues/17223)] - TFileMerger leaves files open resulting in corrupt metadata
  * [[#17222](https://github.com/root-project/root/issues/17222)] - Regression in Python ownership for histograms within subdirectories with ROOT 6.34.00
  * [[#17214](https://github.com/root-project/root/issues/17214)] - [ntuple] Optimize `RArrayField` reading
  * [[#17190](https://github.com/root-project/root/issues/17190)] - Compiler error (GCC 14.2.0 on Linux)
  * [[#17163](https://github.com/root-project/root/issues/17163)] - Warnings from  the deserialization of RooCrystalBall
  * [[#17145](https://github.com/root-project/root/issues/17145)] - Distributed RDataFrame cannot deal with same column name in different branches
  * [[#17142](https://github.com/root-project/root/issues/17142)] - Check Python code formatting in CI
  * [[#17135](https://github.com/root-project/root/issues/17135)] - Add Alias transformation to distributed RDataFrame
  * [[#17109](https://github.com/root-project/root/issues/17109)] - [PyROOT] False positive in cppyy proxy cache
  * [[#17076](https://github.com/root-project/root/issues/17076)] - Double shadow in `TPaveText`
  * [[#17040](https://github.com/root-project/root/issues/17040)] - Small difference between kp6Violet implementation and official value from Petroff paper
  * [[#16976](https://github.com/root-project/root/issues/16976)] - Strange overflow bin bar when plotting TH1D with X1 option
  * [[#16946](https://github.com/root-project/root/issues/16946)] - Crash in RDF constructor with empty file list
  * [[#16936](https://github.com/root-project/root/issues/16936)] - [ntuple] RClusterPool can crash on non-existing cluster
  * [[#16915](https://github.com/root-project/root/issues/16915)] - `TThreadExecutor::Map` is private, users can only run `MapReduce`.
  * [[#16841](https://github.com/root-project/root/issues/16841)] - Validate `REntry` belonging to right model when reading
  * [[#16794](https://github.com/root-project/root/issues/16794)] - TFormula: Pol functions do not accept variable name as arguments
  * [[#16784](https://github.com/root-project/root/issues/16784)] - Remove default value of p from TH1::GetQuantiles() as is the case with TF1::GetQuantiles
  * [[#16736](https://github.com/root-project/root/issues/16736)] - Please improve documentation and/or argument names for TH1::GetQuantiles()
  * [[#16725](https://github.com/root-project/root/issues/16725)] - Pyroot crashes reading TClonesArray in a TTree
  * [[#16720](https://github.com/root-project/root/issues/16720)] - 4 TMVA test requires BLAS library but run even when it was not found.
  * [[#16687](https://github.com/root-project/root/issues/16687)] - Loss of floating point precision when saving TCanvas as ROOT macro
  * [[#16487](https://github.com/root-project/root/issues/16487)] - Doc for `RInterface::Take` isn't clear about `column` parameter
  * [[#16312](https://github.com/root-project/root/issues/16312)] - Broken streaming of vector of enum with underlying type other than int
  * [[#16189](https://github.com/root-project/root/issues/16189)] - TFile::k630forwardCompatibility does not apply to new files correctly
  * [[#16146](https://github.com/root-project/root/issues/16146)] - [ntuple] Additional information in RNTupleInspector::PrintColumnTypeInfo()
  * [[#16034](https://github.com/root-project/root/issues/16034)] - [ntuple] Unable to use `std::vector` in RDF
  * [[#15959](https://github.com/root-project/root/issues/15959)] - [RF] Make Offset(“bin”) usable for CLs method
  * [[#15538](https://github.com/root-project/root/issues/15538)] - Enable additional LLVM checks
  * [[#15364](https://github.com/root-project/root/issues/15364)] - [Win32] tutorial-roostats-rs101_limitexample-py fails with a segfault
  * [[#15267](https://github.com/root-project/root/issues/15267)] - vc.modulemap created (and registered) when ROOT built without support for Vc
  * [[#15091](https://github.com/root-project/root/issues/15091)] - Clad is an optional module but TFormula (and tests of) fails without it.
  * [[#15082](https://github.com/root-project/root/issues/15082)] - [DF] In Histo2D, support different types of Data and Weights columns
  * [[#14953](https://github.com/root-project/root/issues/14953)] - [cmake] Glob when copying headers and tutorials
  * [[#14766](https://github.com/root-project/root/issues/14766)] - RResulPtr could better convey/share ownership of pointee
  * [[#14583](https://github.com/root-project/root/issues/14583)] - [ROOT_5159] Improve TTree documentation about SetMakeClass()
  * [[#14557](https://github.com/root-project/root/issues/14557)] - [ROOT-4550] TMessage doesn't honour kIsOwner bit when compression is used
  * [[#14556](https://github.com/root-project/root/issues/14556)] - [ROOT-3452] Suggestions for Minimizer documentation
  * [[#14425](https://github.com/root-project/root/issues/14425)] - PyROOT calls into deleted copy-constructor in valid C++ scenarios
  * [[#14304](https://github.com/root-project/root/issues/14304)] - [math] Finite difference methods for Gradient
  * [[#14007](https://github.com/root-project/root/issues/14007)] - Cannot create a RNtuple into a TDirectory
  * [[#13880](https://github.com/root-project/root/issues/13880)] - Stats box content changes after zoom/unzoom with mouse wheel
  * [[#13800](https://github.com/root-project/root/issues/13800)] - [DF] RDatasetSpec allows a global range larger than the actual dataset
  * [[#12673](https://github.com/root-project/root/issues/12673)] - [ntuple] Crash when using RNTuple at prompt
  * [[#12537](https://github.com/root-project/root/issues/12537)] - Segmentation violation when trying to open particular cycle using TBrowser
  * [[#12510](https://github.com/root-project/root/issues/12510)] -  Issue with `hadd` when first file has empty tree
  * [[#12505](https://github.com/root-project/root/issues/12505)] - CI: PR should "fail" in case of warnings
  * [[#12497](https://github.com/root-project/root/issues/12497)] - SetShowProjection highlights wrong bins on 2d histogram with logarithmic axis
  * [[#12251](https://github.com/root-project/root/issues/12251)] - Problems with `TH1::GetQuantiles`
  * [[#12225](https://github.com/root-project/root/issues/12225)] - [RF] Allowing to use AddPreprocessFunction for shape factors in HistFactory
  * [[#11959](https://github.com/root-project/root/issues/11959)] - RDataFrame member function RDataFrame::GraphErrors
  * [[#11675](https://github.com/root-project/root/issues/11675)] - `rootls` should not show errors when seeing a branch of class without dictionary
  * [[#11448](https://github.com/root-project/root/issues/11448)] - Irrelevant binaries in CMake export set
  * [[#10948](https://github.com/root-project/root/issues/10948)] - Two ClassImp statements in the same compilation unit result in a redefinition error
  * [[#10774](https://github.com/root-project/root/issues/10774)] - [TTree] Inconcistency with the TTreeReader's SetEntries.
  * [[#10769](https://github.com/root-project/root/issues/10769)] - [hist] Use correct rounding for parameters in TFormula::GetExpFormula
  * [[#10749](https://github.com/root-project/root/issues/10749)] - RNTupleDS should check for column type mismatches
  * [[#10102](https://github.com/root-project/root/issues/10102)] - `hadd` segfaults when the output file is too large
  * [[#10024](https://github.com/root-project/root/issues/10024)] - `TTree::SaveAs` should error out when using with unsupported format.
  * [[#10019](https://github.com/root-project/root/issues/10019)] - [ntuple] Add support for automatic schema evolution and I/O rules
  * [[#9335](https://github.com/root-project/root/issues/9335)] - [hist] TGraphMultiErrors: needs some improvements in interface and  better documentation
  * [[#9319](https://github.com/root-project/root/issues/9319)] - [TTree] Print("clusters") off-by-one error in case of partial clusters
  * [[#8951](https://github.com/root-project/root/issues/8951)] - Add optional flag to include underflow and overflow weights when returning sum of weights from histogram
  * [[#7685](https://github.com/root-project/root/issues/7685)] - [DF] `AsNumpy` should automatically convert RVecs into numpy arrays
  * [[#7567](https://github.com/root-project/root/issues/7567)] - TChain fails to read events if: only a single file is processed
  * [[#7254](https://github.com/root-project/root/issues/7254)] - [RF] RooHypatia2 Analytical integral integration
  * [[#6682](https://github.com/root-project/root/issues/6682)] - Add an option to draw "XKCD" plot in ROOT7
  * [[#6640](https://github.com/root-project/root/issues/6640)] - TFileMerger output file is deleted when TTree::ChangeFile is triggered by reaching a file size greater than TTree::GetMaxTreeSize
  * [[#6607](https://github.com/root-project/root/issues/6607)] - TClassEdit::GetNormalizedName does not strip std::allocator on Windows
  * [[ROOT-11020](https://its.cern.ch/jira/browse/11020)] - Add function to evaluate expressions using fit parameters
  * [[ROOT-10907](https://its.cern.ch/jira/browse/10907)] - Allow to pass non-integer numbers of events to TEfficiency::SetTotalEvents and TEfficiency::SetPassedEvents methods
  * [[ROOT-10823](https://its.cern.ch/jira/browse/10823)] - [TTreeReader] Add method to check whether TTreeReaderArray contents are contiguous
  * [[ROOT-10780](https://its.cern.ch/jira/browse/10780)] - [cling] Cannot use trailing return types at prompt
  * [[ROOT-10621](https://its.cern.ch/jira/browse/10621)] - Segfault if TFile is used with TRint in teardown
  * [[ROOT-10620](https://its.cern.ch/jira/browse/10620)] - Cling gets confused when assigning the result of a malformed template expression
  * [[ROOT-10553](https://its.cern.ch/jira/browse/10553)] - TSQLStatement::GetBinary is not consistently implemented
  * [[ROOT-10537](https://its.cern.ch/jira/browse/10537)] - CMakeList.txt environment cleanup inaccurate
  * [[ROOT-10482](https://its.cern.ch/jira/browse/10482)] - pullHist and residHist biased (or sampling biased)
  * [[ROOT-10249](https://its.cern.ch/jira/browse/10249)] - TDataMember::GetOptions returns empty list for enum members
  * [[ROOT-10239](https://its.cern.ch/jira/browse/10239)] - rootcling crashes on -Werror + unknown warning flag to clang
  * [[ROOT-9886](https://its.cern.ch/jira/browse/9886)] - TTreeReader loads wrong entry from chain friend
  * [[ROOT-9833](https://its.cern.ch/jira/browse/9833)] - TMVA crashes adding single event w/o prior var def
  * [[ROOT-9753](https://its.cern.ch/jira/browse/9753)] - UB Sanitizer Complaining About clang shared_ptr issue
  * [[ROOT-9688](https://its.cern.ch/jira/browse/9688)] - Be more specific about which tutorials/ to copy
  * [[ROOT-9508](https://its.cern.ch/jira/browse/9508)] - rootcling web documentation incomplete and out of date
  * [[ROOT-9204](https://its.cern.ch/jira/browse/9204)] - Impossible to use RooFrame or RooCurve chiSquare method if data histogram has option XErrorSize(0)
  * [[ROOT-9013](https://its.cern.ch/jira/browse/9013)] - Get rid of AbstractMethod()
  * [[ROOT-8775](https://its.cern.ch/jira/browse/8775)] - TTree::MakeSelector can produce invalid C++ code
  * [[ROOT-8725](https://its.cern.ch/jira/browse/8725)] - Thread-unsafe statics in GSLInterpolator
  * [[ROOT-8397](https://its.cern.ch/jira/browse/8397)] - TGraphSmooth::Approx graph misbehaves around the maximal "x" value
  * [[ROOT-8278](https://its.cern.ch/jira/browse/8278)] - roofit asymmetry plots create object names which fail when saving plot as macro
  * [[ROOT-8240](https://its.cern.ch/jira/browse/8240)] - Must not unload or reload cling runtime universe
  * [[ROOT-8162](https://its.cern.ch/jira/browse/8162)] - Can not compile ROOT scripts when compiling ROOT in a temporary directory.
  * [[ROOT-8112](https://its.cern.ch/jira/browse/8112)] - not possible to add filename to TChain which doesn't end in .root, with treename
  * [[ROOT-7926](https://its.cern.ch/jira/browse/7926)] - TTree::Print("toponly") does not account correctly top branches
  * [[ROOT-7855](https://its.cern.ch/jira/browse/7855)] - Inconsistent behaviour when cloning a tree using TChain and TTree
  * [[ROOT-7626](https://its.cern.ch/jira/browse/7626)] - TRegexp::MakeWildcard should support escape sequence
  * [[ROOT-7372](https://its.cern.ch/jira/browse/7372)] - Accessing complex map branches crashes in PyROOT
  * [[ROOT-7322](https://its.cern.ch/jira/browse/7322)] - Missing Overload for TF1 GetParError
  * [[ROOT-7067](https://its.cern.ch/jira/browse/7067)] - tree->GetMaximum() not working with TChain and TEntryList
  * [[ROOT-6874](https://its.cern.ch/jira/browse/6874)] - Suggested function TF1::EvalUncertainty()
  * [[ROOT-6636](https://its.cern.ch/jira/browse/6636)] - Tab completion fails for CV-qualified pointers and objects
  * [[ROOT-5820](https://its.cern.ch/jira/browse/5820)] - Multiple issues with TFileMerger in single file case
  * [[ROOT-5588](https://its.cern.ch/jira/browse/5588)] - GetEntries moves state of fReadEntry in TTree
  * [[ROOT-5439](https://its.cern.ch/jira/browse/5439)] - Dump-output of TH1 not showing pointerness of fArray
  * [[ROOT-5137](https://its.cern.ch/jira/browse/5137)] - Enhance error recover in TTreeCloner
  * [[ROOT-4663](https://its.cern.ch/jira/browse/4663)] - GetFromPipe() loses the return value
  * [[ROOT-4012](https://its.cern.ch/jira/browse/4012)] - TTree::SetAlias() fail to interpret the constant
  * [[ROOT-118](https://its.cern.ch/jira/browse/118)] - Implement support for access to nested objects

## Release 6.36.02

### Items addressed in this release

* [#18665](https://github.com/root-project/root/issues/18665) - `TFileMerger`'s (and `hadd`'s) default behavior on encountering invalid/corrupt objects while merging has changed: previously the corrupt object would be skipped without aborting the merging process; now the merging process will abort by default. The "skip" behavior can be restored by:
  * calling the new method `TFileMerger::SetErrorBehavior()` (if using `TFileMerger` directly), or
  * using the `-k` flag if using `hadd`.
 
