% ROOT Version 6.36 Release Notes
% 2025-05
<a name="TopOfPage"></a>

## Introduction

ROOT version 6.36.00 is scheduled for release at the end of May 2025.

For more information, see:

[http://root.cern](http://root.cern)

The following people have contributed to this new version:

 Bertrand Bellenot, CERN/EP-SFT,\
 Jakob Blomer, CERN/EP-SFT,\
 Philippe Canal, FNAL,\
 Mattias Ellert, Uppsala University,\
 Florine de Geus, CERN/University of Twente,\
 Fernando Hueso Gonzalez, CSIC/University of Valencia,\
 Enrico Lusiani, INFN Padova,\
 Alberto Mecca, University of Turin,\
 Lorenzo Moneta, CERN/EP-SFT,\
 Mark Owen, University of Glasgow,\
 Vincenzo Eduardo Padulano, CERN/EP-SFT,\
 Giacomo Parolini, CERN/EP-SFT,\
 Danilo Piparo, CERN/EP-SFT,\
 Jonas Rembser, CERN/EP-SFT,\
 Manuel Tobias Schiller, University of Glasgow,\
 Surya Somayyajula, UMass Amherst,\
 Petr Stepanov, @petrstepanov,\
 Dongliang Zhang, University of Science and Technology of China

## Deprecation and Removal

* The RooFit legacy interfaces that were deprecated in ROOT 6.34 and scheduled for removal in ROOT 6.36 are removed. See the RooFit section in the 6.34 release notes for a full list.
* The `TPython::Eval()` function that was deprecated in ROOT 6.34 and scheduled for removal in ROOT 6.36 is removed.
* The `RooDataSet` constructors to construct a dataset from a part of an existing dataset are deprecated and will be removed in ROOT 6.38. This is to avoid interface duplication. Please use `RooAbsData::reduce()` instead, or if you need to change the weight column, use the universal constructor with the `Import()`, `Cut()`, and `WeightVar()` arguments.
* The ROOT splash screen was removed for Linux and macOS
* Proof support has been completely removed form RooFit and RooStats, after it was already not working anymore for several releases
* The build options `mysql`, `odbc`, `pgsql` have been deprecated. Please complain with root-dev@cern.ch should you still need one!
* The build options `html` and `qt5web` have been removed.
* The ClassImp macro and derived macros have no effect now and will be deprecated in ROOT 6.38.
* The default TTreeFormula constructor (without arguments) is now deleted, since it lead to an unusable and unsafe object. Instead, this implementation has been reserved now for ROOT I/O exclusively via the TRootIOCtor argument tag.
* The `RooStats::HLFactory` class was deprecated will be removed in ROOT 6.38. It provided little advantage over using the RooWorkspace directly or any of the other higher-level frameworks that exist in the RooFit ecosystem.

## Python Interface

## RDataFrame
- When running multiple computation graphs run concurrently using [`RunGraphs()`](https://root.cern/doc/master/namespaceROOT_1_1RDF.html#a526d77d018bf69462d736bbdd1a695c4),
  the pool of slot numbers that a thread can pick from is now shared across all graphs. This enables use cases where a single resource, which may be expensive to create or copy,
  is shared across tasks belonging to different computation graphs.

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


## I/O

* Honour the `Davix.GSI.CACheck` parameter also in the `ROOT::Internal::RRawFileDavix` class.
* Added support for `enum class` with a non default underlying size, for example `enum smallenum: std::int16_t`.  The default is 32 bits.  All enums, independently of their in memory size are stored on file using 32 bits to enable forward compatibility of the file; files created with a `enum class` with a non default underlying size can be read with old version of ROOT into a `enum` type of default size.
* * Note: `enum class` with an underlying size strictly greater than 32 bits are not supported since they would be truncated when stored on file.
* The version number of `TStreamerInfo` has been increase to 10 to encoded the addition of the support for `enum class` with a non default underlying size. This allows the opportunity to detect files written by old version of ROOT (`v9` and older of `TStreamerInfo`) where  `enum class` with a non default underlying size where stored incorrectly but recoverably.  Those files can be recover by using I/O customization rules that takes in consideration their size at the time of writing (this information is not recorded in the `ROOT` file).  See https://github.com/root-project/root/pull/17009#issuecomment-2522228598 for some examples.
* New attribute for I/O customization rules: `CanIgnore`.  When using this attribute the rule will be ignored if the input is missing from the schema/class-layout they apply to instead of issue a `Warning`

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

## RDataFrame

## Tutorials and Code Examples

## Core

## Histograms

## Math

## Graphics

## Geometry

## Montecarlo

## JavaScript ROOT

## Class Reference Guide

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
