% ROOT Version 6.16 Release Notes
% 2019-01-23
<a name="TopOfPage"></a>

## Introduction

ROOT version 6.16/00 was released on January 23, 2019.

For more information, see:

[http://root.cern](http://root.cern)

The following people have contributed to this new version:

 Kim Albertsson, CERN/ATLAS,\
 Guilherme Amadio, CERN/SFT,\
 Bertrand Bellenot, CERN/SFT,\
 Iliana Betsou, CERN/SFT,\
 Brian Bockelman, UNL,\
 Rene Brun, CERN/SFT,\
 Philippe Canal, FNAL,\
 Olivier Couet, CERN/SFT,\
 Gerri Ganis, CERN/SFT,\
 Andrei Gheata, CERN/SFT,\
 Enrico Guiraud, CERN/SFT,\
 Stephan Hageboeck, CERN/SFT,\
 Siddhartha Rao Kamalakara, GSOC, \
 Sergey Linev, GSI,\
 Pere Mato, CERN/SFT,\
 Lorenzo Moneta, CERN/SFT,\
 Alja Mrak Tadel, UCSD/CMS,\
 Axel Naumann, CERN/SFT,\
 Danilo Piparo, CERN/SFT,\
 Fons Rademakers, CERN/SFT,\
 Enric Tejedor Saavedra, CERN/SFT,\
 Oksana Shadura, UNL,\
 Ravi Kiran Selvam, GSOC, \
 Manos, Stergiadis, GSOC, \
 Matevz Tadel, UCSD/CMS,\
 Yuka Takahashi, Princeton,\
 Massimo Tumolo, Politecnico di Torino,\
 Mohammad Uzair, CERN/SFT, \
 Xavier Valls, CERN/SFT,\
 Vassil Vassilev, Princeton/CMS,\
 Wouter Verkerke, NIKHEF/Atlas,\
 Stefan Wunsch, CERN/SFT

## Deprecation and Removal

### Ruby bindings

The ruby binding has been unmaintained for several years; it does not build with current ruby versions.
Given that this effectively meant that Ruby was dysfunctional and given that nobody (but package maintainers) has complained, we decided to remove it.

### Removal of previously deprecated or disabled packages

The packages `afs`, `chirp`, `glite`, `sapdb`, `srp` and `ios` have been removed from ROOT.
They were deprecated before, or never ported from configure, make to CMake.

### Remove GLUtesselator forward declaration from TVirtualX.h

It was never used in TVirtualX interfaces. If GLUtesselator forward declaration is required, use TGLUtil.h include instead.

### Deprecation of ROOT packages

*Note*: while the deprecation was emitted by `cmake` upon enabling these packages, the original release notes (of 6.16/00) did not mention the deprecation. Apologies for missing this!

The following packages have been deprecated in 6.16 and will be removed in 6.18:
afdsmgrd bonjour castor geocad globus hdfs krb5 ldap memstat odbc qt qtgsi rfio table


## C++ Modules Technology Preview

ROOT has several features which interact with libraries and require implicit
header inclusion. This can be triggered by reading or writing data on disk,
or user actions at the prompt. Often, the headers are immutable and reparsing is
redundant. C++ Modules are designed to minimize the reparsing of the same
header content by providing an efficient on-disk representation of C++ Code.

This is an experimental feature which can be enabled by compiling ROOT with
`-Druntime_cxxmodules=On`. You can read more about the current state of the
feature [here](https://github.com/root-project/root/blob/v6-16-00-patches/README/README.CXXMODULES.md).

## Core Libraries

### New command line flag `--version` for `root`

`root --version` now displays ROOT version and build info and quits:

```
ROOT Version: 6.16/00
Built for linuxx8664gcc on Jan 23 2019, 11:04:35
From tags/v6-16-00@v6-16-00
```

### Fish support for thisroot script

`. bin/thisroot.fish` sets up the needed ROOT environment variables for one of the ROOT team's favorite shells, the [fish shell](https://fishshell.com/).

### Change of setting the compression algorithm in `rootrc`

The previous setting called `ROOT.ZipMode` is now unused and ignored.
Instead, use `Root.CompressionAlgorithm` which sets the compression algorithm according to the values of [ECompression](https://root.cern/doc/master/Compression_8h.html#a0a7df9754a3b7be2b437f357254a771c):

* 0: use the default value of `R__ZipMode` (currently selecting ZLIB)
* 1: use ZLIB (the default until 6.12 and from 6.16)
* 2: use LZMA
* 3: legacy, please don't use
* 4: LZ4

### TRef

* Improve thread scalability of `TRef`. Creating and looking up a lot of `TRef` from the same `processID` now has practically perfect weak scaling.

### Parallelism
* Upgrade the built-in TBB version to 2019_U1.

### Type System
* Upgrade the `TClass::GetMissingDictionaries` method to support `std::unique_ptr`, `std::array` and `std::tuple` without getting trapped in the internal STL implementation details.

## I/O Libraries

* To allow for increase run-time performance and increase thread scalability the override ability of `TFile::GetStreamerInfoList` is replaced by an override of `TFile::GetStreamerInfoListImp` with updated return type and arguments.   If a class override `TFile::GetStreamerInfoList` you will now see a compilation error like:

```
/opt/build/root_builds/rootcling.cmake/include/TSQLFile.h:225:19: error: declaration of 'GetStreamerInfoList' overrides a 'final' function
virtual TList *GetStreamerInfoList();
^
/opt/build/root_builds/rootcling.cmake/include/TFile.h:231:24: note: overridden virtual function is here
virtual TList      *GetStreamerInfoList() final; // Note: to override behavior, please override GetStreamerInfoListImpl
^
```

Instead you need to override the protected method:

```
InfoListRet GetStreamerInfoListImpl(bool lookupSICache);
```

which can be implemented as

```
InfoListRet DerivedClass::GetStreamerInfoListImpl(bool /*lookupSICache*/) {
ROOT::Internal::RConcurrentHashColl::HashValue hash;
TList *infolist = nullptr;
//
// Body of the former Derived::GetStreamerInfoList with the
// return statement replaced with something like:

// The second element indicates success or failure of the load.
// (i.e. {nullptr, 0, hash} indicates the list has already been processed
//  {nullptr, 1, hash} indicates the list failed to be loaded
return {infolist, 0, hash};
}
```

See `TFile::GetStreamerInfoListImpl` implementation for an example on how to implement the caching.

* ZLIB (with compression level 1) is now the default compression algorithm for new ROOT files (LZ4 was default compression algorithm in 6.14). Because of reported "corner cases" for LZ4, we are working on the fix to be landed in a next release and return back LZ4 as a default compression algorithm.

* Introducing a possibility for ROOT to use generic compression algorithm/level/settings, by introducing new generic class RCompressionSetting together with new structs ELevel (compression level), EDefaults (default compression settings) and EAlgorithm (compression algorithm). These changes are the first step in generalization of setup of ROOT compression algorithm. It also provides correctness of resolution of compression level and compression algorithm from defined ROOT compression settings:

```
Attaching file hsimple.root as _file0...
(TFile *) 0x55f79b0e6360
root [1] _file0->GetCompressionAlgorithm()
(int) 1
root [2] _file0->GetCompressionLevel()
(int) 1
root [3] _file0->GetCompressionSettings()
(int) 101
root [4]
```

## TTree Libraries
### RDataFrame
  - Optimise the creation of the set of branch names of an input dataset, doing the work once and caching it in the RInterface.
  - Add [StdDev](https://root.cern/doc/v616/classROOT_1_1RDF_1_1RInterface.html#a482c4e4f81fe1e421c016f89cd281572) action.
  - Add [Display](https://root.cern/doc/v616/classROOT_1_1RDF_1_1RInterface.html#aee68f4411f16f00a1d46eccb6d296f01) action and [tutorial](https://github.com/root-project/root/blob/v616/tutorials/dataframe/df024_Display.C).
  - Add [Graph](https://root.cern/doc/v616/classROOT_1_1RDF_1_1RInterface.html#a804b466ebdbddef5c7e3400cc6b89301) action and [tutorial](https://github.com/root-project/root/blob/v616/tutorials/dataframe/df021_createTGraph.C).
  - Improve [GetColumnNames](https://root.cern/doc/v616/classROOT_1_1RDF_1_1RInterface.html#a951fe60b74d3a9fda37df59fd1dac186) to have no redundancy in the returned names.
  - Add [Kahan Summation tutorial](https://github.com/root-project/root/blob/v616/tutorials/dataframe/df022_useKahan.C) to subscribe a Kahan summation action to the RDataFrame.
  - Add [Cache tutorial](https://github.com/root-project/root/blob/v616/tutorials/dataframe/df019_Cache.C) [Python version](https://github.com/root-project/root/blob/v616/tutorials/dataframe/df019_Cache.py).
  - Add [Aggregate tutorial](https://github.com/root-project/root/blob/v616/tutorials/dataframe/df023_aggregate.C).
  - Fix ambiguous call on Cache() with one or two columns as parameters.
  - Add [GetFilterNames](https://root.cern/doc/v616/classROOT_1_1RDF_1_1RInterface.html#a25026681111897058299161a70ad9bb2).
  - Improve RDF node ownership model. The net effect is that users do not have to worry about keeping the first node of a computation graph in scope anymore.
  - Make RResultPtr copy/move-assignable and copy/move-constructible.
  - Add [GetColumnType](https://root.cern/doc/v616/classROOT_1_1RDF_1_1RInterface.html#ad3ccd813d9fed014ae6a080411c5b5a8a) utility method to query the type of a RDF column (returned as a string).
  - Add [PassAsVec](https://root.cern/doc/v616/namespaceROOT_1_1RDF.html#a1ecc8a41e8f12e65e1bf0d2e65aec36d) helper function.
  - Add [SaveGraph](https://root.cern/doc/v616/namespaceROOT_1_1RDF.html#adc17882b283c3d3ba85b1a236197c533) helper function to write out the RDF computation graph as a graphviz file.
  - Add a [tutorial for RDataFrame helper functions](https://root.cern/doc/v616/df020__helpers_8C.html).
  - Throw if name of a custom column is not a valid C++ name.
  - Allow every RDataFrame variable be cast to a common type `ROOT::RDF::RNode`.
  - Speed up just-in-time compilation (and therefore runtime) of Snapshots with a large number of branches.
  - Create names for histograms and graphs based on the input columns if no model is provided.
  - RCutFlowReport can print cumulative efficiency of cuts.
  - Reading and writing of columns holding `vector<bool>` instances and `bool` C arrays.
  - Support `rdfentry_` and `rdfslot_` implicit columns (`tdfentry_` and `tdfslot_` kept for backwards compatibility).
  - Remove `RDataFrame` from the 32-bit builds.
  - Speed up interpreted usage of RDataFrame (i.e. in macros or from ROOT prompt) by removing certain cling runtime safety checks.
  - Streamline and better document usage of multi-thread RDataFrame: edge cases in which processing of an event could start
    before processing of another event finished have been removed, making it easier for user to write safe parallel RDF operations.
    See the [relevant documentation](https://root.cern.ch/doc/v616/classROOT_1_1RDataFrame.html#parallel-execution) for more information.
  - In multi-thread runs, `rdfentry_` will not take values corresponding to the underlying TChain's global entries anymore.
  - Allow to read sqlite files through the [RSqliteDS](https://root.cern/doc/v616/classROOT_1_1RDF_1_1RSqliteDS.html).

### TTreeProcessorMT
  - Parallelise search of cluster boundaries for input datasets with no friends or TEntryLists. The net effect is a faster initialization time in this common case.
  - Handle gracefully the presence of chains the files associated to which are corrupted.
  - Reduce number of expensive `TChain::LoadTree` calls by spawning nested TBB tasks to ensure clusters of a given file will be most likely processed by the same thread.

### TTree
  - TTrees can be forced to only create new baskets at event cluster boundaries.
    This simplifies file layout and I/O at the cost of memory.  Recommended for
    simple file formats such as ntuples but not more complex data types.  To
    enable, invoke `tree->SetBit(TTree::kOnlyFlushAtCluster)`.

## Math Libraries

  - The built-in VDT library version has been changed from v0.4.1 to v0.4.2

### [Clad](https://github.com/vgvassilev/clad)
  - Enable experimental automatic differentiation techniques to compute
    derivatives and gradients of functions. Automatic differentiation is
    superior to the slow symbolic or often inaccurate numerical differentiation.
    It uses the fact that every computer program can be divided into a set of
    elementary operations (-,+,*,/) and functions (sin, cos, log, etc). By
    applying the chain rule repeatedly to these operations, derivatives of
    arbitrary order can be computed.
  - Implement experimental `TFormula::GradientPar` derivative engine which
    employs clad.

### VecOps
  - Add `All` helper: return true if all of the elements equate to true, return false otherwise.
  - Add `Any` helper: return true if any of the elements equates to true, return false otherwise.
  - Add `ArgSort` helper: return an RVec of indices that sort the vector.
  - Add `Combinations` helper which can:
    - return the indices which represent all combinations of the elements of two vectors.
    - return the indices which represent all unique n-tuple combinations of the elements of a given vector.
  - Add `Intersect` helper: return the intersection of elements of two RVecs.
  - Add `Nonzero` helper: return the indices of the elements which are not zero
  - Add `Reverse` helper: return copy of reversed RVec.
  - Add `Sort` helper: return copy of vector with elements sorted in ascending order (also according to a user defined predicate)
  - Add `Take` helper which can:
    - return elements of a RVec at given indices.
    - return first elements or last elements of an RVec.
  - Add `Where` helper which can:
    - return the elements of v1 if the condition c is true and v2 if the condition c is false.
    - return the elements of v1 if the condition c is true and sets the value v2 if the condition c is false.
    - return the elements of v2 if the condition c is false and sets the value v1 if the condition c is true.
    - return a vector with the value v2 if the condition c is false and sets the value v1 if the condition c is true.

## RooFit Libraries
  - Add value printer for RooAbsArg and daughters.
  - Add a Python version for the majority of the Tutorials.

## TMVA Library

### Deep Learning

This release contains several fixes and improvement for the `MethodDL`. The `MethodDL` is also now the recommended class to use for Deep Learning in TMVA and is replacing the previous existing
`MethodDNN`, which is still available, but it has a limited functionality and it supports only dense layer.

The new features of `MethodDL` are:

 - Support training and evaluation of Convolutional layer on GPU
 - Several ML optimizers are now included and they can be used in addition to SGD. These are ADAM (the new default), ADAGRAD,
RMSPROP, ADADELTA. A new option, *Optimizer* has been added in the option string used to define the training strategy options.
 - Add support for regression in MethodDL
 - Use single precision (float) types as the fundamental type for the neural network architecture. Double precision could be enabled, but it will require recompiling TMVA.
 - Support inference (network evaluation) in batch mode in addition to single event. Batch mode evaluation is now the default when used within the `TMVA::Factory` class (i.e. when calling
`Factory::TestAllMethod()` or `Factory::EvaluateAllMethods()`
 - Support splitting the overall training data in Train and Validation data. The train data is used for finding the optimal network weight and the validation data is used to monitor the validation
error. The weights which are giving a minimal validation error will be stored. For the splitting a new option, *ValidationSize* has been added to the global options for `MethodDL`.
The same option is also available in the `PyKeras` method of `PyMVA`
 - The fast tanh implementation from VDT is now used as activation function when training the network on CPU.
 - Using `Cblas` from the GSL library is supported for CPU training when no other Blas libraries are found. However, it is strongly recommended, to use an optimized Blas implementation such as `libopenblas`, that is
available in cvmfs.
 - Add several performance optimizations for both CPU and GPU versions of `MethodDL`.


### Other New TMVA Features

 - Add a new option to the `DataLoader` to switch off computation of correlation matrix. The new option is called *CalcCorrelations* and it should be used when a large number of input variables are
  provided, otherwise TMVA will spend a long time in setting up the data set before training.

 - Build configuration:
   - Add new cmake flags, `tmva-cpu` and `tmva-gpu`, which can be used to switch on/off the CPU and GPU (based on CUDA) implementations of the TMVA Deep Learning module. `tmva-cpu` is enabled by
   default if a Blas or CBlas library is found in the system. `tmva-gpu` is enabled when the cmake flag `cuda` is enabled and a compatible Cuda library is found.
   enabled if the corre
   - Add possibility to independently configure building of optional pymva part of tmva with flag `-Dpymva=ON|OFF`.

 - New Cross Validation features:
   - Add stratified splitting for cross validation.
   - New plotting option in cross validation, average ROC curve.

 - Bugfixes:
   - Fix bug in BDT training with imt=on
   - Improved handling of large event numbers in cross validation using deterministic splitting

 - Documentation:
   - Update TMVA Users' guide

## 2D Graphics Libraries

  - Highlight mode is implemented for `TH1` and for `TGraph` classes. When
    highlight mode is on, mouse movement over the bin will be represented
    graphically. Histograms bins or graph points will be highlighted. Moreover,
    any highlight emits signal `TCanvas::Highlighted()` which allows the user to
    react and call their own function. For a better understanding see also
    the tutorials `$ROOTSYS/tutorials/hist/hlHisto*.C` and
    `$ROOTSYS/tutorials/graphs/hlGraph*.C` .
  - Implement fonts embedding for PDF output. The "EmbedFonts" option allows to
    embed the fonts used in a PDF file inside that file. This option relies on
    the "gs" command (https://ghostscript.com).

    Example:

~~~ {.cpp}
   canvas->Print("example.pdf","EmbedFonts");
~~~
  - In TAttAxis::SaveAttributes` take into account the new default value for `TitleOffset`.
  - When the histograms' title's font was set in pixel the position of the
    `TPaveText` containing the title was not correct. This problem was reported
    [here](https://root-forum.cern.ch/t/titles-disappear-for-font-precision-3/).
  - In `TGraph2D` when the points were all in the same plane (along X or Y) at a
    negative coordinate, the computed axis limits were not correct. This was reported
    [here](https://root-forum.cern.ch/t/potential-bug-in-tgraph2d/29700/5).
  - Implemented the drawing of filled polygons in NDC space as requested
    [here](https://sft.its.cern.ch/jira/browse/ROOT-9523)
  - Implement the drawing of filled polygons in NDC space.
  - When drawing a histogram with the automatic coloring options (PMC, PLC etc ...)
    it was easy to forget to add a drawing option. This is now fixed. If no drawing
    option is specified the default drawing option for histogram is added.
  - When drawing a `TGraph` if `TH1::SetDefaultSumw2()` was on, then the underlying
    histogram used to draw the `TGraph` axis was created with errors and therefore
    the histogram painter painted it with errors which was a non sense in that
    particular case. This is now fixed. It was discussed
    [here](https://root-forum.cern.ch/t/horizontal-line-at-0-on-y-axis/30244/26)
  - Add `TGraph2D::GetPoint`, with similar interface and behaviour as `TGraph::GetPoint`
  - In the statistics painting for 2D histograms, the central cell of
    the underflow/overflow grid was not properly rendered for very large contents.
    This problem was reported [here](https://root-forum.cern.ch/t/stat-box-for-th2/).

## 3D Graphics Libraries

  - Prototype / demonstrator of EVE-7 (enabled with ROOT-7 build). See code in `graf3d/eve7/` and tutorials in `tutorials/eve7/`.


## Web Graphics Libraries

  - Introduce web-based output for TCanvas. When ROOT compiled with cxx14 support `cmake -Dcxx14=ON ..` and
    started with web option like `root --web hsimple.C`, TCanvas will be displayed in web browser.
  - Most of histograms and graphs classes are supported. See
    [JavaScript ROOT website](https://root.cern/js/latest/examples.htm) for list of supported classes
  - Also some classes with customize painters may be working - like TTree::Draw() into TParallelCoord object. These kind of
    objects handled on server side, web browser used just for display of produced primitives like polylines or text


## Language Bindings

The Ruby binding has been removed.


### PyROOT
  - Fixed support for templated functions when in need of:
    - typedef resolution (`Foo<Float_t>` -> `Foo<float>`)
    - namespace addition (`Foo<vector<float>>` -> `Foo<std::vector<float>>`)
    - full name completion (`Foo<std::vector<float>>` -> `Foo<std::vector<float, std::allocator<float>>>`)

### Experimental PyROOT
  - Added pythonisations for `TTree` and its subclasses (e.g. `TChain`, `TNtuple`)
    - Pythonic iterator (`for event in tree:`)
    - Access tree branches as attributes (`mytree.mybranch`)
    - `TTree::Branch` pythonisation
    - `TTree::SetBranchAddress` pythonisation
  - Added pythonisations for `TDirectory` and its subclasses (e.g `TFile`, `TDirectoryFile`)
    - Access directories/objects in `TDirectory`/`TDirectoryFile`/`TFile` as attributes
    (`mydir1.mydir2.myhist`, `myfile.myhist`, `myfile.mydir.myhist`)
    - `TDirectory::Get` pythonisation
    - `TDirectory::WriteObject` pythonisation
    - `TFile::Open` pythonisation
  - Added pretty printing generic pythonisation for all classes
  - Added interoperability with NumPy arrays for STL vectors and `RVec`s (zero-copy wrapping of
  vectors and `RVec`s into NumPy arrays)

### Jupyter Notebook Integration
  - Make sure the ROOT C++ Jupyter kernel runs with the same Python version (major and minor) that ROOT
  was built with.
  - Make the Jupyter server started with `root --notebook` listen on all interfaces. This can be useful
  if the user wants to connect to the server remotely. It also fixes an issue observed when starting
  the Jupyter server inside a Docker container.

## JavaScript ROOT
  - Support of TWebCanvas functionality. Code for ROOT 6.16 will
    be maintained in v6-16-00-patches branch of JSROOT repository.
  - Significant speed up (factor 10) when drawing canvas with many subpads
  - Many small improvements and bug fixes, see JSROOT release notes for v5.4.2 - v5.6.2

## Tutorials
  - Refurbish text in the `RDataFrame` tutorials category.

## Command line tools
  - Fixed `rooteventselector` when both applying a cut (based on branch values) and selecting only
  a subset of the branches. Previously, the size of the output file was bigger than expected.


## Build, Configuration and Testing Infrastructure

  - The required version of CMake has been updated from 3.4.3 to 3.6. This is
    necessary because both Vc and VecCore builtins require it, and ROOT uses
    some features from newer CMake versions, namely `list(FILTER...)`.


## Bugs and Issues fixed in this release

* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9264'>ROOT-9264</a>] -         TDF: bogus warnings when Snapshotting a TUUID column from multiple threads
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9374'>ROOT-9374</a>] -         Re-enable dataframe_{interface,simple} and test_stringfiltercolumn tests
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9380'>ROOT-9380</a>] -         [TDF] Switch back to using Calc rather than ProcessLine when finished debugging test failures
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9416'>ROOT-9416</a>] -         [DF] Improve dataframe ownership model
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9453'>ROOT-9453</a>] -         [VecOps] Cannot instantiate `RVec<bool>`
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9456'>ROOT-9456</a>] -         [Jenkins][DF] Sporadic failures in test_snapshotNFiles
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9458'>ROOT-9458</a>] -         [DF] Allow to draw a computational graph with Dot
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9460'>ROOT-9460</a>] -         [DF] Add a Standard deviation action
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9464'>ROOT-9464</a>] -         [DF] Add GetDefinedColumns method
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9465'>ROOT-9465</a>] -         [DF] Let each node keep a local list of custom columns
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9468'>ROOT-9468</a>] -         [DF] Jitting of large Snapshots is too slow
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9471'>ROOT-9471</a>] -         [DF] Snapshot does not write any entry if many workers have zero entries to write
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9491'>ROOT-9491</a>] -         [DF] Add common base class to all node types
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9539'>ROOT-9539</a>] -         [DF] Add a tutorial for the Aggregate action
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9566'>ROOT-9566</a>] -         [DF] Add GetFilterNames
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9577'>ROOT-9577</a>] -         [DF] Move result-readiness logic to RAction
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9646'>ROOT-9646</a>] -         [DF] Improve Cache documentation
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9787'>ROOT-9787</a>] -         [DF] Write a tutorial to show the usefulness of RNode
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-5171'>ROOT-5171</a>] -         ACLiC, MacOS and debug symbol
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-5236'>ROOT-5236</a>] -         Memory Leak in RooFit/HistFactory
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-7487'>ROOT-7487</a>] -         Cannot clone histogram if gDirectory is null
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-8988'>ROOT-8988</a>] -         DNN train/test score is nan when batchsize > num events
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9081'>ROOT-9081</a>] -         TMVA BDT Multiclass BoostType error message is incorrect
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9160'>ROOT-9160</a>] -         Cannot interrupt interpreter invocation
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9185'>ROOT-9185</a>] -         rootcling dereferences null pointer
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9193'>ROOT-9193</a>] -         Broken Ruby bindings result in compile-time errors
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9209'>ROOT-9209</a>] -         Crash in cling::Value destruction
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9216'>ROOT-9216</a>] -         [Jenkins] RooFit build dependency issue?
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9258'>ROOT-9258</a>] -         GeoCad and R are incompatible due to name clashes
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9259'>ROOT-9259</a>] -         GCC 8: Fix compilation warnings
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9280'>ROOT-9280</a>] -         Test failure: roottest-root-tree-friend-make
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9319'>ROOT-9319</a>] -         [TTreeReader] main tree branch instead of friend tree branch accessed when branches have the same name
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9331'>ROOT-9331</a>] -         rootcint doesn&#39;t return with an error code when encountering a syntax error
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9363'>ROOT-9363</a>] -         Enum Autoload updates in root master broke cmssw (since Dec 2017)
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9369'>ROOT-9369</a>] -         A couple issues involving SetMustClean
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9377'>ROOT-9377</a>] -         [cling] interpreting a macro is slowed down by nullptr checks
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9396'>ROOT-9396</a>] -         Compilation error with GCC 8.1.1
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9410'>ROOT-9410</a>] -         roottest_root_tree_cloning_assertBranchCount fails sporadically
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9413'>ROOT-9413</a>] -         GCC 8 Warning Class Memory Access
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9417'>ROOT-9417</a>] -         Output files written by 6.13/02 with ZLIB can not be read by older ROOT releases
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9418'>ROOT-9418</a>] -         [TDF] Cannot use Range before a Define with a string expression
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9423'>ROOT-9423</a>] -         GCC 8 Warnings LLVM
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9424'>ROOT-9424</a>] -         GCC 8 Warnings OpenGL
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9425'>ROOT-9425</a>] -         GCC 8 Warnings Core
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9426'>ROOT-9426</a>] -         GCC 8 Warnings Net
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9427'>ROOT-9427</a>] -         GCC 8 Warnings I/O
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9428'>ROOT-9428</a>] -         GCC 8 Warnings PROOF
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9429'>ROOT-9429</a>] -         GCC 8 Warnings Math
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9430'>ROOT-9430</a>] -         GCC 8 Warnings PyROOT
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9431'>ROOT-9431</a>] -         GCC 8 Warnings TTree
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9432'>ROOT-9432</a>] -         GCC 8 Warnings THist
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9434'>ROOT-9434</a>] -         GCC 8 Warnings GUI
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9435'>ROOT-9435</a>] -         GCC 8 Warnings TMVA
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9441'>ROOT-9441</a>] -         Configure broken on Ubuntu 17
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9442'>ROOT-9442</a>] -         TTree::FlushBasket should record the cluster boundary
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9444'>ROOT-9444</a>] -         TMVA/DNN/.../CpuMatrix.H: Macro DEBUG_TMVA_TCPUMATRIX accidentally defined
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9445'>ROOT-9445</a>] -         recent commit causes failed build
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9451'>ROOT-9451</a>] -         testGenVectorVc fails
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9452'>ROOT-9452</a>] -         [DF] Exception thrown when calling jitted snapshot on an aliased column
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9454'>ROOT-9454</a>] -         [Jenkins] Frequent timeout in tp_process_imt
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9466'>ROOT-9466</a>] -         [DF] A lazy snapshot that&#39;s never triggered crashes at teardown
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9467'>ROOT-9467</a>] -         TF1: Error when reading two functions with the same formula
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9473'>ROOT-9473</a>] -         FatalRootError after updating ROOT master in CMs IBs
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9475'>ROOT-9475</a>] -         ROOT / Windows startup warnings
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9478'>ROOT-9478</a>] -         Compilation failure of version 6.14
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9479'>ROOT-9479</a>] -         Build failure: value of type "void" is not contextually convertible to "bool"
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9482'>ROOT-9482</a>] -         rootcling crashes on gcc 8.1 with debug mode enabled
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9483'>ROOT-9483</a>] -         TMVAGui::mvaeffs crashes when compiled stand-alone
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9487'>ROOT-9487</a>] -         [DF] Many warnings printed from multi-thread snapshots in some cases
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9493'>ROOT-9493</a>] -         CMake minimum version
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9495'>ROOT-9495</a>] -         BDTG&#39;s output between `imt=on` and `imt=off` differ
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9503'>ROOT-9503</a>] -         $ROOTSYS/tutorials/http/httpserver.C only shows blank page
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9504'>ROOT-9504</a>] -         RGL does not exist (official binary release for Fedora 28)
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9505'>ROOT-9505</a>] -         Template forward declaration is missing in the .rootmap
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9513'>ROOT-9513</a>] -         Data member not saved due to no streamer or dictionary
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9521'>ROOT-9521</a>] -         Issue when building ROOT with Python 3.7
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9523'>ROOT-9523</a>] -         TPolyLine does not work with "f" draw option and SetNDC
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9526'>ROOT-9526</a>] -         [DF] RResultPtrs cannot be copy- or move-assigned
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9529'>ROOT-9529</a>] -         TBits::operator~() must be const
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9532'>ROOT-9532</a>] -         Built-in OpenSSL results in non-relocatable ROOT installation
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9542'>ROOT-9542</a>] -         RDataFrame Sum gets confused by std::string
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9546'>ROOT-9546</a>] -         Mistake in RDataFrame Documentation: Aliasing &amp; Friend-TTrees
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9550'>ROOT-9550</a>] -         last ROOT versions don&#39;t support PostgreSQL less than 10.x
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9551'>ROOT-9551</a>] -         Automatic legend placement places legend over datapoints
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9552'>ROOT-9552</a>] -         Histograms filled with FillRandom do not display when using PLC, PMC options
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9555'>ROOT-9555</a>] -         [DF] Compilation fails for Reduce on a bool column due to `std::vector<bool>`
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9560'>ROOT-9560</a>] -         TGraph axis titles are lost when gPad->SetLogx(1) is called
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9563'>ROOT-9563</a>] -         [TreeProcMT] Trees in subdirectories are not supported (and their usage lead to a crash)
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9571'>ROOT-9571</a>] -         segfault using RooKeysPdf
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9574'>ROOT-9574</a>] -         TTree::TClusterIterator::GetEstimatedClusterSize broken for ROOT File with clustering disabled.
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9582'>ROOT-9582</a>] -         python crashes at end of script when adding objects to gDirectory
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9588'>ROOT-9588</a>] -         Type declaration shadows local var
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9589'>ROOT-9589</a>] -         Cast between incompatible types in TFile
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9591'>ROOT-9591</a>] -         [GCC8] Cast between incompatible types in TROOT
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9592'>ROOT-9592</a>] -         [GCC8] strncpy length depends on source arg in TClass
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9593'>ROOT-9593</a>] -         [GCC8] Truncation and length issues in rpdutils
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9594'>ROOT-9594</a>] -         [GCC8] Possible byte overlap in rootd
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9595'>ROOT-9595</a>] -         Class memaccess violation in XrdProofd
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9596'>ROOT-9596</a>] -         [GCC8] Byte overlap in TSystem
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9597'>ROOT-9597</a>] -         [GCC8] Truncation issue in TTabCom
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9598'>ROOT-9598</a>] -         Union declaration shadows previous local in BitReproducible
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9600'>ROOT-9600</a>] -         [GCC8] Unsafe print in TSecContent
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9601'>ROOT-9601</a>] -         [GCC8] Function input arguments misinterpretation in TAuthenticate
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9602'>ROOT-9602</a>] -         [GCC8] String truncation issues in TAuthentication
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9603'>ROOT-9603</a>] -         [GCC8] Truncation issue in THostAuth
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9604'>ROOT-9604</a>] -         Class memaccess violation in TTreeCacheUnzip
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9605'>ROOT-9605</a>] -         [GCC8] snprintf directive truncation in TTree
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9608'>ROOT-9608</a>] -         Ignored qualifiers in TUnfoldBinning
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9609'>ROOT-9609</a>] -         [GCC8] Class memaccess violation in TGGC
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9610'>ROOT-9610</a>] -         [GCC8] Directive truncation in TTreeFormula
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9611'>ROOT-9611</a>] -         Catching polymorphic by value in DataSet, Reader, VariableGaussTransform
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9613'>ROOT-9613</a>] -         [GCC8] Unrecognised command line option
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9615'>ROOT-9615</a>] -         Unknown argument "-fno-plt"
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9624'>ROOT-9624</a>] -         TMySQLStatement.h incompatible with MySQL 8
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9628'>ROOT-9628</a>] -         RDataFrame segfaults when processing TChain with zombie entries
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9630'>ROOT-9630</a>] -         Arrow RDS and lack of int32 support
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9632'>ROOT-9632</a>] -         ROOT::RDataFrame::Define struggles with user-defined return types when built stand-alone
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9635'>ROOT-9635</a>] -         Configuration and Build fails when CXXFLAGS has &#39;--save-temps&#39;
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9637'>ROOT-9637</a>] -         "root-config --cflags" echoes bad -std=c++1y
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9640'>ROOT-9640</a>] -         Segmentation violation after starting TBrowser b
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9660'>ROOT-9660</a>] -         cling forward declaration issue with enums
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9662'>ROOT-9662</a>] -         Thread problem in I/O for ROOT master
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9664'>ROOT-9664</a>] -         "pmarker->Draw("nodraw #1" with SaveAs method
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9665'>ROOT-9665</a>] -         RooDataSet Import
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9666'>ROOT-9666</a>] -         [DF] Reading/writing `std::vector<bool>` and C-arrays of bool is broken
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9667'>ROOT-9667</a>] -         BDT difference IMT=on and off
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9681'>ROOT-9681</a>] -         Typedef clash in global namespace between ROOT and Geant4
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9684'>ROOT-9684</a>] -         Wrong cast of functions in TGlobalMappedFunction
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9693'>ROOT-9693</a>] -         <TFormula::Streamer> errors from TF1s from an older root 6 version
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9694'>ROOT-9694</a>] -         Infinite recursion in GetStreamerInfoList when opening corrupt file.
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9701'>ROOT-9701</a>] -         IO read rules no longer executed
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9729'>ROOT-9729</a>] -         libPyROOT.so linked against libRDataFrame.so and others in master
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9738'>ROOT-9738</a>] -         Crashing ROOT in 5 characters
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9751'>ROOT-9751</a>] -         Cannot link code against RDataFrame if builtin Vdt is enabled
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9754'>ROOT-9754</a>] -         [DF] Cannot define custom columns with same name in two different branches of the computation graph if both of them are jitted
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9762'>ROOT-9762</a>] -         Custom read rule fails to execute in certain situations
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9769'>ROOT-9769</a>] -         Increased memory when creating an empty TFile in a loop
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9773'>ROOT-9773</a>] -         TChain::GetEntry reads multiple clusters without prefetching
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9784'>ROOT-9784</a>] -         ROOT 6 does not read ROOT 5  RooFit workspaces because of cint <-> cling difference
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9794'>ROOT-9794</a>] -         pymva misbehaves on fail-on-missing
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9801'>ROOT-9801</a>] -         Copying a TFormula doesn&#39;t preserve fLazyInitialization
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9808'>ROOT-9808</a>] -         GCC8: no dict for __pair_base
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9826'>ROOT-9826</a>] -         AsymptoticCalculator::MakeAsimovData doesn&#39;t work with nested likelihood functions
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9827'>ROOT-9827</a>] -         RootNewMacros.cmake doesn&#39;t work for dependent projects due to depedency on ROOT&#39;s own options
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9837'>ROOT-9837</a>] -         GeoCad and anything are incompatible due to name clashes
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9840'>ROOT-9840</a>] -         Problem using static method in TF1 and TFormula
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9841'>ROOT-9841</a>] -         Build Error: ‘string_view’ in namespace ‘std’ does not name a type
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9860'>ROOT-9860</a>] -         [DF] Snapshot can write wrong values in some cases
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9865'>ROOT-9865</a>] -         [DF] Snapshot of datasets containing C arrays
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9872'>ROOT-9872</a>] -         ROOT-9660 broke root 6.14 and master
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9879'>ROOT-9879</a>] -         GL viewer does not render the volumes
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9890'>ROOT-9890</a>] -         Memory error when using Roofit objects as attributes of a class in python
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-8122'>ROOT-8122</a>] -         Terminate interpreter in clean way if EOT / Ctrl+D is read and current line is empty
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9702'>ROOT-9702</a>] -         Cannot seamlessly evolve a datamember of type auto_ptr<T> into unique_ptr<T>
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9710'>ROOT-9710</a>] -         CMake warning for OpenGL setup (CMake policy CMP0072)
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9066'>ROOT-9066</a>] -         Suppression of TBB warnings (c9d74075911) leads to missing dependency
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9384'>ROOT-9384</a>] -         let users get root version from terminal with "root --version";
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9437'>ROOT-9437</a>] -         Improve the processing of a TChain with many files in TTreeProcessorMT
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9668'>ROOT-9668</a>] -         IMT code segfaults when branches are changed
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9739'>ROOT-9739</a>] -         Propagate configuration environment to the externals

## HEAD of the v6-16-00-patches branch

These changes will be part of the future 6.16/02.

## Core Libraries

* Speed-up startup, in particular in case of no or poor network accesibility, by avoiding
a network access that was used as input to generate a globally unique ID for the current
process.
 * This network access is replaced by a passive scan of the network interface. This
reduces somewhat the uniqueness of the unique ID as the IP address is no longer
guaranteed by the DNS server to be unique.   Note that this was already the case when
the network access (used to look up the hostname and its IP address) failed.

## I/O Libraries

* Add renaming rule for instances of the math classes from `genvector` and `smatrix` to
instance for one floating point type (`float`, `double`, `Double32_t`, `Float16_t`) to
instances for any other floating point type.
* Corrected the application of  `I/O customization rules` when the target classes contained
typedefs (in particular `Double32_t`)
* Set offset of the used-for-write element in case of `I/O` rule on 'current' `StreamerInfo.`
* Fix `TTreeReader`'s use of `Set[Local]Entry`
* Avoid deleted memory access in `MakeProject` and in handling of
`I/O customization rules`.
* Significantly improved the scaling of hadd tear-down/cleanup-phase in the presence
of large number histograms and in the presence of large number of directories.


## TTree Libraries

* Reset the `TCutG` in `TTreeFormula::ResetLoading`.
* Properly handle `TTree` aliases contain just function calls.
* Prevent a situation in `TTreeFormula` when stale cached information was re-used.

## Histogram Libraries

* Update all `TH1` derived class version number.
 * The files produced by `v6.14/00, v6.14/02, v6.14/04, v6.14/06, v6.14/08`
and `v6.16/00` when read by older version of `ROOT` (eg. `v5.34`) will put the `Core/Meta`
in a state that prevent writing of any histogram (from within that process).
 * The files produced by this version and later no longer suffer from this deficiency.
*  Allow reading v5 TF1 that were stored memberwise in a TClonesArray.
