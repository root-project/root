% ROOT Version 6.38 Release Notes
% 2025-11
<a name="TopOfPage"></a>

## Introduction

For more information, see:

[http://root.cern](http://root.cern)

The following people have contributed to this new version:

 Bertrand Bellenot, CERN/EP-SFT,\
 Jakob Blomer, CERN/EP-SFT,\
 Lukas Breitwieser, CERN/EP-SFT,\
 Carsten Burgard, University of Hamburg and TU Dortmund,\
 Philippe Canal, FNAL,\
 Simon Cello, TU Dortmund,\
 Olivier Couet, CERN/EP-SFT,\
 Marta Czurylo, CERN/EP-SFT,\
 Florine de Geus, CERN/EP-SFT and University of Twente,\
 Edward Finkelstein, UCI and SDSU,\
 Jonas Hahnfeld, CERN/EP-SFT and Goethe University Frankfurt,\
 Fernando Hueso Gonzalez, IFIC (CSIC-University of Valencia),\
 Stephan Hageboeck, CERN/EP-SFT,\
 Petr Jacka, Czech Technical University in Prague,\
 Aaron Jomy, CERN/EP-SFT,\
 Sergey Linev, GSI Darmstadt,\
 Lorenzo Moneta, CERN/EP-SFT,\
 Vincenzo Eduardo Padulano, CERN/EP-SFT,\
 Giacomo Parolini, CERN/EP-SFT,\
 Danilo Piparo, CERN/EP-SFT,\
 Jonas Rembser, CERN/EP-SFT,\
 Sanjiban Sengupta, CERN/EP-SFT and Manchester University,\
 Silia Taider, CERN/EP-SFT,\
 Florian Uhlig, GSI,\
 Devajith Valaparambil Sreeramaswamy, CERN/EP-SFT,\
 Vassil Vassilev, Princeton,\
 Sandro Wenzel, CERN/ALICE,\

## Deprecation and Removal

* The `RooDataSet` constructors to construct a dataset from a part of an existing dataset were deprecated in ROOT 6.36 and are now removed. This is to avoid interface duplication. Please use `RooAbsData::reduce()` instead, or if you need to change the weight column, use the universal constructor with the `Import()`, `Cut()`, and `WeightVar()` arguments.
* The `RooStats::HLFactory` class that was deprecated in ROOT 6.36 is now removed. It provided little advantage over using the RooWorkspace directly or any of the other higher-level frameworks that exist in the RooFit ecosystem.
* The build options `mysql`, `odbc` and `pgsql`, that were deprecated in ROOT 6.36, are now removed.
* The `TGLIncludes.h` and `TGLWSIncludes.h` headers are deprecated and will be removed in ROOT 6.40. Please include your required headers like `<GL/gl.h>` or `<GL/glu.h>` directly.
* The GLEW headers (`GL/eglew.h`, `GL/glew.h`, `GL/glxew.h`, and `GL/wglew.h`) that are installed when building ROOT with `builtin_glew=ON` are deprecated and will be removed in ROOT 6.40. This is done because ROOT will move away from GLEW for loading OpenGL extensions.
* The `ROOT::Math::TDataPointN` class that can be used with the `ROOT::Math::KDETree` was removed. Use the templated `TDataPoint<N>` instead.
* The Parallel ROOT Facility, `PROOF`, has been removed from the repository.
* After being deprecated for a long period, the `-r` option of `rootcling` has been removed.
* The `rpath` build option is deprecated. It is now without effect.
  Relative RPATHs to the main ROOT libraries are unconditionally appended to all ROOT executables and libraries if the operating system supports it.
  If you want a ROOT build without RPATHs, use the canonical CMake variable `CMAKE_SKIP_INSTALL_RPATH=TRUE`.
* The `TH1K` class is deprecated and will be removed in 6.40. It did not implement the `TH1` interface consistently, and limited the usability of the k-neighbors method it implemented by closely coupling the algorithm with the histogram class. Please use the new `TMath::KNNDensity` function that implements the same mathematical logic.
* Comparing C++ `nullptr` objects with `None` in Python now results in a `FutureWarning`, and will result in a `TypeError` starting from ROOT 6.40. This is to prevent confusing results where `x == None` and `x is None` are not identical. Use truth-value checks like `if not x` or `x is None` instead.

## Build System
* Improve building ROOT when ROOT is already installed in the system. ROOT now correctly handles system-include folders that both contain a package that ROOT depends on and a ROOT installation. Dependent packages are included with `-isystem` instead of `-I`, so installed ROOT headers will not interfere with a ROOT build. See [#8708](https://github.com/root-project/root/issues/8708) for details.
* Add support of `-isystem` to rootcling.

## Core Libraries
* ROOT and the Cling C++ interpreter now relies on LLVM version 20.
* Experimental SYCL support in the ROOT prompt. This feature can be enabled by building ROOT with `-Dexperimental_adaptivecpp=ON`.
* Behavior change: when selecting a template instantiation for a dictionary, all the template arguments have to be fully defined - the forward declarations are not enough any more. The error prompted by the dictionary generator will be `Warning: Unused class rule: MyTemplate<MyFwdDeclaredClass>`.
* New expert option to reduce static startup cost of ROOT by setting environment variables
```bash
export ROOT_LDSYSPATH=$(LD_DEBUG=libs LD_PRELOAD=DOESNOTEXIST ls /tmp/DOESNOTEXIST 2>&1 | grep -m 1 "system search path" | sed 's/.*=//g' | awk '//{print $1}')
export CLING_LDSYSPATH=ROOT_LDSYSPATH
export CLING_CPPSYSINCL=$(LC_ALL=C c++ -xc++ -E -v /dev/null 2>&1 | sed -n '/^.include/,${/^ \/.*++/{p}}' | tr '\n' ':' | tr ' ' ':')
```
This caching reduces sub-process creation during initialization and can be useful when multiple ROOT instances or binaries linked to ROOT are executed (less system-calls, cleaner debugging).
* It is now possible to read a user configuration file (in jeargon, a "rootrc file") at startup in a custom path instead of the one in the home directory, by specifying its full path with the `ROOTENV_USER_PATH` environment variable.
* ROOT reacts to the environment variable `ROOT_MAX_THREADS`. This can be used to select the number of worker threads when implicit multithreading is enabled. It is supported since 2021, but better documentation was added in the context of the [RDataFrame documentation](https://root.cern.ch/doc/v638/classROOT_1_1RDataFrame.html#parallel-execution).
* ROOT now determines `std::hardware_destructive_interference_size` at configure time, and defines the macro `R__HARDWARE_INTERFERENCE_SIZE` in `RConfigure.h`. Not keeping it fixed could lead to ABI breakages when code is interpreted on a machine that is different from the machine where ROOT was compiled.

## I/O

### TTree

* Behaviour change: the behaviour of `TChain::SetBranchStatus` has been aligned to the one of `TTree::SetBranchStatus`. In particular, when `SetBranchStatus` is called to deactivate all branches, a subsequent call to `TChain::SetBranchAddress` would override the previous instruction and activate that single branch. Instead `TTree::SetBranchAddress` respects the rule imposed by `SetBranchStatus`. If a user needs to activate only one or more branches, they should call `SetBranchStatus("brName", true)` on each branch that needs to be active in the TChain, like it was already necessary for a TTree. See https://github.com/root-project/root/pull/19221 for more details.

### RNTuple

* The parallel writer is now part of the public, stable API. The `RNTupleParallelWriter` and the closely related `RNTupleFillContext` moved from the `ROOT::Experimental` to the `ROOT` namespace.
* Automatic schema evolution (in addition to I/O customization rules, which are already supported) akin to the classic I/O behavior. Expert documentation added in `tree/ntuple/doc/SchemaEvolution.md`.
* RNTuple support in the classic browser together with several visual improvements, including a new treemap visualization of field sizes.
* Bulk I/O optimization for fixed-size arrays and vectors of fixed-size arrays.
* The interface to access field data through the `RNTupleProcessor` has been refactored, and now includes a mechanism to safely handle reading from entries where not every field may hold a valid value. While the `RNTupleProcessor` is still experimental, and more interface changes may take place before it becomes part of the public, stable API, users are encouraged to try out this new feature. Feedback is highly appreciated!

## Math
* Added GenVectorX, an extended version of the GenVector library supporting multi-target execution with SYCL. Enable by configuring CMake with:
```bash
-Dexperimental_adaptivecpp=ON -Dexperimental_genvectorx=ON
```
* When a Chi2 test is used with TProfiles, the "WW" option is used by default, because a weighted-to-weighted comparison is required.
* When histograms compute automatic axis ranges (e.g. in TTree::Draw), the range is now slightly extended beyond the min/max of the underlying distribution. Otherwise, the maximum of the distribution will fall in the overflow bin, which is often not desired.
* Acoplanarity and asymmetry functions have been added to [the GenVector library (e.g. LorentzVector)](https://root.cern.ch/doc/v638/group__GenVector.html).
* Several classes in the genvector package (e.g. ROOT::Math::LorentzVector) are now nothrow move constructible. The ABI didn't change, but members are default initialised and outdated copy constructors have been removed, enabling compiler-generated copy and move constructors.

### Minimizer interface

* The function `double ROOT::Math::Minimizer::GlobalCC(unsigned int ivar)` was changed to `std::vector<double> ROOT::Math::Minimizer::GlobalCC()`, always returning the full list of global correlations.
  This change was needed because global correlations are not unconditionally computed and cached in the minimizer anymore. Only computing them when calling `GlobalCC()` avoids unneeded overhead when global correlations are not required.

### Minuit2

* Behavior change: building ROOT using `minuit2_omp=ON` option no longer enables OpenMP parallelization by default. One has to call now additionally GradientCalculator::SetParallelOMP().
* The functionality of the `FCNGradAdapter` got absorbed into the `FCNAdapter`. This also means that the `FCNGradAdapter.h` header is gone.
* The `ROOT::Minuit2::FCNAdapter` is no longer templated and now handle only functions with this exact signatures:
  * `double(double const *)` for the wrapped function
  * `void(double const *, double *)` for the gradient
  The advantage of this restriction is that the `FCNAdapter` can now also wrap any Python function that respects this interface.

## RooFit

### Error out when setting out-of-range variable value instead of silent clipping

In previous versions, if you set the value of a variable with `RooRealVar::setVal()`, the value was silently clippend when it was outside the variable range.
This silent mutation of data can be dangerous.
With ROOT 6.38, an exception will be thrown instead.
If you know what you are doing and want to restore the old clipping behavior, you can do so with `RooRealVar::enableSilentClipping()`, but this is not recommended.

### Changed return type of `RooAbsData::split()`

The return type of `RooAbsData::split()` was changed. So far, it returned a `TList*`, which is changed to `std::vector<std::unique_ptr<RooAbsData>>` in this release. The reason for this breaking change was memory safety. The returned `TList` *as well* as the `RooAbsData` objects it contains had to be deleted by the caller, which is usually forgotten in user frameworks and even RooFit itself. The new return type enforces memory safety.

Furthermore, the `RooAbsData::split()` method is not virtual anymore, as it's not meant the be overridden by inheriting classes.

The names of the datasets in the return object still correspond to the channel names from the category that was used to split the dataset.
It is quite common to look up the data for a given channel from the data splits, which could previously done with `TList::FindObject()`.
```C++
TList *splits{data->split(*category)};
std::string channelName = "channel_a";
RooAbsData* channelData = static_cast<RooAbsData*>(splits->FindObject(channelName.c_str()));
// ... do something with channelData ...
splits->Delete();
delete splits;
```

With the new return type, one has to use algorithms from the standard library to do the lookups:
```C++
std::vector<std::unique_ptr<RooAbsData>> splits{data->split(*category)};
std::string channelName = "channel_a";
auto found = std::find_if(splits.begin(), splits.end(), [&](auto const &item) {
  return nameIdx.first == item->GetName();
});
RooAbsData *dataForChan = found != splits.end() ? found->get() : nullptr;
// ... do something with channelData ...
```

If you want to keep using `TList*` return values, you can write a small adapter function as described in the documentation of `RooAbsData::split()`.

### RooCrystalBall alternative

- A [simpler alternative](http://arxiv.org/abs/1603.08591v1) to RooCrystalBall using a Gaussian with exponential tails has been implemented: `class RooGaussExpTails`.


## RDataFrame
- Memory savings in RDataFrame: When many Histo3D are filled in RDataFrame, the memory consumption in multi-threaded runs can be prohibitively large, because
  RDF uses one copy of each histogram per thread. Now, RDataFrame can reduce the number of clones using `ROOT::RDF::Experimental::ThreadsPerTH3()`. Setting this
  to numbers such as 8 would share one 3-d histogram among 8 threads, significantly reducing the memory consumption. This might slow down execution if the histograms
  are filled at very high rates, in which case lower numbers are better.
- HistoNSparseD action which fills a sparse N-dimensional histogram is now added.
- RDatasetSpec class now also supports RNTuple, including the usage of the factory function FromSpec.

### Snapshot
- The Snapshot method has been refactored so that it does not need anymore compile-time information (i.e. either template arguments or JIT-ting) to know the input column types. This means that any Snapshot call that specifies the template arguments, e.g. `Snapshot<int, float>(..., {"intCol", "floatCol"})` is now redundant and the template arguments can safely be removed from the call. At the same time, Snapshot does not need to JIT compile the column types, practically giving huge speedups depending on the number of columns that need to be written to disk. In certain cases (e.g. when writing O(10000) columns) the speedup can be larger than an order of magnitude. The Snapshot template is now deprecated and it will issue a compile-time warning when called. The function overload is scheduled for removal in ROOT 6.40.
- Experimental support for systematic variations has been added to snapshots. Refer to the section on [systematic variations](https://root.cern.ch/doc/v638/classROOT_1_1RDataFrame.html#systematics) in the RDataFrame manual. The support currently only encompasses single-threaded snapshots to TTree, but support will be extended to multi-threaded snapshots in TTree and to RNTuple in future ROOT versions.
- The default compression setting for the output dataset used by Snapshot has been changed from 101 (ZLIB level 1, the TTree default) to 505 (ZSTD level 5). This is a better setting on average, and makes more sense for RDataFrame since now the Snapshot operation supports more than just the TTree output data format. This change may result in smaller output file sizes for your analyses that use Snapshot with default settings. During the 6.38 development release cycle, Snapshot will print information about this change once per program run. Starting from 6.40.00, the information will not be printed. The message can be suppressed by setting ROOT_RDF_SILENCE_SNAPSHOT_INFO=1 in your environment or by setting 'ROOT.RDF.Snapshot.Info: 0' in your .rootrc.
- Multithreaded snapshotting to RNTuple has been added. This will be used automatically when `ROOT::SetImplicitMT()` has been called and `RSnapshotOptions::fOutputFormat` has been set to `ESnapshotOutputFormat::kRNTuple`.

### Distributed RDataFrame
- Processing of RNTuples is now done in the same way as for TTrees, as it is a more efficient way of distributing the computations. The API remains unchanged.

## Python Interface

ROOT dropped support for Python 3.8, meaning ROOT now requires at least Python 3.9.

### Deprecation of the `TObject` equality pythonization

`TObject.__eq__` is deprecated and will be removed in ROOT 6.40.

It forwards to `TObject::Equals()`, which uses pointer comparison if not overridden in derived classes.
This may be confusing, because people expect value comparisons.
Use Pythons `is` for pointer comparison, or request an implementation of `operator==` on the C++ side if you need value-based equality checks for a given class.

### Deprecate the attribute pythonization of `TDirectory` in favor of item-getting syntax

Since ROOT 6.32, the recommended way to get objects from a `TFile` or any `TDirectory` in general is via `__getitem__`:

```python
tree = my_file["my_tree"] # instead of my_file.my_tree, which gave you a deprecation warning since ROOT 6.32
```

The deprecated pythonization with the `__getattr__` syntax is now removed.
It was originally schedeuled for removal in 6.34 according to the 6.32 release notes, but since it was still used quite a bit,
the deprecation period was extended.

### Enhancements to the RDataFrame Pythonic API

#### Support for C++ free functions in `.Define` and `.Filter`
- It is now possible to pass user-defined C++ free functions (including templated and overloaded ones) directly to `RDataFrame.Define()` and `Filter()`.
- Functions taking templated input types such as STL containers (e.g. std::vector<T>) are not yet supported.

#### Extended Numba support for C++ containers and ROOT classes
- The `@ROOT.Numba.Declare` decorator was extended to support functions using `std::vector`, `std::array`, and a selected set of ROOT classes (limited by [cppyy’s Numba extension](https://cppyy.readthedocs.io/en/latest/numba.html)).

#### Further Unified Histogram Interface (UHI) integration and histogram pythonizations
- Implemented `__iter__` for histograms to return an immutable copy of bin contents, including flow bins.
- Integrated the [UHI testing suite](https://uhi.readthedocs.io/en/latest/testing.html) providing cross-library validation tests to ensure consistency and interoperability with the UHI specification.

## ROOT executable

- Removed stray linebreak when running `root -q` with no input files.
  This ensures that there is no superfluous output when running `root` without the banner and without input files (`root -q -l`).

## Command-line utilities
- The `rootls` utility has a new native implementation and can now be run without Python.
  The options and the output of the new rootls are identical to the previous implementation but it should run faster (typically about 3 to 4x faster).

- The `rootbrowse` utility has a new native implementation and can now be run without Python.

## JavaScript ROOT
- A new configuration option `Jupyter.JSRoot` was added in .rootrc to set the default mode for JSROOT in Jupyter notebooks (on or off).
- JSROOT 7.10.0:
   * `RNtuple` support, thanks to Kriti Mahajan https://github.com/Krmjn09
   * Implement `RTreeMapPainter` to display `RNTuple` structure, thanks to Patryk Pilichowski https://github.com/magnustymoteus
   * Implement `build3d` function for building three.js objects for `TH1/2/3`, `TLatex` `TGeo`, `TGraph2D` classes
   * Draw `TAnnotation3D` in real 3D with handling scene rotation
   * Let use hex colors in histogram draw options like "fill_00ff00" or "line_77aa1166"
   * Let configure exact axis ticks position via draw option like "xticks:[-3,-1,1,3]"
   * Support gStyle.fBarOffset for `TGraph` bar drawing
   * Support "fill_<id>" and "line_<id>" draw options for `TGraph`
   * Support dark mode when store images
   * With 'Shift' key pressed whole graph is moved by dragging action
   * Support `Xall` and `Yall` as projections width
   * Implement `unzipJSON()` function for data embedding in jupyter
   * Support reading `TBranch` from very old ROOT files with custom streamers
   * Upgrade three.js r174 -> r180
   * Upgrade lil-gui.mjs 0.19.2 -> 0.20.0
   * Upgrade svg2pdf.js 2.3.0 -> 2.6.0
   * Upgrade jsPDF 2.5.2 -> 3.0.3, exclude gif, bmp, jpeg support
   * Use ES6 modules to implement geoworker, enable node.js usage
   * Remove countGeometryFaces function - use numGeometryFaces instead
   * Remove experimental RHist classes, deprecated in ROOT 6.38
   * Internal - ws members are private, new methods has to be used
   * Fix - ticks size and labels with kMoreLogLabels axis bit
   * Fix - first color in palette drawing
   * Fix - latex parsing error of `#delta_{0}_suffix` string
   * Fix - reduce plain HTML usage to minimize danger of JS code injection


## Experimental features

### RFile

* A new experimental interface for ROOT files, `ROOT::Experimental::RFile`, has been added. This aims to be a
  minimal and modern interface to ROOT files with explicit ownership of objects.

## Optimization of ROOT header files

More unused includes were removed from ROOT header files.
For instance, `#include "TMathBase.h"` was removed from `TString.h`.
This change may cause errors during compilation of ROOT-based code. To fix it, provide missing the includes
where they are really required.
This improves compile times and reduces code inter-dependency; see https://github.com/include-what-you-use/include-what-you-use/blob/master/docs/WhyIWYU.md for a good overview of the motivation.

## Versions of built-in packages

* The version of openssl has been updated to 3.5.0
* CFITSIO has been updated to 4.6.3

## Items addressed for this release

More than 240 items were addressed for this release:

  * [[#20266](https://github.com/root-project/root/issues/20266)] - Cannot access methods of branches with TChain::Scan and friends
  * [[#20252](https://github.com/root-project/root/issues/20252)] - TGraph::SetHistogram() can leak memory
  * [[#20251](https://github.com/root-project/root/issues/20251)] - TInterpreter crash
  * [[#20249](https://github.com/root-project/root/issues/20249)] - TTree cannot find branch of friend TChain stored in inner TTree
  * [[#20248](https://github.com/root-project/root/issues/20248)] - TChain cannot find branch of its own TTree if names differ
  * [[#20228](https://github.com/root-project/root/issues/20228)] - Integer overflow in TTree::Draw
  * [[#20226](https://github.com/root-project/root/issues/20226)] - Integer overflow in TTree::Scan
  * [[#20213](https://github.com/root-project/root/issues/20213)] - Implement RNTuple redirection from FUSE to XRootD
  * [[#20189](https://github.com/root-project/root/issues/20189)] - Invalid memory access in RooArgSet::contentsString() with an empty RooArgSet
  * [[#20185](https://github.com/root-project/root/issues/20185)] - TAxis constructor from bin edges silently accepts the same edges in the list
  * [[#20164](https://github.com/root-project/root/issues/20164)] - Strange  failure  for RDataFrame.Snapshot for the  *ancient*  input ROOT file
  * [[#20147](https://github.com/root-project/root/issues/20147)] - compilation failure on emulated ARM for 6.36.04
  * [[#20145](https://github.com/root-project/root/issues/20145)] - `TestSofieModels` is failing with PyTorch >= 2.9.0
  * [[#20132](https://github.com/root-project/root/issues/20132)] - Look for .rootrc in a custom dir
  * [[#20083](https://github.com/root-project/root/issues/20083)] - [RF] Offset() option is ignored by the fitTo() function
  * [[#20081](https://github.com/root-project/root/issues/20081)] - Also consider `Bool_t` in numpy array conversion
  * [[#20077](https://github.com/root-project/root/issues/20077)] - Histogram projections and UHI fail on histograms with axes having infinite upper edges.
  * [[#20063](https://github.com/root-project/root/issues/20063)] - ROOT fails to build on alma9 aarch64 with gcc13/14/15
  * [[#20033](https://github.com/root-project/root/issues/20033)] - Erroneous values and crash in TTreePlayer when mixing many TChains and friendship
  * [[#20015](https://github.com/root-project/root/issues/20015)] - libcppyy shows up in top-level site-packages
  * [[#20014](https://github.com/root-project/root/issues/20014)] - `TH2D operator+` not working in Python
  * [[#19986](https://github.com/root-project/root/issues/19986)] - [PyROOT] Proper way to check if snapshot exists in workspace
  * [[#19965](https://github.com/root-project/root/issues/19965)] - thisroot.sh fails on macos when called from script with #!/bin/sh
  * [[#19963](https://github.com/root-project/root/issues/19963)] - Reading individual TTree sub-branch into a std::map lead to write into delete memory.
  * [[#19942](https://github.com/root-project/root/issues/19942)] - allow setting name of TStatistic or RDataFrame.Stats
  * [[#19889](https://github.com/root-project/root/issues/19889)] - Mixing Clang 20 and ROOT causes very sporadic crashes
  * [[#19888](https://github.com/root-project/root/issues/19888)] - [roottest] remove deprecated roottest/boost headers
  * [[#19879](https://github.com/root-project/root/issues/19879)] - add user-defined handling of NaN and inf to RDataFrame statistics
  * [[#19867](https://github.com/root-project/root/issues/19867)] - ROOT dictionary crash related to default template arguments
  * [[#19850](https://github.com/root-project/root/issues/19850)] - Error: "no member named 'getenv' in the global namespace" during the compilation with libc++
  * [[#19834](https://github.com/root-project/root/issues/19834)] - HasColumn crashing for an empty RDataFrame
  * [[#19820](https://github.com/root-project/root/issues/19820)] - Name collisions in Workspace when using RooAbsPdf.derivative()
  * [[#19814](https://github.com/root-project/root/issues/19814)] - Regression in 6.34: segfault when using TTreeReader on partially initialized TChain
  * [[#19777](https://github.com/root-project/root/issues/19777)] - Complete roottest/root/meta/callfunc runmemberFunc.C
  * [[#19776](https://github.com/root-project/root/issues/19776)] - “HIST” option for TRatioPlot(TH1*, THStack)
  * [[#19770](https://github.com/root-project/root/issues/19770)] - `RooWorkspace` return values unintuitive?
  * [[#19768](https://github.com/root-project/root/issues/19768)] - JSON output and infinity.
  * [[#19706](https://github.com/root-project/root/issues/19706)] - Crash when exiting Python interpreter after using `ROOT.RDF.FromNumpy`
  * [[#19687](https://github.com/root-project/root/issues/19687)] - Enable test roottest/root/treeformula/array/nonsplit.C
  * [[#19650](https://github.com/root-project/root/issues/19650)] - [tree] missing staged data in I/O rules of split vector<T> --> vector<U>
  * [[#19592](https://github.com/root-project/root/issues/19592)] - No Constant Properties exported for Optical surfaces
  * [[#19568](https://github.com/root-project/root/issues/19568)] - Ineffective `TMatrixTSym` symmetry check during construction
  * [[#19560](https://github.com/root-project/root/issues/19560)] - TTreeIndex fails under Valgrind due to long double emulation
  * [[#19555](https://github.com/root-project/root/issues/19555)] - TRandomMixMax wrong link in documentation
  * [[#19533](https://github.com/root-project/root/issues/19533)] - [ntuple] Unknown column types should throw
  * [[#19497](https://github.com/root-project/root/issues/19497)] - EL9:: EPEL:: root tags/6-36-02@6-36-02 :: cppyy warning in rootls
  * [[#19485](https://github.com/root-project/root/issues/19485)] - `hadd` reports inefficient parallel merging, no setting possible
  * [[#19479](https://github.com/root-project/root/issues/19479)] - Info() logs should not go to `stderr`
  * [[#19476](https://github.com/root-project/root/issues/19476)] - `std::string_view` adds characters to strings in Python
  * [[#19452](https://github.com/root-project/root/issues/19452)] - error: no member named 'hessian' in namespace 'clad'
  * [[#19442](https://github.com/root-project/root/issues/19442)] - [ntuple] Unable to read back RNTuples without normalized type names from v6.34
  * [[#19438](https://github.com/root-project/root/issues/19438)] - TH2D COLZ0 option color bug
  * [[#19422](https://github.com/root-project/root/issues/19422)] - Make `RooFit::MsgTopic` enum name for numeric integration consistent
  * [[#19419](https://github.com/root-project/root/issues/19419)] - [RF] FrequentistCalculator cannot handle non-parametric Pdfs
  * [[#19412](https://github.com/root-project/root/issues/19412)] - TASImage Issue with latest libafterimage
  * [[#19410](https://github.com/root-project/root/issues/19410)] - [ntuple] Heap corruption on Windows in ntuple_show
  * [[#19390](https://github.com/root-project/root/issues/19390)] - c.Print when Title contains tex (old ROOT-7087 bug)
  * [[#19366](https://github.com/root-project/root/issues/19366)] - `THnSparse::CreateSparse` creates histograms that are not sparse
  * [[#19362](https://github.com/root-project/root/issues/19362)] - [RDF] Automatic axis extension not working with weighted filling
  * [[#19359](https://github.com/root-project/root/issues/19359)] - Incorrect bin error with `TH1::SetBuffer` and `TH1::Sumw2`
  * [[#19349](https://github.com/root-project/root/issues/19349)] - typo: documentation for RooCrystalBall mathematical implementation does not match code implementation
  * [[#19346](https://github.com/root-project/root/issues/19346)] - Duplicated tests in tree/treeplayer/test lead to random failures
  * [[#19338](https://github.com/root-project/root/issues/19338)] - Feature Request ( Functions for Acoplanarity, Vector and Scalar Pt asymmetries)
  * [[#19333](https://github.com/root-project/root/issues/19333)] - Ninja build cannot start if `builtin_xrootd=ON`
  * [[#19330](https://github.com/root-project/root/issues/19330)] - A partial merge in TFileMerger creates directories that users didn't ask for
  * [[#19325](https://github.com/root-project/root/issues/19325)] - ROOT_CHECK_CONNECTION is too aggressive for sites with limited network access
  * [[#19320](https://github.com/root-project/root/issues/19320)] - Jupyter Lab does not start when a read-only ROOT installation is present (loaded via thisroot.sh)
  * [[#19313](https://github.com/root-project/root/issues/19313)] - Problem linking against VDT in the latest versions
  * [[#19311](https://github.com/root-project/root/issues/19311)] - Histogram python `__len__` inconsistent with `__getitem__`
  * [[#19269](https://github.com/root-project/root/issues/19269)] - Use of -r argument crashes rootcling
  * [[#19267](https://github.com/root-project/root/issues/19267)] - [cppyy] New crash when iterating over polymorphic std::vector
  * [[#19256](https://github.com/root-project/root/issues/19256)] - [ntuple] Schema evolution of repetitive fields
  * [[#19254](https://github.com/root-project/root/issues/19254)] - [ntuple] Schema evolution of record fields
  * [[#19253](https://github.com/root-project/root/issues/19253)] - [ntuple] Schema evolution of `std::variant`
  * [[#19241](https://github.com/root-project/root/issues/19241)] - THnBase::ProjectionAny does not calculate errors correctly
  * [[#19224](https://github.com/root-project/root/issues/19224)] - [Python] Unit tests fail with Python debug builds
  * [[#19220](https://github.com/root-project/root/issues/19220)] - Inconsistent behaviour of SetBranchStatus between TTree and TChain
  * [[#19168](https://github.com/root-project/root/issues/19168)] - Documentation of RNTupleWriter / RNTuple could be improved a bit
  * [[#19166](https://github.com/root-project/root/issues/19166)] - RooSimultaneous cannot accept ConditionalObservables properly.
  * [[#19116](https://github.com/root-project/root/issues/19116)] - Re-evaluate the implementation of RField<T>::TypeName
  * [[#19113](https://github.com/root-project/root/issues/19113)] - [Python][UHI] Incorrect number of entries reported in TH1 after slicing
  * [[#19104](https://github.com/root-project/root/issues/19104)] - RDataFrame Reads Garbage Data instead of File
  * [[#19068](https://github.com/root-project/root/issues/19068)] - [jupyter] let disable jsroot by default
  * [[#19045](https://github.com/root-project/root/issues/19045)] - The 3D visualization tutorial glViewerExercise.C doesn't work with web graphics
  * [[#19042](https://github.com/root-project/root/issues/19042)] - Ensure basic data classes are nothrow_move_constructible
  * [[#19038](https://github.com/root-project/root/issues/19038)] - [Python] TH1 equality operator pythonization not appropriate for ROOT histograms
  * [[#19035](https://github.com/root-project/root/issues/19035)] - The `rootcp --replace` logic behaves counterintuitively
  * [[#19022](https://github.com/root-project/root/issues/19022)] - Can not access REntry fields with DataVector in python
  * [[#18998](https://github.com/root-project/root/issues/18998)] - Ninja failed to build ROOT 6.36 for AlmaLinux10/x86_64 with cyclic deps error
  * [[#18988](https://github.com/root-project/root/issues/18988)] - [PyROOT] Test failures with Python 3.14 due to reference counting changes
  * [[#18982](https://github.com/root-project/root/issues/18982)] - [cling] Unexpected warning and errors printed by dictionary generation
  * [[#18974](https://github.com/root-project/root/issues/18974)] - `ROOT/RSpan.hxx` not found error in standalone Minuit2 source package
  * [[#18972](https://github.com/root-project/root/issues/18972)] - Segmentation fault while running interpreted macro
  * [[#18965](https://github.com/root-project/root/issues/18965)] - Support creating ntuples inside a directory with the RNTupleImporter
  * [[#18963](https://github.com/root-project/root/issues/18963)] - Add sanity checks to BuildIndex() in case of multiple events matching the keys
  * [[#18962](https://github.com/root-project/root/issues/18962)] - Add an option to suppress errors from BuildIndex()
  * [[#18955](https://github.com/root-project/root/issues/18955)] - Change in behaviour in `TTreeReaderValueBase::GetSetupStatus()`
  * [[#18953](https://github.com/root-project/root/issues/18953)] - error opening ZIP archive member (>4GB archive)
  * [[#18951](https://github.com/root-project/root/issues/18951)] - [gui] regression in TGNumberEntry
  * [[#18935](https://github.com/root-project/root/issues/18935)] - Yet another example of default template argument causing problems
  * [[#18929](https://github.com/root-project/root/issues/18929)] - [RF] Plotting extended pdf together with data unexpectedly scales pdf to number of events in data
  * [[#18924](https://github.com/root-project/root/issues/18924)] - Python Interface: assigning a numpy array to a std vector leaks memory
  * [[#18909](https://github.com/root-project/root/issues/18909)] - Patch Version Format has three cyphers
  * [[#18881](https://github.com/root-project/root/issues/18881)] - Overlaps found with 6.32 no longer found in 6.34+
  * [[#18863](https://github.com/root-project/root/issues/18863)] - ThreadSanitizer reports data race between TTree::Branch and TCling::Evaluate
  * [[#18833](https://github.com/root-project/root/issues/18833)] - segfault in TStreamerInfo::Compile with type involving std::function
  * [[#18827](https://github.com/root-project/root/issues/18827)] - ROOT uses the wrong value for Planck's constant
  * [[#18816](https://github.com/root-project/root/issues/18816)] - [Minuit2] Run time switch for multithreading in minuit2
  * [[#18811](https://github.com/root-project/root/issues/18811)] - assertion failure in clang::TemplateParameterList::getParam
  * [[#18792](https://github.com/root-project/root/issues/18792)] - Thread problem in TEnum::GetEnum
  * [[#18782](https://github.com/root-project/root/issues/18782)] - Segfault in TTree::Branch with seemingly simple custom class
  * [[#18778](https://github.com/root-project/root/issues/18778)] - [RF] Definitely lost reported by Valgrind in RooDataSet::emptyClone()
  * [[#18768](https://github.com/root-project/root/issues/18768)] - Regression in i-adding array.array to std::vector in PyROOT
  * [[#18751](https://github.com/root-project/root/issues/18751)] - Better thread safety in TClassEdit::CleanType
  * [[#18736](https://github.com/root-project/root/issues/18736)] - Test failure with -Dminimal=ON -Dtesting=ON
  * [[#18712](https://github.com/root-project/root/issues/18712)] - Bug Report: Incorrect Error Propagation in Histo2D Division
  * [[#18697](https://github.com/root-project/root/issues/18697)] - HS3/RooFit issue with ROOT_STANDARD_LIBRARY_PACKAGE usage
  * [[#18669](https://github.com/root-project/root/issues/18669)] - Incremental builds with `dev=ON` break when modifying Core header
  * [[#18665](https://github.com/root-project/root/issues/18665)] - hadd: Error in header does not cause hadd failure (as I'd expect)
  * [[#18663](https://github.com/root-project/root/issues/18663)] - Rootssh /tmp/root.XXXXXXXXX chmod issue and consequent THttpEngine (arg?) problem
  * [[#18658](https://github.com/root-project/root/issues/18658)] - Merging compatible TProfile2D does not preserve bin entries
  * [[#18654](https://github.com/root-project/root/issues/18654)] - TClassEdit name normalisation fails for unordered_map, leading to a crash in df104_CSVDataSource
  * [[#18643](https://github.com/root-project/root/issues/18643)] - [C++23][macOS] Failing test `roottest-root-io-tclass-make` with C++23
  * [[#18623](https://github.com/root-project/root/issues/18623)] - DaviX errors from ROOT.File.Cp
  * [[#18615](https://github.com/root-project/root/issues/18615)] - [RF] RooAbsReal::derivative needs to be fixed
  * [[#18556](https://github.com/root-project/root/issues/18556)] - Class methods not available to python in cases of transient data members of private nested class type
  * [[#18555](https://github.com/root-project/root/issues/18555)] - Consider not deleting the roottest build directory at each reconfigure
  * [[#18554](https://github.com/root-project/root/issues/18554)] - Spurious TMVA related rebuild
  * [[#18542](https://github.com/root-project/root/issues/18542)] - TMinuit warning about array out of bounds access
  * [[#18536](https://github.com/root-project/root/issues/18536)] - TClass::GetClass hoards memory when called for numerical types
  * [[#18535](https://github.com/root-project/root/issues/18535)] - Running an invariant reconfigure lead to spurious rebuild
  * [[#18524](https://github.com/root-project/root/issues/18524)] - Memory not released after TTree GetEntry() in PyROOT
  * [[#18520](https://github.com/root-project/root/issues/18520)] - Missing lock deep inside TClassEdit::ResolveTypedef
  * [[#18519](https://github.com/root-project/root/issues/18519)] - Concurrency issue with TClassEdit::ResolveTypedef and TClass::GetListOfMethods
  * [[#18499](https://github.com/root-project/root/issues/18499)] - CI artifacts are currently useless on Linux ...
  * [[#18455](https://github.com/root-project/root/issues/18455)] - Tiny memory leak reported by valgrind in TStreamerInfoActions
  * [[#18453](https://github.com/root-project/root/issues/18453)] - Some test fails on linux with Davix disabled
  * [[#18398](https://github.com/root-project/root/issues/18398)] - [ntuple] Allow usage of IMT with `RNTupleParallelWriter`
  * [[#18363](https://github.com/root-project/root/issues/18363)] - Incorrect name normalization for template with size_t template argument
  * [[#18316](https://github.com/root-project/root/issues/18316)] - [ntuple] RValue::GetRef<T> should type-check
  * [[#18301](https://github.com/root-project/root/issues/18301)] - Performance issue seen by CMS for ROOT's use of tbb::task_arena
  * [[#18172](https://github.com/root-project/root/issues/18172)] - [RF] Inefficient un-registering of histograms in HistFactory
  * [[#17855](https://github.com/root-project/root/issues/17855)] - Add `linspace`, `regspace`, and `logspace` to `ROOT::VecOps`
  * [[#17841](https://github.com/root-project/root/issues/17841)] - PyROOT multidimenstional int16_t arrays stopped working in 6.34
  * [[#17797](https://github.com/root-project/root/issues/17797)] - [RF] RooWorkspace.factory() parse dependence on LC_NUMERIC settings
  * [[#17334](https://github.com/root-project/root/issues/17334)] - TGNumberEntry string length checks are inaccurate/dangerous.
  * [[#17225](https://github.com/root-project/root/issues/17225)] - TFormula: Possibility of failure during dynamic compilation of predefined functions "gausn" and "landau"
  * [[#16981](https://github.com/root-project/root/issues/16981)] - Many roottest tests fail if you build ROOT with `-DNDEBUG` in `CXXFLAGS`
  * [[#16892](https://github.com/root-project/root/issues/16892)] - [ntuple] Add optimized bulk read of fixed-size arrays
  * [[#16805](https://github.com/root-project/root/issues/16805)] - A TChain whose trees have friend TChains stops updating
  * [[#16804](https://github.com/root-project/root/issues/16804)] - TChain as a friend of TTree gets stuck after the first file
  * [[#16719](https://github.com/root-project/root/issues/16719)] - PyTorch test/tutorials are (likely) using the same model files.
  * [[#16573](https://github.com/root-project/root/issues/16573)] - [Python] Overloads like `foo(bool=true)` not correctly resolved
  * [[#16488](https://github.com/root-project/root/issues/16488)] - WebGUI doesn't work with snap sandboxing
  * [[#16406](https://github.com/root-project/root/issues/16406)] - cppyy passing arguments by name to C++ binding
  * [[#16119](https://github.com/root-project/root/issues/16119)] - Handling of std::conditional by Cling in ROOT 6.32.02.
  * [[#16062](https://github.com/root-project/root/issues/16062)] - [PyROOT] Polymorphic Type Handling Issue in std::map with ROOT Python Bindings
  * [[#15927](https://github.com/root-project/root/issues/15927)] - RtypesCore.h does not have a signed character type
  * [[#15922](https://github.com/root-project/root/issues/15922)] - TScatter palette settings
  * [[#15422](https://github.com/root-project/root/issues/15422)] - Enable Fortran also for MacOS builds
  * [[#15309](https://github.com/root-project/root/issues/15309)] - [TMVA] Python module compatibility problems
  * [[#14917](https://github.com/root-project/root/issues/14917)] - ROOT installs libraries at the top-level site-packages directory
  * [[#14778](https://github.com/root-project/root/issues/14778)] - `-Dbuiltin_llvm=OFF` conflicts with use of 3D graphics (via Mesa built with LLVM)
  * [[#14581](https://github.com/root-project/root/issues/14581)] - [ROOT-9733] TRandom3 does not implement perfectly a Mersenne Twister PRNG
  * [[#14579](https://github.com/root-project/root/issues/14579)] - [ROOT-10190] Excluded fastcloningeventtree test in roottest.git should be re-enabled back
  * [[#14442](https://github.com/root-project/root/issues/14442)] - Cling not recognizing __sincospif on MacOS (from math.h)
  * [[#14155](https://github.com/root-project/root/issues/14155)] - roottest.root.hist.h2root Fails always on march=native builds and sporadically on all other platforms
  * [[#14024](https://github.com/root-project/root/issues/14024)] - New CI does not rebase roottest branch on top of roottest master
  * [[#14012](https://github.com/root-project/root/issues/14012)] - Reloading sparsehist.C triggers an assert in LLVM
  * [[#13843](https://github.com/root-project/root/issues/13843)] - RConcurrentHashColl.hxx 
  * [[#13548](https://github.com/root-project/root/issues/13548)] - Missing `test` installation folder in $ROOTSYS
  * [[#13274](https://github.com/root-project/root/issues/13274)] - [core] Inconsistent behavior in `TClassEdit::ResolveTypedef()` wrt. `TDataMember::GetTrueTypeName()`
  * [[#13252](https://github.com/root-project/root/issues/13252)] - root crashes due to typo in the macro
  * [[#13122](https://github.com/root-project/root/issues/13122)] - `TF1::DrawCopy()` does not copy histogram properties
  * [[#13101](https://github.com/root-project/root/issues/13101)] - [CMake] Existing ROOT headers from the system might be picked up when compiling ROOT
  * [[#12698](https://github.com/root-project/root/issues/12698)] - [core] Difference in `typedef`/`using` resolution in Windows and Linux builds
  * [[#12441](https://github.com/root-project/root/issues/12441)] - JsMVA should be updated to use the same jsroot version as the rest of ROOT
  * [[#12208](https://github.com/root-project/root/issues/12208)] - Deprecating RtypesCore.h
  * [[#11992](https://github.com/root-project/root/issues/11992)] - [RF] RooFormulaVar is too eager in variable name substitution
  * [[#11883](https://github.com/root-project/root/issues/11883)] - TTree indexes not correctly created when using branches made of vectors
  * [[#11806](https://github.com/root-project/root/issues/11806)] - Structured bindings for vectors
  * [[#11712](https://github.com/root-project/root/issues/11712)] - 1-1 not the same as (1-1) in TTree arrays indices
  * [[#11651](https://github.com/root-project/root/issues/11651)] - Syntax error leads to interpreter assert with `Must not nest within unloading transaction` when reading object
  * [[#11616](https://github.com/root-project/root/issues/11616)] - [ntuple] Check the usage of `Long_t` in RField (size of the type might differ between 32- and 64-bit platforms)
  * [[#11460](https://github.com/root-project/root/issues/11460)] - TF2 (with same name) construction can not be made thread safe without taking a lock
  * [[#11431](https://github.com/root-project/root/issues/11431)] - Installing with CMAKE_INSTALL_PYTHONDIR different from CMAKE_INSTALL_LIBDIR leads to incorrect runpath configuration
  * [[#11218](https://github.com/root-project/root/issues/11218)] - [DF] Add support for varied `Snapshot`
  * [[#11138](https://github.com/root-project/root/issues/11138)] - Broken CMake EXPORT set with external LLVM 
  * [[#10556](https://github.com/root-project/root/issues/10556)] - ROOT cannot read compiler includes when compiled with a ccache compiler wrapper 
  * [[#10522](https://github.com/root-project/root/issues/10522)] - complex numbers in RDataFrame (PyROOT)
  * [[#10423](https://github.com/root-project/root/issues/10423)] - Reading XYZVectors with TTreeReader is broken in some cases
  * [[#10264](https://github.com/root-project/root/issues/10264)] - Add ability to `RBrowser` to use file order
  * [[#10259](https://github.com/root-project/root/issues/10259)] - Improve `RBrowser` documentation
  * [[#10059](https://github.com/root-project/root/issues/10059)] - ROOT tag file contains hard coded paths
  * [[#9624](https://github.com/root-project/root/issues/9624)] - Crash when setting `LearningRateSchedule` for training Keras model on TMVA
  * [[#9530](https://github.com/root-project/root/issues/9530)] - TH1D::GetRandom and histograms with non-uniform binning
  * [[#9022](https://github.com/root-project/root/issues/9022)] - hadd with -n option misses some content
  * [[#8899](https://github.com/root-project/root/issues/8899)] - dataframe_concurrency LLVM valgrind error: SelectionDAG::Combine
  * [[#8708](https://github.com/root-project/root/issues/8708)] - [Build system] ROOT can wrongly pick up includes from system directories
  * [[#8642](https://github.com/root-project/root/issues/8642)] - Switching CMAKE_CXX_STANDARD breaks build
  * [[#8444](https://github.com/root-project/root/issues/8444)] - Very difficult to go from compression algorithm name (e.g. "zstd") to the number that must be passed e.g. to `hadd -f`
  * [[#7987](https://github.com/root-project/root/issues/7987)] - Error message when opening a GDML with TEveManager
  * [[#7822](https://github.com/root-project/root/issues/7822)] - [RF] RooDataHist::printDataHistogram should be renamed and moved
  * [[#7808](https://github.com/root-project/root/issues/7808)] - Bug in TKDE::Fill
  * [[#7786](https://github.com/root-project/root/issues/7786)] - Building built-in libAfterImage fails if build path contains `@`
  * [[#7565](https://github.com/root-project/root/issues/7565)] - ROOT 5 created TTree read with ROOT6 looks broken for int8_t
  * [[#7470](https://github.com/root-project/root/issues/7470)] - Dictionary generation fails for typedef to template specialization with default parameters
  * [[#7440](https://github.com/root-project/root/issues/7440)] - Merge reference documentation of MathCore and MathMore
  * [[#7210](https://github.com/root-project/root/issues/7210)] - [RF] Implement checking of parameter ranges
  * [[#7051](https://github.com/root-project/root/issues/7051)] - rootcp odd behavior
  * [[#6791](https://github.com/root-project/root/issues/6791)] - dictionary payload parsing diagnostics: no more error message when header not found
  * [[ROOT-11017](https://its.cern.ch/jira/browse/11017)] - Show correct number of digits in number based on its uncertainty
  * [[ROOT-10641](https://its.cern.ch/jira/browse/10641)] - mismatch after switch -Dbuiltin_zlib
  * [[ROOT-10484](https://its.cern.ch/jira/browse/10484)] - autoloading failure with module enabled
  * [[ROOT-10463](https://its.cern.ch/jira/browse/10463)] - Nested transaction not run
  * [[ROOT-10415](https://its.cern.ch/jira/browse/10415)] - Allow thisroot.sh to work as a symbolic lin
  * [[ROOT-10352](https://its.cern.ch/jira/browse/10352)] - TDirectory::fSeekParent points at the wrong location for sub-dirs
  * [[ROOT-10251](https://its.cern.ch/jira/browse/10251)] - TF1. Problem with the internal template function GetTheRightOp with std::bind
  * [[ROOT-10149](https://its.cern.ch/jira/browse/10149)] - [IO] TLeaf::GetLeafCounter broken for branches holding arrays of Double32_t with attributes
  * [[ROOT-9415](https://its.cern.ch/jira/browse/9415)] - Correct the 'interpreted' version of CheckTObjectHashConsistency
  * [[ROOT-9385](https://its.cern.ch/jira/browse/9385)] - CMake unexpected behavior with -Dfail-on-missing=ON
  * [[ROOT-9323](https://its.cern.ch/jira/browse/9323)] - TDataPointN::kDimension
  * [[ROOT-9028](https://its.cern.ch/jira/browse/9028)] - Int_t TTree::GetEntryWithIndex(Int_t major, Int_t minor = 0) using Int_t instead of Long64_t as TTree::GetEntryNumberWithIndex
  * [[ROOT-8842](https://its.cern.ch/jira/browse/8842)] - TTreeReaderFast crashes reading flat TTree (fork: bbockelm/root branch: root-bulkapi-fastread-v2)
  * [[ROOT-8730](https://its.cern.ch/jira/browse/8730)] - Confidence level for RooFitResult
  * [[ROOT-8577](https://its.cern.ch/jira/browse/8577)] - TTreeFormula alias branchname.NotAleafname to branchname.firstleafname
  * [[ROOT-8093](https://its.cern.ch/jira/browse/8093)] - Gaussian with exponential tail(s) in roofit
  * [[ROOT-7973](https://its.cern.ch/jira/browse/7973)] - TChain::LoadTree() can crash when a different tree is read from the same file
  * [[ROOT-7743](https://its.cern.ch/jira/browse/7743)] - TTree AddFriend not working
  * [[ROOT-7686](https://its.cern.ch/jira/browse/7686)] - TTreeFormula and function arguments
  * [[ROOT-7465](https://its.cern.ch/jira/browse/7465)] - TTreeFormula unable to access all but first index of vector branch
  * [[ROOT-7439](https://its.cern.ch/jira/browse/7439)] - Missing TMethodCall::Execute signature
  * [[ROOT-6960](https://its.cern.ch/jira/browse/6960)] - TFormula/TTreeFormula: Virtualize Optimize() and part of AnalyzeFunction()
  * [[ROOT-6889](https://its.cern.ch/jira/browse/6889)] - TParticlePDG::Stable for antiparticles
  * [[ROOT-6849](https://its.cern.ch/jira/browse/6849)] - TMath::HalfSampleMode Algorithm
  * [[ROOT-6741](https://its.cern.ch/jira/browse/6741)] - TTreeFormula parser has problems with member functions
  * [[ROOT-6431](https://its.cern.ch/jira/browse/6431)] - Not properly normalized paths cannot be added to include path
  * [[ROOT-6118](https://its.cern.ch/jira/browse/6118)] - TUnixSystem::FindFile fails with folder names containing a colon
  * [[ROOT-5567](https://its.cern.ch/jira/browse/5567)] - FindBranch fails where GetBranch succeeds if the branch name contains [
  * [[ROOT-5430](https://its.cern.ch/jira/browse/5430)] - TUrl inconsistently handles local path with double slashes at beginning
  * [[ROOT-3709](https://its.cern.ch/jira/browse/3709)] - Crash when writing object with schema rule
