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
 Philippe Canal, FNAL,\
 Olivier Couet, CERN/EP-SFT,\
 Marta Czurylo, CERN/EP-SFT,\
 Florine de Geus, CERN/EP-SFT and University of Twente,\
 Jonas Hahnfeld, CERN/EP-SFT and Goethe University Frankfurt,\
 Fernando Hueso Gonzalez, IFIC (CSIC-University of Valencia),\
 Stephan Hageboeck, CERN/EP-SFT,\
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
 Petr Jacka, Czech Technical University in Prague

## Deprecation and Removal

* The `RooDataSet` constructors to construct a dataset from a part of an existing dataset were deprecated in ROOT 6.36 and are now removed. This is to avoid interface duplication. Please use `RooAbsData::reduce()` instead, or if you need to change the weight column, use the universal constructor with the `Import()`, `Cut()`, and `WeightVar()` arguments.
* The `RooStats::HLFactory` class that was deprecated in ROOT 6.36 is now removed. It provided little advantage over using the RooWorkspace directly or any of the other higher-level frameworks that exist in the RooFit ecosystem.
* The build options `mysql`, `odbc` and `pgsql`, that were deprecated in ROOT 6.36, are now removed.
* The `TGLWSIncludes.h` header is deprecated and will be removed in ROOT 6.40
* The `ROOT::Math::TDataPointN` class that can be used with the `ROOT::Math::KDETree` was removed. Use the templated `TDataPoint<N>` instead.
* The Parallel ROOT Facility, `PROOF`, has been removed from the repository.
* After being deprecated for a long period, the `-r` option of `rootcling` has been removed.
* The `rpath` build option is deprecated. It is now without effect.
  Relative RPATHs to the main ROOT libraries are unconditionally appended to all ROOT executables and libraries if the operating system supports it.
  If you want a ROOT build without RPATHs, use the canonical CMake variable `CMAKE_SKIP_INSTALL_RPATH=TRUE`.
* The `TH1K` class is deprecated and will be removed in 6.40. It did not implement the `TH1` interface consistently, and limited the usability of the k-neighbors method it implemented by closely coupling the algorithm with the histogram class. Please use the new `TMath::KNNDensity` function that implements the same mathematical logic.

## Core Libraries
* Behavior change: when selecting a template instantiation for a dictionary, all the template arguments have to be fully defined - the forward declarations are not enough any more. The error prompted by the dictionary generator will be `Warning: Unused class rule: MyTemplate<MyFwdDeclaredClass>`.
* New expert option to reduce static startup cost of ROOT by setting environment variables
```bash
export ROOT_LDSYSPATH=$(LD_DEBUG=libs LD_PRELOAD=DOESNOTEXIST ls /tmp/DOESNOTEXIST 2>&1 | grep -m 1 "system search path" | sed 's/.*=//g' | awk '//{print $1}')
export CLING_LDSYSPATH=ROOT_LDSYSPATH
export CLING_CPPSYSINCL=$(LC_ALL=C c++ -xc++ -E -v /dev/null 2>&1 | sed -n '/^.include/,${/^ \/.*++/{p}}' | tr '\n' ':' | tr ' ' ':')
```
This caching reduces sub-process creation during initialization and can be useful when multiple ROOT instances or binaries linked to ROOT are executed (less system-calls, cleaner debugging).
* It is now possible to read a user configuration file (in jeargon, a "rootrc file") at startup in a custom path instead of the one in the home directory, by specifying its full path with the `ROOTENV_USER_PATH` environment variable.

## I/O

### TTree

* Behaviour change: the behaviour of `TChain::SetBranchStatus` has been aligned to the one of `TTree::SetBranchStatus`. In particular, when `SetBranchStatus` is called to deactivate all branches, a subsequent call to `TChain::SetBranchAddress` would override the previous instruction and activate that single branch. Instead `TTree::SetBranchAddress` respects the rule imposed by `SetBranchStatus`. If a user needs to activate only one or more branches, they should call `SetBranchStatus("brName", true)` on each branch that needs to be active in the TChain, like it was already necessary for a TTree. See https://github.com/root-project/root/pull/19221 for more details.

### RNTuple

* The parallel writer is now part of the public, stable API. The `RNTupleParallelWriter` and the closely related `RNTupleFillContext` moved from the `ROOT::Experimental` to the `ROOT` namespace.

## Math

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
  to numbers such as 8 would share one 3-d histogram among 8 threads, greatly reducing the memory consumption. This might slow down execution if the histograms
  are filled at very high rates. Use lower number in this case.
- Add HistoNSparseD action that fills a sparse N-dimensional histogram.

### Snapshot
- The Snapshot method has been refactored so that it does not need anymore compile-time information (i.e. either template arguments or JIT-ting) to know the input column types. This means that any Snapshot call that specifies the template arguments, e.g. `Snapshot<int, float>(..., {"intCol", "floatCol"})` is now redundant and the template arguments can safely be removed from the call. At the same time, Snapshot does not need to JIT compile the column types, practically giving huge speedups depending on the number of columns that need to be written to disk. In certain cases (e.g. when writing O(10000) columns) the speedup can be larger than an order of magnitude. The Snapshot template is now deprecated and it will issue a compile-time warning when called. The function overload is scheduled for removal in ROOT 6.40.
- The default compression setting for the output dataset used by Snapshot has been changed from 101 (ZLIB level 1, the TTree default) to 505 (ZSTD level 5). This is a better setting on average, and makes more sense for RDataFrame since now the Snapshot operation supports more than just the TTree output data format. This change may result in smaller output file sizes for your analyses that use Snapshot with default settings. During the 6.38 development release cycle, Snapshot will print information about this change once per program run. Starting from 6.40.00, the information will not be printed. The message can be suppressed by setting ROOT_RDF_SILENCE_SNAPSHOT_INFO=1 in your environment or by setting 'ROOT.RDF.Snapshot.Info: 0' in your .rootrc.

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

## ROOT executable

- Removed stray linebreak when running `root -q` with no input files.
  This ensures that there is no superfluous output when running `root` without the banner and without input files (`root -q -l`).

## Command-line utilities
- The `rootls` utility has a new native implementation and can now be run without Python.
  The options and the output of the new rootls are identical to the previous implementation but it should run faster (typically about 3 to 4x faster).

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
   * Implement `unzipJSON()` function for data embeding in jupyter
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


## Optimization of ROOT header files

More unused includes were removed from ROOT header files.
For instance, `#include "TMathBase.h"` was removed from `TString.h`.
This change may cause errors during compilation of ROOT-based code. To fix it, provide missing the includes
where they are really required.
This improves compile times and reduces code inter-dependency; see https://github.com/include-what-you-use/include-what-you-use/blob/master/docs/WhyIWYU.md for a good overview of the motivation.

## Versions of built-in packages

* The version of openssl has been updated to 3.5.0
* CFITSIO has been updated to 4.6.3
