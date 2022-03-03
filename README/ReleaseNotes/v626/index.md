% ROOT Version 6.26 Release Notes
% 2022-03-03
<a name="TopOfPage"></a>

## Introduction

ROOT version 6.26/00 was released on March 03, 2022.

For more information, see:

[http://root.cern](http://root.cern)

The following people have contributed to this new version:

 Sitong An, CERN/SFT,\
 Simone Azeglio, CERN/SFT,\
 Rahul Balasubramanian, NIKHEF/ATLAS,\
 Bertrand Bellenot, CERN/SFT,\
 Josh Bendavid, CERN/CMS,\
 Jakob Blomer, CERN/SFT,\
 Patrick Bos, Netherlands eScience Center,\
 Rene Brun, CERN/SFT,\
 Carsten D. Burgard, DESY/ATLAS,\
 Will Buttinger, STFC/ATLAS,\
 Philippe Canal, FNAL,\
 Olivier Couet, CERN/SFT,\
 Mattias Ellert, Uppsala University, \
 Gerri Ganis, CERN/SFT,\
 Andrei Gheata, CERN/SFT,\
 Enrico Guiraud, CERN/SFT,\
 Stephan Hageboeck, CERN/IT,\
 Jonas Hahnfeld, CERN/SFT,\
 Ahmat Hamdan, GSOC, \
 Fernando Hueso-González, University of Valencia,\
 Ivan Kabadzhov, CERN/SFT,\
 Shamrock Lee (@ShamrockLee),\
 Sergey Linev, GSI,\
 Javier Lopez-Gomez, CERN/SFT,\
 Pere Mato, CERN/SFT,\
 Emmanouil Michalainas, CERN/SFT, \
 Lorenzo Moneta, CERN/SFT,\
 Nicolas Morange, CNRS/IJCLab, \
 Axel Naumann, CERN/SFT,\
 Vincenzo Eduardo Padulano, CERN/SFT and UPV,\
 Max Orok, U Ottawa,\
 Alexander Penev, University of Plovdiv,\
 Danilo Piparo, CERN/SFT,\
 Fons Rademakers, CERN/SFT,\
 Jonas Rembser, CERN/SFT,\
 Enric Tejedor Saavedra, CERN/SFT,\
 Aaradhya Saxena, GSOC,\
 Oksana Shadura, UNL/CMS,\
 Sanjiban Sengupta, GSOC,\
 Harshal Shende, GSOC,\
 Federico Sossai, CERN/SFT,\
 Matevz Tadel, UCSD/CMS,\
 Vassil Vassilev, Princeton/CMS,\
 Wouter Verkerke, NIKHEF/ATLAS,\
 Zef Wolffs, NIKHEF/ATLAS,\
 Stefan Wunsch, CERN/SFT

## Deprecation, Removal, Backward Incompatibilities

The "Virtual MonteCarlo" facility VMC (`montecarlo/vmc`) has been removed from ROOT. The development of this package has moved to a [separate project](https://github.com/vmc-project/). ROOT's copy of VMC was deprecated since v6.18.
The previously deprecated packages memstat, memcheck have been removed. Please use for instance `valgrind` or memory sanitizers instead.
ROOT's "alien" package has been deprecated and will be removed in 6.28. Please contact ALICE software support if you still rely on it.


`TTree.AsMatrix` has been removed, after being deprecated in 6.24. Instead, please use `RDataFrame.AsNumpy` from now on as a way to read and process data in ROOT files and store it in NumPy arrays (a tutorial can be found [here](https://root.cern/doc/master/df026__AsNumpyArrays_8py.html)).
`TTreeProcessorMT::SetMaxTasksPerFilePerWorker` has been removed. `TTreeProcessorMT::SetTasksPerWorkerHint` is a superior alternative.
`TTree::GetEntry()` and `TTree::GetEvent()` no longer have 0 as the default value for the first parameter `entry`. We are not aware of correct uses of this function without providing an entry number. If you have one, please simply pass `0` from now on.
`TBufferMerger` is now production ready and not experimental anymore: `ROOT::Experimental::TBufferMerger` is deprecated, please use `ROOT::TBufferMerger` instead.

RooFit container classes marked as deprecated with this release: `RooHashTable`, `RooNameSet`, `RooSetPair`, and `RooList`. These classes are still available in this release, but will be removed in the next one. Please migrate to STL container classes, such as `std::unordered_map`, `std::set`, and `std::vector`.
The `RooFit::FitOptions(const char*)` command to steer [RooAbsPdf::fitTo()](https://root.cern.ch/doc/v628/classRooAbsPdf.html) with an option string in now deprecated and will be removed in ROOT v6.28. Please migrate to the RooCmdArg-based fit configuration. The former character flags map to RooFit command arguments as follows:

- `'h'` : RooFit::Hesse()
- `'m'` : RooFit::Minos()
- `'o'` : RooFit::Optimize(1)
- `'r'` : RooFit::Save()
- `'t'` : RooFit::Timer()
- `'v'` : RooFit::Verbose()
- `'0'` : RooFit::Strategy(0)

Subsequently, the `RooMinimizer::fit(const char*)` function and the [RooMCStudy](https://root.cern.ch/doc/v626/classRooMCStudy.html) constructor that takes an option string is deprecated as well.

## Core Libraries

### Interpreter

As of v6.26, cling diagnostic messages can be redirected to the ROOT error handler. Users may enable/disable this via `TCling::ReportDiagnosticsToErrorHandler()`, e.g.
```cpp
root [1] gInterpreter->ReportDiagnosticsToErrorHandler();
root [2] int f() { return; }
Error in <cling>: ROOT_prompt_2:1:11: non-void function 'f' should return a value [-Wreturn-type]
int f() { return; }
          ^
```
More details at [PR #8737](https://github.com/root-project/root/pull/8737).

Continuation of input lines using backslash `\` is supported in ROOT's prompt, e.g.
```
root [0] std::cout \
root (cont'ed, cancel with .@) [1]<< "ROOT\n";
```

ROOT now interprets code with optimization (`-O1`) by default, with
proper inlining optimization and alike. This accelerates especially "modern"
code (C++ stdlib, RDataFrame, etc) significantly. According to our measurements
the increased time for just-in-time compiling code is reasonable given the
runtime speed-up. Optimization can be switched with `.O 0`, `.O 1`, etc; the
current optimization level is shown by `.O`.
The CPP macro `NDEBUG` is now set unconditionally for interpreted code.
Note that symbols that have been emitted with a given optimization level will not get
re-emitted once the optimization level changes.

Unless ROOT is used with an interactive prompt (`root [0]`), ROOT does not
inject the pointer checks anymore, accelerating code execution at the cost
of not diagnosing the dereferencing of `nullptr` or uninitialized pointers.

Several internal optimizations of cling reduce the amount of symbols cling emits,
and improve the just-in-time compilation time.

## I/O Libraries

- `TDirectory::WriteObject` now always saves the object's title to the file if it is derived from `TObject` (PR [#8394](https://github.com/root-project/root/pull/8934)).

### Command line utilities

- `rootls` now follows the same logic of `TFile::ls()` to print the key cycle number and its tag when listing contents of a file with the `-l` option (PR [#7878](https://github.com/root-project/root/pull/7878)):
```
$: rootls -l https://root.cern/files/ttree_read_imt.root
TTree  Mar 13 17:17 2019 TreeIMT;2 "TTree for IMT test" [current cycle]
TTree  Mar 13 17:17 2019 TreeIMT;1 "TTree for IMT test" [backup cycle]
```
- `root` will now error on receiving unrecognized options, similarly to other command line tools (PR [#8868](https://github.com/root-project/root/pull/8868)):
```
$: root --random -z --nonexistingoption
root: unrecognized option '--random'
root: unrecognized option '-z'
root: unrecognized option '--nonexistingoption'
Try 'root --help' for more information.
```

## TTree Libraries

- `TTreeReader::GetEntryStatus` now always reports `kEntryBeyondEnd` after an event loop correctly completes. In previous versions, it could sometime return `kEntryNotFound` even for well-behaved event loops.
- Add `TEntryList::AddSubList` to specifically add a sub-list to the main list of entries. Consequently, add also a new option `"sync"` in `TChain::SetEntryList` to connect the sub-trees of the chain to the sub-lists of the entry list in lockstep (PR [#8660](https://github.com/root-project/root/pull/8660)).
- Add `TEntryList::EnterRange` to add all entries in a certain range `[start, end)` to the entry list (PR [#8740](https://github.com/root-project/root/pull/8740)).


## RNTuple

ROOT's experimental successor of TTree has been upgraded to the version 1 of the binary format specification. Compared to the v0 format, the header is ~40% smaller and the footer ~100% smaller (after zstd compression). More details in PR [#8897](https://github.com/root-project/root/pull/8897).
RNTuple is still experimental and is scheduled to become production grade in 2024. Thus, we appreciate feedback and suggestions for improvement.

If you have been trying RNTuple for a while, these are the other important changes that you will notice:

- Support for aligned friends (PR [#6979](https://github.com/root-project/root/pull/6979)). Refer to the `RNTupleReader::OpenFriends()` function.
- Cluster and page sizes in `RNTupleWriteOptions` now refer to their target size in bytes (as opposed to the number of entries). Defaults are 64 kB for the page size and 50 MB for the cluster size (PR [#8703](https://github.com/root-project/root/pull/8703)).
- Storing objects of user-defined classes via `TClass` now also includes members inherited from all the base classes (PR [#8552](https://github.com/root-project/root/pull/8552)).
- Support for RFields whose type is a typedef to some other type.


## RDataFrame

### New features

- Add [`Redefine`](https://root.cern/doc/master/classROOT_1_1RDF_1_1RInterface.html#a4e882a949c8a1022a38ec6936c2ff29c) to the `RDataFrame` interface, which allows to overwrite the value of an existing column.
- Add [`Describe`](https://root.cern/doc/master/classROOT_1_1RDF_1_1RInterface.html#a53f3e3d81e041a804481df228fe0081c) to the `RDataFrame` interface, which allows to get useful information, e.g. the columns and their types.
- Add [`DescribeDataset`](https://root.cern/doc/master/classROOT_1_1RDF_1_1RInterface.html#a1bc5b86a2a834bb06711fb535451146d) to the `RDataFrame` interface, which allows to get information about the dataset (subset of the output of Describe()).
- Add [`DefinePerSample`](https://root.cern/doc/master/classROOT_1_1RDF_1_1RInterface.html#a29d77593e95c0f84e359a802e6836a0e), a method which makes it possible to define columns based on the sample and entry range being processed. It is also a useful way to register callbacks that should only be called when the input dataset/TTree changes.
- Add [`HistoND`](https://root.cern/doc/master/classROOT_1_1RDF_1_1RInterface.html#a0c9956a0f48c26f8e4294e17376c7fea) action that fills a N-dimensional histogram.
- `Book` now supports just-in-time compilation, i.e. it can be called without passing the column types as template parameters (with some performance penalty, as usual).
- As an aid to `RDataSource` implementations with which collection sizes can be retrieved more efficiently than the full collection, `#var` can now be used as a short-hand notation for column name `R_rdf_sizeof_var`.
- Helpers have been added to export data from `RDataFrame` to RooFit datasets. See the "RooFit Libraries" section below for more details, or see [the tutorial](https://root.cern/doc/master/rf408__RDataFrameToRooFit_8C.html).
- Add initial support for expressing systematic variations through the `Vary` method and the `VariationsFor` helper function. As a major feature with important performance and usability benefits, we want to get this right: for now `VariationsFor` is kept in the `ROOT::RDF::Experimental` namespace to signal that we might make slight changes to the programming model in the future as we gather more user feedback. More information on working with systematic variations with RDataFrame are available in [the RDF user guide](https://root.cern/doc/master/classROOT_1_1RDataFrame.html).

### Notable changes in behavior

- Using `Alias`, it is now possible to register homonymous aliases (alternative column names) in different branches of the computation graph, in line with the behavior of `Define` (until now, aliases were required to be unique in the whole computation graph).
- The `Histo*D` methods now support the combination of scalar values and vector-like weight values. For each entry, the histogram is filled once for each weight, always with the same scalar value.
- The `Histo*D` methods do not work on columns of type `std::string` anymore. They used to fill the histogram with the integer value corresponding to each of the characters in the string. Please use `Fill` with a custom class to recover the old behavior if that was what was desired.

### Other improvements

- The scaling to a large amount of threads of computation graphs with many simple `Filter`s or `Define`s has been greatly improved, see also [this talk](https://indico.cern.ch/event/1036730/#1-a-performance-study-of-the-r) for more details
- The output format of `Display` has been significantly improved.
- The `Fill` method now correctly supports user-defined classes with arbitrary `Fill` signatures (see [#9428](https://github.com/root-project/root/issues/9428))

### Experimental Distributed RDataFrame

The distributed RDataFrame module has been improved. Now it supports sending RDataFrame tasks to a [Dask](https://dask.org/) scheduler. Through Dask, RDataFrame can be also scaled to a cluster of machines managed through a batch system like HTCondor or Slurm. Here is an example:

```python
import ROOT
from dask.distributed import Client
RDataFrame = ROOT.RDF.Experimental.Distributed.Dask.RDataFrame

# In a Python script the Dask client needs to be initalized in a context
# Jupyter notebooks / Python session don't need this
if __name__ == "__main__":
    client = Client("SCHEDULER_ADDRESS")
    df = RDataFrame("mytree","myfile.root", daskclient=client)
    # Proceed as usual
    df.Define("x","someoperation").Histo1D("x")
```

Other notable additions and improvements include:

- Enable triggering multiple distributed computation graphs through `RunGraphs`. This also allows sending both Spark and Dask jobs at the same time through a single function call.
- Greatly reduce distributed tasks processing overhead in TTree-based analyses by refactoring the translation from task metadata to RDataFrame object on the workers.
- Refactor triggering of the computation graph in the distributed tasks, so that it now runs with the Python GIL released. This allows interoperability with frameworks like Dask that run different Python threads along the main processing one.
- Set minimum Python version to use this tool to 3.7. This allows using more modern Python functionality in distributed RDataFrame code and is in line with the Python support provided by Spark and Dask.
- Add support for the following operations:
  - `DefinePerSample`
  - `HistoND`
  - `Redefine`
- Make sure a user-provided `npartitions` parameter to a distributed RDataFrame constructor always takes precedence over the value computed by default.
- Improve support for friend trees in distributed executions, now any kind of friendship layout between the main tree and the friend tree(s) is expected to work.
- Add support for TChain data sources with no tree name and multiple different tree subnames.
- Creating a distributed RDataFrame with an in-memory-only tree is prohibited, thus such usage now raises an error at runtime.

## Histogram Libraries

- Implement the `SetStats` method for `TGraph` to turn ON or OFF the statistics box display
  for an individual `TGraph`.

- Use INT_MAX in TH classes instead of an arbitrary big number.

- Implement option `AXIS`, to draw only axis, for TH2Poly.

- The logic to Paint fit parameters for TGraph was not following the one implemented for
  histograms. The v field described here was not working the same way. They are now implemented
  the same way.

- Implement the option X+ and Y+ for reverse axis on TGraph.

- TGLAxisPainter silently modified the histogram's Z axis parameters.

- Call automatically `Deflate` at drawing time of alphanumeric labels. It makes sense as
  nobody wants to see extra blank labels.

- The Confidence interval colors set by SetConfidenceIntervalColors (TRatioPlot) were inverted.

- Add GetZaxis for THStack.

- Fix Graph Errorbar Offsets for the new Marker Styles and thick markers.

- When the palette width is bigger than the palette height, the palette
  is automatically drawn horizontally.

- THStack::GetXaxis->SetRange did not auto-zoom Yaxis range.

- The Paint method of THStack always redrew the histograms in the sub-pads defined by the
  THStack drawing option "pads". Like the "pad dividing" the "histograms' drawing" should be
  done only the first time the THStack is painted otherwise any additional graphics objects
  added in one of the pads (created by the "pads" option) will be removed.

- Improve TRatioPlot axes drawing.

## Math Libraries

- `RVec` has been heavily re-engineered in order to add a small buffer optimization and to streamline its internals. The change should provide a small performance boost to
  applications that make heavy use of `RVec`s and should otherwise be user-transparent. Please report any issues you should encounter.
- I/O support of `RVec` objects has been optimized. As a side-effect, `RVec`s can now be read back as `std::vector`s and vice-versa.
- Add `ROOT::VecOps::Drop`, an operation that removes `RVec` elements at the specified indices.
- handy aliases `ROOT::RVecI`, `ROOT::RVecD`, `ROOT::RVecF`, ..., have been introduced as short-hands for `RVec<int>`, `RVec<double>`, `RVec<float>`, ...
- Add `VecOps::StableArgsort` and `VecOps::StableSort` operations


## RooFit Libraries
### Parallel calculation of likelihood gradients during fitting
This release features two new optional RooFit libraries: `RooFit::MultiProcess` and `RooFit::TestStatistics`.
To activate both, build with `-Droofit_multiprocess=ON`.

The `RooFit::TestStatistics` namespace contains a major refactoring of the `RooAbsTestStatistic`-`RooAbsOptTestStatistic`-`RooNLLVar` inheritance tree into:

1. statistics-based classes on the one hand;
2. calculation/evaluation/optimization based classes on the other hand.

The main selling point of using `RooFit::TestStatistics` from a performance point of view is the implementation of the `RooFit::MultiProcess` based `LikelihoodGradientJob` calculator class.
To use it to perform a "migrad" fit (using Minuit2), one should create a `RooMinimizer` using a new constructor with a `RooAbsL` likelihood parameter as follows:

```c++
using RooFit::TestStatistics::RooAbsL;
using RooFit::TestStatistics::buildLikelihood;

RooAbsPdf* pdf = ...;   // build a pdf
RooAbsData* data = ...; // get some data

std::shared_ptr<RooAbsL> likelihood = buildLikelihood(pdf, data, [OPTIONAL ARGUMENTS]);

RooMinimizer m(likelihood);
m.migrad();
```

The `RooMinimizer` object behaves as usual, except that behind the scenes it will now calculate each partial derivative on a separate process, ideally running on a separate CPU core.
This can be used to speed up fits with many parameters (at least as many as there are cores to parallelize over), since every parameter corresponds to a partial derivative.
The resulting fit parameters will be identical to those obtained with the non-parallelized gradients minimizer in most cases (see the usage notes linked below for exceptions).

In upcoming releases, further developments are planned:

- Benchmark/profile and optimize performance further
- Add a `RooAbsPdf::fitTo` interface around these new classes
- Achieve feature parity with existing `RooNLLVar` functionality, e.g. ranges are not yet supported.

For more details, consult the usage notes in the [TestStatistics README.md](https://github.com/root-project/root/tree/master/roofit/roofitcore/src/TestStatistics/README.md).
For benchmarking results on the prototype version of the parallelized gradient calculator, see the corresponding [CHEP19 proceedings paper](https://doi.org/10.1051/epjconf/202024506027).

### New pythonizations

Various new pythonizations are introduced to streamline your RooFit code in Python.

For a complete list of all pythonized classes and functions, please see the [RooFit pythonizations page in the reference guide](https://root.cern/doc/v626/group__RoofitPythonizations.html).
All RooFit Python tutorials have been updated to profit from all available pythonizations.

Some notable highlights are listed in the following.

#### Keyword argument pythonizations

All functions that take RooFit command arguments as parameters now accept equivalent Python keyword arguments, for example simplifying calls to [RooAbsPdf::fitTo()](https://root.cern/doc/v626/classRooAbsPdf.html#a5f79f16f4a26a19c9e66fb5c080f59c5) such as:
```Python
model.fitTo(data, ROOT.RooFit.Range("left,right"), ROOT.RooFit.Save())
```
which becomes:
```Python
model.fitTo(data, Range="left,right", Save=True)
```

#### String to enum pythonizations

Many functions that take an enum as a parameter now accept also a string with the enum label.

Take for example this expression:
```Python
data.plotOn(frame, ROOT.RooFit.DataError(ROOT.RooAbsData.SumW2)
```
Combining the enum pythonization with the keyword argument pythonization explained before, this becomes:
```Python
data.plotOn(frame, DataError="SumW2")
```

This pythonization is also useful for your calls to [RooFit::LineColor()](https://root.cern/doc/v626/group__Plotting.html#gad309cf5f63ec87ae5a7025d530f0398f) or [RooFit::LineStyle](https://root.cern/doc/v626/group__Plotting.html#gaf1f7922ba5965c1a5a9791a00ef354cb), to give some more common examples.

#### Implicit conversion from Python collections to RooFit collections

You can now benefit from implicit conversion from Python lists to RooArgLists, and from Python sets to RooArgSets.

For example, you can call [RooAbsPdf::generate()](https://root.cern/doc/v626/classRooAbsPdf.html#a87926e1044acf4403d8d5f1d366f6591) with a Python set to specify the observables:
```Python
pdf.generate({x, y, cut}, 10000)
```

Or, you can create a [RooPolynomial](https://root.cern/doc/v626/classRooPolynomial.html) from a Python list of coefficients:
```Python
ROOT.RooPolynomial("p", "p", x, [0.01, -0.01, 0.0004])
```

Note that here we benefit from another new feature: the implicit call to [RooFit::RooConst()](https://root.cern/doc/v626/group__CmdArgs.html#gaabf71812817894196e743cf2ef1d1e7b) when passing raw numbers to the RooFit collection constructors.

#### Allow for use of Python collections instead of C++ STL containers

Some RooFit functions take STL map-like types such as `std::map` as parameters, for example the [RooCategory constructor](https://root.cern/doc/v626/classRooCategory.html#ae63ae78231765d184b7a839c74746a49). In the past, you had to create the correct C++ class in Python, but now you can usually pass a Python dictionary instead. For example, a RooCategory can be created like this:
```Python
sample = ROOT.RooCategory("sample", "sample", {"Sample1": 1, "Sample2": 2, "Sample3": 3})
```

#### RooWorkspace accessors

In Python, you can now get objects stored in a [RooWorkspace](https://root.cern/doc/v626/classRooWorkspace.html) with the item retrieval operator, and the return value is also always downcasted to the correct type. That means in Python you don't have to use [RooWorkspace::var()](https://root.cern/doc/v626/classRooWorkspace.html#acf5f9126ee264c234721a4ed1f9bf837) to access variables or [RooWorkspace::pdf()](https://root.cern/doc/v626/classRooWorkspace.html#afa7384cece424a1a94a644bb05549eee) to access pdfs, but you can always get any object using square brackets. For example:
```Python
# w is a RooWorkspace instance that contains the variables `x`, `y`, and `z` for which we want to generate toy data:
model.generate({w["x"], w["y"], w["z"]}, 1000)
```

### New PyROOT functions for interoperability with NumPy and Pandas

New member functions of RooFit classes were introduced exclusively to PyROOT for better interoperability between RooFit and Numpy and Pandas:

- `RooDataSet.from_numpy`: Import a RooDataSet from a dictionary of numpy arrays (static method)
- `RooDataSet.to_numpy`: Export a RooDataSet to a dictionary of numpy arrays
- `RooDataSet.from_pandas`: Import a RooDataSet from a Pandas dataframe (static method)
- `RooDataSet.to_pandas`: Export a RooDataSet to a Pandas dataframe
- `RooDataHist.from_numpy`: Import a RooDataHist from numpy arrays with histogram counts and bin edges (static method)
- `RooDataHist.to_numpy`: Export a RooDataHist to numpy arrays with histogram counts and bin edges
- `RooRealVar.bins`: Get bin boundaries for a `RooRealVar` as a NumPy array

For more details, consult the tutorial [rf409_NumPyPandasToRooFit.py](https://root.cern/doc/v626/rf409__NumPyPandasToRooFit_8C.html).

### Modeling Effective Field Theory distributions with RooLadgrangianMorphFunc

The [**RooLagrangianMorphFunc**](https://root.cern/doc/v626/classRooLagrangianMorphFunc.html) class is a new RooFit class for modeling a continuous distribution of an observable as a function of the parameters of an effective field theory given the distribution sampled at some points in the parameter space.
Two new classes to help to provide this functionality:

* [RooRatio](https://root.cern/doc/v626/classRooRatio.html): computes the ratio of RooFit objects
* [RooPolyFunc](https://root.cern/doc/v626/classRooPolyFunc.html): multi-dimensional polynomial function, were [RooPolyFunc::taylorExpand()](https://root.cern/doc/v626/classRooPolyFunc.html#a21c6671e1b2391d08dbc62c5a5e7be38) can be used to obtain the (multi-dimensional) Taylor expansion up to the second order

For example usage of the RooLagrangianMorphFunc class, please consult the tutorials for a single parameter case ([rf711_lagrangianmorph.C](https://root.cern/doc/v626/rf711__lagrangianmorph_8C.html) / [.py](https://root.cern/doc/v626/rf711__lagrangianmorph_8py.html)) and for a multi-parameter case ([rf712_lagrangianmorphfit.C](https://root.cern/doc/v626/rf712__lagrangianmorphfit_8C.html) / [.py](https://root.cern/doc/v626/rf712__lagrangianmorphfit_8py.html)).

A `RooLagrangianMorphFunc` can also be created with the `RooWorkspace::factory` interface, showcased in [rf512_wsfactory_oper.C](https://root.cern/doc/v626/rf512__wsfactory__oper_8C.html) / [.py](https://root.cern/doc/master/rf512__wsfactory__oper_8py.html).

### Exporting and importing `RooWorkspace` to and from JSON and YML

The new component `RooFitHS3` implements serialization and
deserialization of `RooWorkspace` objects to and from JSON and YML.
The main class providing this functionality is
[RooJSONFactoryWSTool](https://root.cern/doc/v626/classRooJSONFactoryWSTool.html).
For now, this functionality is not feature complete with respect to
all available functions and pdfs available in `RooFit`, but provides
an interface that is easily extensible by users, which is documented
in the corresponding
[README](https://github.com/root-project/root/blob/master/roofit/hs3/README.md). It
is hoped that, though user contributions, a sufficiently comprehensive
library of serializers and deserializers will emerge over time.

For more details, consult the tutorial [rf515_hfJSON](https://root.cern/doc/v626/rf515__hfJSON_8py.html).

### Creating RooFit datasets from RDataFrame
RooFit now contains two RDataFrame action helpers, `RooDataSetHelper` and `RooDataHistHelper`, which allow for creating RooFit datasets by booking an action:
```c++
  RooRealVar x("x", "x", -5.,   5.);
  RooRealVar y("y", "y", -50., 50.);
  auto myDataSet = rdataframe.Book<double, double>(
    RooDataSetHelper{"dataset",          // Name   (directly forwarded to RooDataSet::RooDataSet())
                    "Title of dataset",  // Title  (                  ~ '' ~                      )
                    RooArgSet(x, y) },   // Variables to create in dataset
    {"x", "y"}                           // Column names from RDataFrame
  );
```
For more details, consult the tutorial [rf408_RDataFrameToRooFit](https://root.cern/doc/v626/rf408__RDataFrameToRooFit_8C.html).

### Storing global observables in RooFit datasets

RooFit groups model variables into *observables* and *parameters*, depending on if their values are stored in the dataset.
For fits with parameter constraints, there is a third kind of variables, called *global observables*.
These represent the results of auxiliary measurements that constrain the nuisance parameters.
In the RooFit implementation, a likelihood is generally the sum of two terms:

* the likelihood of the data given the parameters, where the normalization set is the set of observables (implemented by `RooNLLVar`)
* the constraint term, where the normalization set is the set of *global observables* (implemented by `RooConstraintSum`)

Before this release, the global observable values were always taken from the model/pdf.
With this release, a mechanism is added to store a snapshot of global observables in any `RooDataSet` or `RooDataHist`.
For toy studies where the global observables assume a different values for each toy, the bookkeeping of the set of global observables and in particular their values is much easier with this change.

Usage example for a model with global observables `g1` and `g2`:
```C++
auto data = model.generate(x, 1000); // data has only the single observables x
data->setGlobalObservables(g1, g2); // now, data also stores a snapshot of g1 and g2

// If you fit the model to the data, the global observables and their values
// are taken from the dataset:
model.fitTo(*data);

// You can still define the set of global observables yourself, but the values
// will be takes from the dataset if available:
model.fitTo(*data, GlobalObservables(g1, g2));

// To force `fitTo` to take the global observable values from the model even
// though they are in the dataset, you can use the new `GlobalObservablesSource`
// command argument:
model.fitTo(*data, GlobalObservables(g1, g2), GlobalObservablesSource("model"));
// The only other allowed value for `GlobalObservablesSource` is "data", which
// corresponds to the new default behavior explained above.
```

In case you create a RooFit dataset directly by calling its constructor, you can also pass the global observables in a command argument instead of calling `setGlobalObservables()` later:
```C++
RooDataSet data{"dataset", "dataset", x, RooFit::GlobalObservables(g1, g2)};
```

To access the set of global observables stored in a `RooAbsData`, call `RooAbsData::getGlobalObservables()`.
It returns a `nullptr` if no global observable snapshots are stored in the dataset.

For more information of global observables and how to attach them to the toy datasets, please take a look at the new
[rf613_global_observables.C](https://root.cern/doc/v626/rf613__global_observables_8C.html) / [.py](https://root.cern/doc/v626/rf613__global_observables_8py.html) tutorial.

### Changes in `RooAbsPdf::fitTo` behaviour for multi-range fits

The `RooAbsPdf::fitTo` and `RooAbsPdf::createNLL` functions accept a command argument to specify the fit range.
One can also fit in multiple ranges simultaneously.
The definition of such multi-range likelihoods for non-extended fits changes in this release.
Previously, the individual likelihoods were normalized separately in each range, which meant that the relative number of events in each sub-range was not used to estimate the PDF parameters.
From now on, the likelihoods are normalized by the sum of integrals in each range. This implies that the likelihood takes into account all inter-range and intra-range information.

### Deprecation of the `RooMinuit` class

The `RooMinuit` class was the old interface between RooFit and minuit. With ROOT version 5.24, the more general `RooMinimizer` adapter was introduced, which became the default with ROOT 6.08.

Before 6.26, it was possible to still use the `RooMinuit` by passing the `Minimizer("OldMinuit", "minimizer")` command argument to `RooAbsPdf::fitTo()`. This option is now removed.

### Increase of the `RooAbsArg` class version

The class version of `RooAbsArg` was incremented from 7 to 8 in this release. In some circumstances, this can cause warnings in `TStreamerInfo` for classes inheriting from `RooAbsArg` when reading
older RooFit models from a file. These warnings are harmless and can be avoided by incrementing also the class version of the inheriting class.

### Compile-time protection against creating empty `RooCmdArg`s from strings

The implicit [RooCmdArg](https://root.cern/doc/v626/classRooCmdArg.html) constructor from `const char*` was removed to avoid the accidental construction of meaningless RooCmdArgs that only have a name but no payload.
This causes new compiler errors in your code if you pass a string instead of a RooCmdArg to various RooFit functions, such as [RooAbsPdf::fitTo()](https://root.cern/doc/v626/classRooAbsPdf.html#a5f79f16f4a26a19c9e66fb5c080f59c5).
If this happens, please consult the documentation of [fitTo()](https://root.cern/doc/v626/classRooAbsPdf.html#a5f79f16f4a26a19c9e66fb5c080f59c5) to check which of the [free functions in the `RooFit` namespace](https://root.cern/doc/v626/group__CmdArgs.html) you need to use to achieve the desired configuration.

**Example** of an error that is now caught at compile time: confusing the [RooAbsPdf::fitTo()]() function signature with the one of [TH1::Fit()](https://root.cern/doc/v626/classTH1.html#a63eb028df86bc86c8e20c989eb23fb2a) and passing the fit range name as a string literal:

```C++
pdf.fitTo(*data, "r"); // ERROR!
// Will not compile anymore, as `"r"` is not a recognized command and will be ignored!
// Instead, to restrict to a range called "r", use:
pdf.fitTo(*data, RooFit::Range("r"));
```

## TMVA

### SOFIE : Code generation for fast inference of Deep Learning models

ROOT/TMVA SOFIE (“System for Optimized Fast Inference code Emit”) is a new package introduced in this release that generates C++ functions easily invokable for the fast inference of trained neural network models. It takes ONNX model files as inputs and produces C++ header files that can be included and utilized in a “plug-and-go” style.
This is a new development and it is currently still in experimental stage.

From ROOT command line, or in a ROOT macro you can use this code for parsing a model in ONNX file format
and generate C++ code that can be used to evaluate the model:

```
using namespace TMVA::Experimental;
SOFIE::RModelParser_ONNX parser;
SOFIE::RModel model = parser.Parse(“./example_model.onnx”);
model.Generate();
model.OutputGenerated(“./example_output.hxx”);
```
And an C++ header file will be generated. In addition also a text file, `example_output.dat` will be also generated. This file will contain the model weight values that will be used to initialize the model.
A full example for parsing an ONNX input file is given by the tutorial [`TMVA_SOFIE_ONNX.C`](https://root.cern/doc/master/TMVA__SOFIE__ONNX_8C.html). 

To use the generated inference code, you need to create a `Session` class and call the function `Session::inder(float *)`:

```
#include "example_output.hxx"
float input[INPUT_SIZE] = {.....};   // input data 
TMVA_SOFIE_example_model::Session s("example_output.dat");
std::vector<float> out = s.infer(input);
```

For using the ONNX parser you need to build ROOT with the configure option `tmva-sofie=ON`, which will be enabled when a Google Protocol Buffer library (`protobuf`, see https://developers.google.com/protocol-buffers) is found in your system.   

If you don't have `protobuf` and you don't want to install you can still use SOFIE, although with some more limited operator support parsing directly Keras `.h5` input files or PyTorch `.pt` files. 
In tis case you can convert directly the model to a `RModel` representation which can be used as above to generate the header and the weight file. 

For parsing a Keras input file you need to do:
```
SOFIE::RModel model = SOFIE::PyKeras::Parse("KerasModel.h5");
```
See the tutorial [`TMVA_SOFIE_Keras.C`](https://root.cern/doc/master/TMVA__SOFIE__Keras_8C.html).
For parsing a PyTorch input file : 
```
SOFIE::RModel model = SOFIE::PyTorch::Parse("PyTorchModel.pt",inputShapes);
```
where `inputShapes` is a `std::vector<std::vector<size_t>>` defining the inputs shape tensors. This information is required by PyTorch since it is not stored in the model.
A full example for parsing a PyTorch file is in the [`TMVA_SOFIE_PyTorch.C`](https://root.cern/doc/master/TMVA__SOFIE__PyTorch_8C.html) tutorial.

For using the Keras and/or the PyTorch parser you need to have installed Keras and/or PyTorch in your Python system and in addition build root with the support for `pymva`, obtained when configuring with `-Dtmva-pymva=On`.

For using the Keras and/or the PyTorch parser you need to have installed Keras and/or PyTorch in your Python system and in addition build root with the support for `pymva`, obtained when configuring with `-Dtmva-pymva=On`. 
 
Note that the created `SOFIE::RModel` class after parsing can be stored also in a ROOT file, using standard ROOT I/O functionality:
```
SOFIE::RModel model = SOFIE::PyKeras::Parse("KerasModel.h5");
TFile file("model.root","NEW");
model.Write();
file.Close(); 
```


## 2D Graphics Libraries

- Implement the option `X+` and `Y+` for reverse axis on TGraph.

- Offsets for axis titles with absolute-sized fonts (size%10 == 3) are now relative only to the font size (i.e. no longer relative to pad dimensions).

- In `TPaletteAxis` when the palette width is bigger than the palette height, the palette
  in automatically drawn horizontally.

- The `.tex` file produced when saving canvas as `.tex`, needed to be included in an existing
  LateX document to be visualized. The new `Standalone` option allows to generate a `.tex`
  file which can be directly processed by LateX (for example with the `pdflatex` command)
  in order to visualise it. This is done via the command:
```
canvas->Print(".tex", "Standalone");
```
  The generated  `.tex` file has the form:
```
\usepackage{tikz}
\usetikzlibrary{patterns,plotmarks}
\begin{document}
<----- here the graphics output
\end{document}
```

- Implement `ChangeLabel` in case `SetMoreLogLabels` is set. Implement it also for alphanumeric  labels.

- Some extra lines were drawn when histograms have negative content.

- TMathText did not display with high value coordinates.

- When a TCanvas contains TGraph with a huge number of points, the auto-placement of TLegend
  took ages. It may even look like an infinite loop.

- Fix title offsetting when using absolute-size fonts and multiple pads.

- The function TLatex::DrawLatex() only copied the Text-Attributes, but not the Line-Attributes
  to the newly created TLatex-Object.

- SaveAs png failed in batch mode with two canvases, one divided.

- The text size computed in TLatex::FirstParse was not correct in case the text precision was 3.

- Return pointer to the ABC object in DrawABC methods. This was not uniform.

## Geometry Libraries

- Prevent the TColor palette being silently set by TGeoPainter.

## GUI Libraries

- On Mac, with Cocoa, the pattern selector did not work anymore and the fit panel range did not work.

- Fix in Cocoa. XSGui crashed on Mac M1.

## WebGUI Libraries

- provide `--web=server` mode, which only printout window URLs instead of starting real web browser.
  Dedicated for the case when ROOT should be running as server application, providing different RWebWindow instances for connection.

## PyROOT

- A decorator called `@pythonization` is now provided to inject extra behaviour in user C++ classes that are used from Python. The aim here is to make C++ classes more "pythonic" or easier to use from Python. The way it works is the following: the user defines a function - the pythonizor - that is decorated with `@pythonization`; the decorator arguments specify the target C++ class or classes, and the pythonizor is responsible for injecting the new behaviour if those classes are actually used from the application. For a more complete description of the `@pythonization` decorator, please refer to this [entry](https://root.cern.ch/manual/python/#pythonizing-c-user-classes) of the ROOT manual and this [tutorial](https://root.cern/doc/master/pyroot002__pythonizationDecorator_8py.html).
- The `ROOT` Python module is now properly serializable so that it is automatically available in the Python environment if a function or ROOT object needs to be serialized. See issue [#6764](https://github.com/root-project/root/issues/6764) for a concrete usecase.
- Improve overload resolution of functions that accept classes with long inheritance trees. Now prefer to call the function overload of the most derived class type (PR [#9092](https://github.com/root-project/root/pull/9092)).

## Jupyter lab

- Let use created notebooks with viewers like https://nbviewer.jupyter.org/
- Fix problem with using of local JSROOT version


## Tutorials

- The tutorial games.C was not working properly

- Improve tutorial ErrorIntegral.C

- Schrödinger's Hydrogen Atom example.

- Tutorial demonstrating how the changing of the range can zoom into the histogram.

- Tutorial demonstrating how a Histogram can be read from a ROOT File.

- histMax.C: a tutorial demoing how the hist->GetMaximumBin() can be used.

## Class Reference Guide

- Images for ROOT7 tutorials can be generated, in json format, using the directive using
  `\macro_image (json)` in the macro header.

- Clarify THStack drawing options.

- Add missing documentation to TH1 functions.

- Restructure the the math reference guide.

- Make the web gui documentation visible in the reference guide

- Make clear THtml is legacy code. Add deprecated flag on PROOF and TGeoTrack.

- Improve many classes documentation: TContext, TTreePlayer, THistPainter, TGraph, TSelector,
  integrator, GUI, TH1, TH2, TH3, TColor classes ...

- Make the TFile layout doc visible in Reference Guide.

- Update the external links of the reference guide main page

- Reformat TMVA mathcore Unuran Roostats documentation .

## Build, Configuration and Testing Infrastructure

For users building from source the `latest-stable` branch and passing `-Droottest=ON` to the CMake command line, the corresponding revision of roottest pointed to by `latest-stable` will be downloaded as required.

ROOT now requires CMake version 3.16 or later.
ROOT cannot be built with C++11 anymore; the supported standards are currently C++14 and C++17.
ROOT's experimental features (RNTuple, RHist, etc) now require C++17.


## Bugs and Issues fixed in this release

* [[ROOT-10321]](https://sft.its.cern.ch/jira/browse/ROOT-10321) RResultPtr<Derived> should be convertible to RResultPtr<Base>
* [[ROOT-10486]](https://sft.its.cern.ch/jira/browse/ROOT-10486) [RDF] Decide how to proceed with multi-dimensional RDF fills & broadcasting
* [[ROOT-4373]](https://sft.its.cern.ch/jira/browse/ROOT-4373) Different behavior of ProjWData in plotting since root version 5.32.00
* [[ROOT-4899]](https://sft.its.cern.ch/jira/browse/ROOT-4899) Inherited object problems with FactorizePdf
* [[ROOT-4958]](https://sft.its.cern.ch/jira/browse/ROOT-4958) HistFactory does not warn or abort if the input histograms have variable bin widths
* [[ROOT-5022]](https://sft.its.cern.ch/jira/browse/ROOT-5022) RooCmdArg ProjWData not working if passed indirectly
* [[ROOT-5268]](https://sft.its.cern.ch/jira/browse/ROOT-5268) mistyped type results in bogus error message on prompt
* [[ROOT-5529]](https://sft.its.cern.ch/jira/browse/ROOT-5529) plotting RooRealSumPdf with RelativeExpected does not work first time
* [[ROOT-5557]](https://sft.its.cern.ch/jira/browse/ROOT-5557) PDF Normalization broken using NumCPU
* [[ROOT-6095]](https://sft.its.cern.ch/jira/browse/ROOT-6095) Constants of unnamed enums not found by lookup
* [[ROOT-6895]](https://sft.its.cern.ch/jira/browse/ROOT-6895) Crash in constructor of RooNLLVar
* [[ROOT-8367]](https://sft.its.cern.ch/jira/browse/ROOT-8367) type conversion failure in TBufferFile::ReadObjectAny
* [[ROOT-8440]](https://sft.its.cern.ch/jira/browse/ROOT-8440) Non extended likelihood fit in two separated ranges with RooFit
* [[ROOT-9202]](https://sft.its.cern.ch/jira/browse/ROOT-9202) regression in cling: a final `;` is expected after `int a,b`
* [[ROOT-9283]](https://sft.its.cern.ch/jira/browse/ROOT-9283) [TTree] Deleted friend trees are not deregistered from their friend
* [[ROOT-9530]](https://sft.its.cern.ch/jira/browse/ROOT-9530) RooFit side-band fit inconsistent with fit to full range
* [[ROOT-9548]](https://sft.its.cern.ch/jira/browse/ROOT-9548) Fit results differ with multiple ranges
* [[ROOT-9558]](https://sft.its.cern.ch/jira/browse/ROOT-9558) [DF] RDataFrame Snapshot throws for branches with branch name!=variable name
* [[ROOT-9737]](https://sft.its.cern.ch/jira/browse/ROOT-9737) [DF] Cannot use fill if the object is not an histogram
* [[ROOT-9861]](https://sft.its.cern.ch/jira/browse/ROOT-9861) [RF] RooCmdArgs copy pointers to temporaries
* [[ROOT-10038]](https://sft.its.cern.ch/jira/browse/ROOT-10038) RooChi2Var seems to ignore fit range
* [[ROOT-10396]](https://sft.its.cern.ch/jira/browse/ROOT-10396) Failure when instantiating RDataFrame::Fill
* [[ROOT-10582]](https://sft.its.cern.ch/jira/browse/ROOT-10582) cannot import name `TPyMultiGenFunction` from `ROOT`
* [[ROOT-10625]](https://sft.its.cern.ch/jira/browse/ROOT-10625) Issues with RDataFrame if name and leaflist of a TBranch are different
* [[ROOT-10636]](https://sft.its.cern.ch/jira/browse/ROOT-10636) TBB is built without support for exact exception propagation on some platforms
* [[ROOT-10806]](https://sft.its.cern.ch/jira/browse/ROOT-10806) [cling] valgrind error: source and destination overlap in memcpy
* [[ROOT-10829]](https://sft.its.cern.ch/jira/browse/ROOT-10829) pure virtual method called error
* [[ROOT-7587]](https://sft.its.cern.ch/jira/browse/ROOT-7587) Cling diagnostics shall be capturable
* [[#7754](https://github.com/root-project/root/issues/7754)] - Segfault when using schema evolution
* [[#9583](https://github.com/root-project/root/issues/9583)] - Interpreter autoload lookup failure when `runtime_cxxmodules=ON`
* [[#7561](https://github.com/root-project/root/issues/7561)] - [DF] Add convenience function to describe the dataframe
* [[#7692](https://github.com/root-project/root/issues/7692)] - Potential double-deletion in `CPyCppyy::CPPMethod::Destroy_`
* [[#6881](https://github.com/root-project/root/issues/6881)] - [TTreeReader] Partial leaf/branch names not recognized in cases that TTree::Draw supports
* [[#7912](https://github.com/root-project/root/issues/7912)] - TStreamerInfo Crash - V5 File Backward incompatibility introduced somewhere since 6.22.00
* [[#6764](https://github.com/root-project/root/issues/6764)] - Pickling functions using the ROOT module with cloudpickle breaks
* [[#7903](https://github.com/root-project/root/issues/7903)] - Invalid read in TClassEdit
* [[#7169](https://github.com/root-project/root/issues/7169)] - RDataFrame and CreatePoxy/TTreeReaderValue error for selection on string branch in Filter method
* [[#7400](https://github.com/root-project/root/issues/7400)] - [bindings] Clang-Tidy Clazy Analyzer leaks / bugs report
* [[#8060](https://github.com/root-project/root/issues/8060)] - Terminal flooded with cmake warnings when building with ninja+cmake 3.20
* [[#7501](https://github.com/root-project/root/issues/7501)] - Problem initialising C++ class members from PyROOT through object properties
* [[#8226](https://github.com/root-project/root/issues/8226)] - [DF] Crash in multi-thread Snapshot in sub-directory
* [[#8128](https://github.com/root-project/root/issues/8128)] - Many warnings coming from cppyy with Python 3.9
* [[#8046](https://github.com/root-project/root/issues/8046)] - TDirectory::GetObject is needlessly slow
* [[#8183](https://github.com/root-project/root/issues/8183)] - Python iteration over a TTreeReader is broken
* [[#8276](https://github.com/root-project/root/issues/8276)] - [DF] Possible use after delete of the functor passed to PassAsVec
* [[#8127](https://github.com/root-project/root/issues/8127)] - TGCommandLinePlugin autocomplete when single entry
* [[#8071](https://github.com/root-project/root/issues/8071)] - Problems compiling root 6.24.00 with gcc 11 on Fedora 34
* [[#8307](https://github.com/root-project/root/issues/8307)] - Issue with RooSimultaneous in 6.24.00 ?
* [[#8326](https://github.com/root-project/root/issues/8326)] - Jupyter + ROOT: too many open files
* [[#7727](https://github.com/root-project/root/issues/7727)] - TChain::CopyAddresses:0 warning in RDataFrame Snapshot of TChain with multiple files with objects
* [[#8295](https://github.com/root-project/root/issues/8295)] - TChain::AddClone failing for sub-branches of branch of type with ClassDef
* [[#6520](https://github.com/root-project/root/issues/6520)] - Integer types not available for Bulk API
* [[#8317](https://github.com/root-project/root/issues/8317)] - [DF] Compilation failure when a mutable lambda is passed to Foreach
* [[#8055](https://github.com/root-project/root/issues/8055)] - [JupyROOT] Notebook hangs when importing ROOT first from a secondary process
* [[#6681](https://github.com/root-project/root/issues/6681)] - Support UTF-8 encoding in ROOT7
* [[#7053](https://github.com/root-project/root/issues/7053)] - rootls does not seem to print the cycle number information
* [[#7959](https://github.com/root-project/root/issues/7959)] - RColor must be able to be constructed as "0.2" from RPalette
* [[#7966](https://github.com/root-project/root/issues/7966)] - Merge `RPadUserAxisBase` `RAxisAttr`
* [[#8037](https://github.com/root-project/root/issues/8037)] - [RF] Test schema evolution of RooDataHist
* [[#7662](https://github.com/root-project/root/issues/7662)] - `root-config --has-feature` should complain if `feature` is not a ROOT feature
* [[#7009](https://github.com/root-project/root/issues/7009)] - CMake fails when there is a '+' in the source directory.  It may have problems with other REGEX characters.
* [[#6577](https://github.com/root-project/root/issues/6577)] - Use CMAKE_CXX_FLAGS_${BUILD_TYPE}_INIT to overwrite cmake defaults
* [[#8284](https://github.com/root-project/root/issues/8284)] - [ntuple] TKey with the same name as requested RNTuple causes internal RNTupleReader errors
* [[#8281](https://github.com/root-project/root/issues/8281)] - ROOT 6.24 fails to compile with GCC 11.1 in C++11 mode
* [[#8267](https://github.com/root-project/root/issues/8267)] - -Wundef RStringView MSVC_LANG
* [[#8250](https://github.com/root-project/root/issues/8250)] - Root 6.20.06+ - New bug with compiled code using TClass / dictionaries
* [[#8180](https://github.com/root-project/root/issues/8180)] - ROOT 6.24 breaks Alice O2 due to symbol confusion with system llvm 11
* [[#7081](https://github.com/root-project/root/issues/7081)] - `cmake -Droottest=On` tests wrong ROOT build!
* [[#6410](https://github.com/root-project/root/issues/6410)] - Without memstat support, `root -memstat` should not be available
* [[#7936](https://github.com/root-project/root/issues/7936)] - Test failures with ROOT master on Gentoo
* [[#8033](https://github.com/root-project/root/issues/8033)] - StreamerElement retrieved from file has incorrect element name for (some) arrays.
* [[#8022](https://github.com/root-project/root/issues/8022)] - TBB exposed in public includes of Thread library
* [[#7829](https://github.com/root-project/root/issues/7829)] - [DF] Introduce GetDatasetInfo and add info to Describe
* [[#8120](https://github.com/root-project/root/issues/8120)] - Implement a Pythonisation for RooAbsCollection.addOwned()
* [[#8074](https://github.com/root-project/root/issues/8074)] - [ntuple] Run DAOS unit tests if `-Dtesting=ON` was specified
* [[#8079](https://github.com/root-project/root/issues/8079)] - [ntuple, daos] Implement reading and writing of sealed pages
* [[#8283](https://github.com/root-project/root/issues/8283)] - [rntuple] Add performance counters to classes derived from `RPageSink`
* [[#8360](https://github.com/root-project/root/issues/8360)] - [ntuple] Move common performance counters to `RPage{Sink,Source}` base class
* [[#7996](https://github.com/root-project/root/issues/7996)] - TColor doc html tag issues "a name"
* [[#8305](https://github.com/root-project/root/issues/8305)] - [cmake] directory checking using string(REGEX) should be replaced with string(FIND)
* [[#7853](https://github.com/root-project/root/issues/7853)] - [ntuple] Allow the user to customize cluster/page sizes
* [[#8145](https://github.com/root-project/root/issues/8145)] - tutorial/math/exampleFunction.py, tutorials/math/multivarGaus.C not disabled without MathMore
* [[#8129](https://github.com/root-project/root/issues/8129)] - Warnings compiling with g++-11.1.0
* [[#8342](https://github.com/root-project/root/issues/8342)] - Tutorials order in the reference guide
* [[#8438](https://github.com/root-project/root/issues/8438)] - Error pickling derived class in PyROOT
* [[#8252](https://github.com/root-project/root/issues/8252)] - Bug when using a Convolution in TF1 the  second time with a different range
* [[#8519](https://github.com/root-project/root/issues/8519)] - Documentation for RooChi2Var missing
* [[#6974](https://github.com/root-project/root/issues/6974)] -  pyROOT doesn't work with ipyparallel
* [[#8450](https://github.com/root-project/root/issues/8450)] - [DF] `Display` gets confused by friend columns, prints them twice
* [[#8465](https://github.com/root-project/root/issues/8465)] - [DF] Cannot pass an lvalue to Fill, Book
* [[#8269](https://github.com/root-project/root/issues/8269)] - TGCommandPlugin TTimer
* [[#8537](https://github.com/root-project/root/issues/8537)] - Warnings when configuring with CUDA 11.3
* [[#8554](https://github.com/root-project/root/issues/8554)] - Add ability to run `RBrowser` and similar commands in "server mode"
* [[#8513](https://github.com/root-project/root/issues/8513)] - Fail to build core/meta/src/TEnumConstant.cxx in gcc 11. Need to include <limits> in header file core/foundation/inc/ROOT/libcpp_string_view.h
* [[#8300](https://github.com/root-project/root/issues/8300)] - TTree::GetEntry should not take the entry number as default parameter
* [[#8599](https://github.com/root-project/root/issues/8599)] - Preserve JSROOT when generating reference guide
* [[#7584](https://github.com/root-project/root/issues/7584)] - [DF] Distributed RDataFrame doesn't handle friend trees correctly
* [[#8551](https://github.com/root-project/root/issues/8551)] - Compiler Version not correctly parsed
* [[#8547](https://github.com/root-project/root/issues/8547)] - Crash in TDataMember::GetOffsetCint() method
* [[#8205](https://github.com/root-project/root/issues/8205)] - [ntuple, daos] Enable user customization of DAOS object class
* [[#8136](https://github.com/root-project/root/issues/8136)] - Possible buffer overflow in TF1 / TString
* [[#8618](https://github.com/root-project/root/issues/8618)] - TH2 palette tick length is silently modified when passed to TGLAxisPainter
* [[#8503](https://github.com/root-project/root/issues/8503)] - ranluxpp code is using left shift of negative value -1 which is undefined per the C standard
* [[#8424](https://github.com/root-project/root/issues/8424)] - Installation without internet access
* [[#8292](https://github.com/root-project/root/issues/8292)] - Compilation error on redhat 8.3 / no internet
* [[#8297](https://github.com/root-project/root/issues/8297)] - valgrind TThread Init Printf leak definitely lost bytes
* [[#8604](https://github.com/root-project/root/issues/8604)] - [RF] Remove "No RooMinimizer" code paths.
* [[#8641](https://github.com/root-project/root/issues/8641)] - RooStats::HypoTestInverterResult::FindIndex fails if xvalue == 1
* [[#8000](https://github.com/root-project/root/issues/8000)] - RooAddPdf fails if pdfs have the same name
* [[#7673](https://github.com/root-project/root/issues/7673)] - Feature request: allow TH2Poly to be drawn with "AXIS" option
* [[#7666](https://github.com/root-project/root/issues/7666)] - [DF] Add RunGraphs to df103*.(py|C)
* [[#7659](https://github.com/root-project/root/issues/7659)] - hesse fails after calling getVal on nll
* [[#7654](https://github.com/root-project/root/issues/7654)] - Build fails with gcc 9 and CMake < 3.11
* [[#7639](https://github.com/root-project/root/issues/7639)] - `rootbench-benchJohnson` fills log with "use fixCoefNormalization!"
* [[#7633](https://github.com/root-project/root/issues/7633)] - Provide `TIter &TIter::operator=(TIterator *it)` signature
* [[#7632](https://github.com/root-project/root/issues/7632)] - [Doc] No syntax highlighting for Python in Doxygen
* [[#7619](https://github.com/root-project/root/issues/7619)] - Multiple number of memory leaks in RooFit 
* [[#7536](https://github.com/root-project/root/issues/7536)] - [RF] Clang-Tidy Clazy Warnings
* [[#7532](https://github.com/root-project/root/issues/7532)] - [hist] Clang-Tidy Clazy warnings
* [[#7531](https://github.com/root-project/root/issues/7531)] - [graf3d] Clang-Tidy Clazy Warnings
* [[#7528](https://github.com/root-project/root/issues/7528)] - [net] Clang-Tidy Clazy Warnings
* [[#7527](https://github.com/root-project/root/issues/7527)] - [rootx main] Clang-Tidy Clazy Warnings
* [[#7526](https://github.com/root-project/root/issues/7526)] - [tmva] Clang-tidy Clazy warnings
* [[#7525](https://github.com/root-project/root/issues/7525)] - [sql html] Clang-tidy Clazy Warnings
* [[#8052](https://github.com/root-project/root/issues/8052)] - RooUniform in RooArgList in RooProdPdf: segmentation violation during fit
* [[#8523](https://github.com/root-project/root/issues/8523)] - [RF] Translate all RooFit tutorials to Python
* [[#8505](https://github.com/root-project/root/issues/8505)] - TEntryList + TChain reads wrong number of entries if reading the same file multiple times
* [[#8767](https://github.com/root-project/root/issues/8767)] - [netxng] Crash in on-exit destruction of an TNetXNGFile object
* [[#8665](https://github.com/root-project/root/issues/8665)] - GrabKey not working with kKeyControlMask on TGX11
* [[#8471](https://github.com/root-project/root/issues/8471)] - Add Lambert W function to TMath
* [[#8739](https://github.com/root-project/root/issues/8739)] - [DF] Cannot read files that don't have a `.root` extension with IMT on
* [[#8490](https://github.com/root-project/root/issues/8490)] - Avoid Python globbing when we actually need TChain globbing
* [[#8774](https://github.com/root-project/root/issues/8774)] - [TGaxis] secAxis->SetMaxDigits changes all other axes
* [[#8807](https://github.com/root-project/root/issues/8807)] - Move and rename the `tokenise()` function for splitting strings to `core/foundation`
* [[#8857](https://github.com/root-project/root/issues/8857)] - [DF] `Redefine()`ing a `Defined()` column causes segmentation violation 
* [[#8693](https://github.com/root-project/root/issues/8693)] - "Running utility command for G__TreeFormulaEvent" even if the build should be no-op
* [[#7893](https://github.com/root-project/root/issues/7893)] - The world gets rebuilt after a small git checkout
* [[#8713](https://github.com/root-project/root/issues/8713)] - [tree] TTreeCache is turned off when `fAutoFlush == 0`
* [[#8750](https://github.com/root-project/root/issues/8750)] - Support chains with subtrees with different names in distributed RDF
* [[#8598](https://github.com/root-project/root/issues/8598)] - Call automatically LabelsDeflate at drawing time
* [[#8141](https://github.com/root-project/root/issues/8141)] - Cmake get_directory_property fails on external LLVM
* [[#8931](https://github.com/root-project/root/issues/8931)] - `TDirectory::RegisterGDirectory` is MT unsafe
* [[#8892](https://github.com/root-project/root/issues/8892)] - Exception in TGeoMixture::ComputeDerivedQuantities
* [[#8707](https://github.com/root-project/root/issues/8707)] - Add ctest fixtures to root / roottest
* [[#8011](https://github.com/root-project/root/issues/8011)] - Bug in RooAbsPdf::extendedTerm when expected events is negative
* [[#7332](https://github.com/root-project/root/issues/7332)] - Improvements in TUnuran and ROOT::Math::DistSampler
* [[#8850](https://github.com/root-project/root/issues/8850)] - `root` command ignores unknown options - it should complain instead 
* [[#8964](https://github.com/root-project/root/issues/8964)] - JavaScript complains when displaying MultiGraph
* [[#8762](https://github.com/root-project/root/issues/8762)] - [rint] Properly support line continuation after backslash `\` 
* [[#8933](https://github.com/root-project/root/issues/8933)] - CMAKE_INSTALL_PYTHONDIR on macOS creates broken symlinks
* [[#8981](https://github.com/root-project/root/issues/8981)] - [Warning] TStreamerInfo.cxx: ‘this’ pointer is null [-Wnonnull]
* [[#9017](https://github.com/root-project/root/issues/9017)] - Access of deleted object during hadd tear down.
* [[#8783](https://github.com/root-project/root/issues/8783)] - Branch sync of roottest broken for latest-stable
* [[#8712](https://github.com/root-project/root/issues/8712)] - Documentation in C++ classes gets mixed with PyROOT doc box
* [[#8123](https://github.com/root-project/root/issues/8123)] - Add RooArgSet storage to RooAbsData for GlobalObservables
* [[#6745](https://github.com/root-project/root/issues/6745)] - Add RDataFrame::DefinePerSample
* [[#9032](https://github.com/root-project/root/issues/9032)] - [ntuple,daos] `RNTupleWriter::Recreate()` does not overwrite data in DAOS container
* [[#7856](https://github.com/root-project/root/issues/7856)] - [ntuple] Add support for storing class hierarchies in RNTuple
* [[#6796](https://github.com/root-project/root/issues/6796)] - [VecOps] Interaction between memory adoption and `clear` yields wrong results
* [[#9020](https://github.com/root-project/root/issues/9020)] - -Wclass-memaccess with builtin_tbb and gcc 11
* [[#8947](https://github.com/root-project/root/issues/8947)] - [doc] Building Doxygen documentation pollutes source directory
* [[#8643](https://github.com/root-project/root/issues/8643)] - Some parts of ROOT still mention C++11
* [[#6868](https://github.com/root-project/root/issues/6868)] - TGDMLParse::Tessellated handling of "type" attribute (not added with G4 10.5 export)
* [[#8817](https://github.com/root-project/root/issues/8817)] - Wrong function overload selection with long inheritance chain
* [[#9098](https://github.com/root-project/root/issues/9098)] - typo in TGraphErrors docs; expand the docs for high and low errors
* [[#6981](https://github.com/root-project/root/issues/6981)] - [DF] Display should list all elements of a collection by default, like TTree::Scan
* [[#7205](https://github.com/root-project/root/issues/7205)] - RDataFrame::Display should display all requested columns
* [[#9143](https://github.com/root-project/root/issues/9143)] - star marker invisible in latex output
* [[#8086](https://github.com/root-project/root/issues/8086)] - X11 Display Depth 30 no fonts displayed
* [[#6991](https://github.com/root-project/root/issues/6991)] - THStack::GetXaxis->SetRange does not auto-zoom Yaxis range
* [[#6985](https://github.com/root-project/root/issues/6985)] - TGPictureButton does not load image from disk
* [[#9118](https://github.com/root-project/root/issues/9118)] - Problem running weighted binned fit in batch mode
* [[#9140](https://github.com/root-project/root/issues/9140)] - [ntuple] `const`-qualified members of a user-defined struct do not work
* [[#9176](https://github.com/root-project/root/issues/9176)] - Add documentation for saving TCanvas and TPad (reference and manual)
* [[#8778](https://github.com/root-project/root/issues/8778)] - rootdrawtree crash
* [[#9133](https://github.com/root-project/root/issues/9133)] - [ntuple] Memory leak when reading std::vector of complex objects
* [[#9207](https://github.com/root-project/root/issues/9207)] - WebSocket data handler does not check opcode 
* [[#9116](https://github.com/root-project/root/issues/9116)] - [DF] Beauty fixes to Display 
* [[#8389](https://github.com/root-project/root/issues/8389)] - cling assertion crash
* [[#9011](https://github.com/root-project/root/issues/9011)] - TMultiGraph wrong scale with logarithmic axes : regession introduced in 6.22
* [[#9117](https://github.com/root-project/root/issues/9117)] - [DF] Use RVecI, RVecD, RVecF aliases in tutorials
* [[#8622](https://github.com/root-project/root/issues/8622)] - PyROOT triggers a warning about [[nodiscard]] vector::empty with GCC11
* [[#8098](https://github.com/root-project/root/issues/8098)] - TColor palettes identify CVD-friendly
* [[#9172](https://github.com/root-project/root/issues/9172)] - Number of divisions of ratio plot
* [[#9294](https://github.com/root-project/root/issues/9294)] - Root Lockguard
* [[#9189](https://github.com/root-project/root/issues/9189)] - TEfficiency constructors not appended to current directory
* [[#9263](https://github.com/root-project/root/issues/9263)] - TRatioPlot axes sync
* [[#9253](https://github.com/root-project/root/issues/9253)] - ROOT does not compile on musl libc
* [[#9231](https://github.com/root-project/root/issues/9231)] - Error creating SqliteDataFrame
* [[#9362](https://github.com/root-project/root/issues/9362)] - Candle plot with low statistic histograms makes strange output
* [[#9202](https://github.com/root-project/root/issues/9202)] - [ntuple] Members of an aliased type do not resolve to the underlying type
* [[#9315](https://github.com/root-project/root/issues/9315)] - [PyROOT] Memory leak when iterating over std::map from Python
* [[#8377](https://github.com/root-project/root/issues/8377)] - [ntuple] Add field description to RNTupleDescriptor::PrintInfo output?
* [[#9383](https://github.com/root-project/root/issues/9383)] - [TGeo][Regression] RadLen and IntLen changed by factor of 10 (default unit system)
* [[#8072](https://github.com/root-project/root/issues/8072)] - Failures with root 6.24.00 on Fedora 33 ppc64le
* [[#9297](https://github.com/root-project/root/issues/9297)] - ROOT 6.24 debug build failed on ppc64le
* [[#9384](https://github.com/root-project/root/issues/9384)] - TPad::SaveAs .tex Standalone option
* [[#9424](https://github.com/root-project/root/issues/9424)] - ROOT 6.24 failed to build with GCC10 or 11 on ppc64le arch
* [[#9429](https://github.com/root-project/root/issues/9429)] - [DF] Crash with distributed RDataFrame on dask with dask_jobqueue
* [[#7861](https://github.com/root-project/root/issues/7861)] - [ntuple] RNTuple error when serializing TClass typedef members
* [[#9453](https://github.com/root-project/root/issues/9453)] - Compiler failure in `tmva/sofie/inc/TMVA/ROperator_Conv.hxx`
* [[#9312](https://github.com/root-project/root/issues/9312)] - Possible optimizer optimization
* [[#9173](https://github.com/root-project/root/issues/9173)] - Root v6.24/06 using XrdSysDNS: Failure to compile using g++ 10.3.1
* [[#9468](https://github.com/root-project/root/issues/9468)] - Roofit doesn't compile on Windows 64 bit
* [[#9428](https://github.com/root-project/root/issues/9428)] - [DF] Fill method does not support arbitrary signatures
* [[#7381](https://github.com/root-project/root/issues/7381)] - [DF] Let Aliases be defined per computation graph branch, not globally
* [[#8073](https://github.com/root-project/root/issues/8073)] - Test failures in root7 tests
* [[#9487](https://github.com/root-project/root/issues/9487)] - SofieCompileModels_PyTorch.vcxproj fails to compile on pytest version 6.2.5
* [[#9547](https://github.com/root-project/root/issues/9547)] - RooFit crashes when ROOT is built with Clang 13
* [[#9379](https://github.com/root-project/root/issues/9379)] - Make sure cling also adds `-DNDEBUG` to the compilation flags besides `-O1`
* [[#8893](https://github.com/root-project/root/issues/8893)] - [DF] `Describe` output does not display correctly in Jupyter notebooks
* [[#7125](https://github.com/root-project/root/issues/7125)] - lib/module.idx needs to be refreshed up library removal.
* [[#6966](https://github.com/root-project/root/issues/6966)] - Return pointer to XXX object in DrawXXX methods.
* [[#9543](https://github.com/root-project/root/issues/9543)] - roottest-root-treeformula-stl-make crashes during process termination
* [[#8987](https://github.com/root-project/root/issues/8987)] - Missing operator= in code generated by MakeProject
* [[#9626](https://github.com/root-project/root/issues/9626)] - THStack docu broken link and TSpectrum TGeoMatrix
* [[#9600](https://github.com/root-project/root/issues/9600)] - Missing dependency when building roottest as part of ROOT.
* [[#7366](https://github.com/root-project/root/issues/7366)] - ACLiC compilation confuses compiled binaries with shared objects, breaking compilation in some cases
* [[#7874](https://github.com/root-project/root/issues/7874)] - [RF] IMT parallel for in the computation library
* [[#9664](https://github.com/root-project/root/issues/9664)] - [Cling] Interpreter regression 6.24/06 -> 6.25/02
* [[#9730](https://github.com/root-project/root/issues/9730)] - When working on a Windows partition from Linux, rootcling is not able to rename an output file
* [[#9632](https://github.com/root-project/root/issues/9632)] - Iteration on `std.vector['char']` is broken
* [[#9740](https://github.com/root-project/root/issues/9740)] - Crash after zoom and unzoom with secondary axes
* [[#9240](https://github.com/root-project/root/issues/9240)] - [DF] Issues managing `TClonesArray` branches
* [[#9136](https://github.com/root-project/root/issues/9136)] - [I/O] Cannot read RVecs written with v6.24 with TTreeReader in current master
* [[#8428](https://github.com/root-project/root/issues/8428)] - I/O customization rule not run on split sub-object of a non-collection object.
* [[#9316](https://github.com/root-project/root/issues/9316)] - [TMVA] Experimental::RBDT segfaults if input file does not exist
* [[#9115](https://github.com/root-project/root/issues/9115)] - Spectator type in TMVACrossValidationApplication
* [[#9793](https://github.com/root-project/root/issues/9793)] - RPATH does not match linked lib when building ROOT with system Python3 on MacOS(11,12)
* [[#9523](https://github.com/root-project/root/issues/9523)] - Documentation of `RooAbsReal::getValues` is broken
* [[#9744](https://github.com/root-project/root/issues/9744)] - GDML module not working with Tessellated solid
* [[#9899](https://github.com/root-project/root/issues/9899)] - TTree incorrectly run I/O customization rules on "new" data members.
* [[#9967](https://github.com/root-project/root/issues/9967)] - Update builtin XRootD to v5.4.1

## HEAD of the v6-26-00-patches branch

These changes will be part of a future 6.26/02.

- None so far.
