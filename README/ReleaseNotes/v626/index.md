% ROOT Version 6.26 Release Notes
% 2021-03-03
<a name="TopOfPage"></a>

## Introduction

ROOT version 6.26/00 is scheduled for release in May, 2021.

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
 Federico Sossai, CERN/SFT,\
 Harshal Shende, GSOC,\
 Matevz Tadel, UCSD/CMS,\
 Vassil Vassilev, Princeton/CMS,\
 Wouter Verkerke, NIKHEF/ATLAS,\
 Zef Wolffs, NIKHEF/ATLAS,\
 Stefan Wunsch, CERN/SFT

## Deprecation, Removal, Backward Incompatibilities

- The "Virtual MonteCarlo" facility VMC (`montecarlo/vmc`) has been removed from ROOT. The development of this package has moved to a [separate project](https://github.com/vmc-project/). ROOT's copy of VMC was deprecated since v6.18.
- `TTreeProcessorMT::SetMaxTasksPerFilePerWorker` has been removed. `TTreeProcessorMT::SetTasksPerWorkerHint` is a superior alternative.
- `TTree::GetEntry()` and `TTree::GetEvent()` no longer have 0 as the default value for the first parameter `entry`. We are not aware of correct uses of this function without providing an entry number. If you have one, please simply pass `0` from now on.
- `TBufferMerger` is now out of the `Experimental` namespace (`ROOT::Experimental::TBufferMerger` is deprecated, please use `ROOT::TBufferMerger` instead)
- RooFit container classes marked as deprecated with this release: `RooHashTable`, `RooNameSet`, `RooSetPair`, and `RooList`. These classes are still available in this release, but will be removed in the next one. Please migrate to STL container classes, such as `std::unordered_map`, `std::set`, and `std::vector`.
- The `RooFit::FitOptions(const char*)` command to steer [RooAbsPdf::fitTo()](https://root.cern.ch/doc/v628/classRooAbsPdf.html) with an option string in now deprecated and will be removed in ROOT v6.28. Please migrate to the RooCmdArg-based fit configuration. The former character flags map to RooFit command arguments as follows:
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

- As of v6.26, cling diagnostic messages can be redirected to the ROOT error handler. Users may enable/disable this via `TCling::ReportDiagnosticsToErrorHandler()`, e.g.
```cpp
root [1] gInterpreter->ReportDiagnosticsToErrorHandler();
root [2] int f() { return; }
Error in <cling>: ROOT_prompt_2:1:11: non-void function 'f' should return a value [-Wreturn-type]
int f() { return; }
          ^
```
More details at [PR #8737](https://github.com/root-project/root/pull/8737).
- Continuation of input lines using backslash `\` is supported in ROOT's prompt, e.g.
```cpp
root [0] std::cout \
root (cont'ed, cancel with .@) [1]<< "ROOT\n";
```

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

### Notable changes in behavior

- Using `Alias`, it is now possible to register homonymous aliases (alternative column names) in different branches of the computation graph, in line with the behavior of `Define` (until now, aliases were required to be unique in the whole computaton graph).
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
  histograms. The v field described here was not working the same way. They are now implemente
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

### Experimental CUDA support for RooFit's `BatchMode`

RooFit's [`BatchMode`](https://root.cern/doc/master/classRooAbsPdf.html#a8f802a3a93467d5b7b089e3ccaec0fa8) has been around
[since ROOT 6.20](https://root.cern/doc/v620/release-notes.html#fast-function-evaluation-and-vectorisation).
It was further [improved in ROOT 6.24](https://root.cern/doc/v624/release-notes.html#massive-speed-up-of-roofits-batchmode-on-cpus-with-vector-extensions) to use vector extensions of modern CPUs without recompiling ROOT, introducing the new `RooBatchCompute` library as a backend that is compiled multiple times for different instruction sets.
With this release, `RooBatchCompute` is also compiled with the Nvidia CUDA compiler to support the computation on GPUs if supported by the RooFit object.
You can use the CUDA mode by passing `"cuda"` to the `BatchMode()` command argument:
```C++
model.fitTo(data);                            // not using the batch mode
model.fitTo(data, RooFit::BatchMode(true));   // using the BatchMode on CPU (RooFit::BatchMode("cpu") is equivalent)
model.fitTo(data, RooFit::BatchMode("cuda")); // using the new CUDA backend
```

The `RooBatchCompute` backend now also supports ROOT's implicit multithreading (similar to RDataFrame), which can be enabled as follows:
```C++
ROOT::EnableImplicitMT(nThreads);
```

For more information, please have a look at this [contribution to the ACAT 2021 conference](https://indico.cern.ch/event/855454/contributions/4596763/) or consult the [RooBatchComupte README](https://github.com/root-project/root/tree/v6-26-00-patches/roofit/batchcompute).
The README also describes how to enable BatchMode support for your own PDFs.

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
                    "Title of dataset",  // Title  (                   ~ " ~                      )
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

## 3D Graphics Libraries


## Geometry Libraries

- Prevent the TColor palette being silently set by TGeoPainter.

## Database Libraries


## Networking Libraries


## GUI Libraries

- On Mac, with Cocoa, the pattern selector did not work anymore and the fit panel range did not work.

- Fix in Cocoa. XSGui crashed on Mac M1.

## WebGUI Libraries

- provide `--web=server` mode, which only printout window URLs instead of starting real web browser.
  Dedicated for the case when ROOT should be running as server application, providing different RWebWindow instances for connection.


## Montecarlo Libraries


## PROOF Libraries


## Language Bindings


## JavaScript ROOT


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

- For users building from source the `latest-stable` branch and passing `-Droottest=ON` to the CMake command line, the corresponding revision of roottest pointed to by `latest-stable` will be downloaded as required.

## PyROOT

- The `ROOT` Python module is now properly serializable so that it is automatically available in the Python environment if a function or ROOT object needs to be serialized. See issue [#6764](https://github.com/root-project/root/issues/6764) for a concrete usecase.
- Improve overload resolution of functions that accept classes with long inheritance trees. Now prefer to call the function overload of the most derived class type (PR [#9092](https://github.com/root-project/root/pull/9092)).
