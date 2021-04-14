% ROOT Version 6.24 Release Notes
% 2020-05-19
<a name="TopOfPage"></a>

## Introduction

ROOT version 6.24/00 is scheduled for release in November 2020.

For more information, see:

[http://root.cern](http://root.cern)

The following people have contributed to this new version:

 Guilherme Amadio, CERN/SFT,\
 Bertrand Bellenot, CERN/SFT,\
 Josh Bendavid, CERN/CMS,\
 Jakob Blomer, CERN/SFT,\
 Rene Brun, CERN/SFT,\
 Philippe Canal, FNAL,\
 Olivier Couet, CERN/SFT,\
 Anirudh Dagar, CERN-SFT/GSOC, \
 Hans Dembinski, TU DOrtmund/LHCb,\
 Massimiliano Galli, CERN/SFT,\
 Andrei Gheata, CERN/SFT,\
 Hadrien Grasland, IJCLab/LAL,\
 Enrico Guiraud, CERN/SFT,\
 Claire Guyot, CERN/SFT,\
 Jonas Hahnfeld, CERN/SFT,\
 Emmanouil Michalainas, CERN/SFT,\
 Stephan Hageboeck, CERN/SFT,\
 Sergey Linev, GSI,\
 Javier Lopez-Gomez, CERN/SFT,\
 Pere Mato, CERN/SFT,\
 Lorenzo Moneta, CERN/SFT,\
 Alja Mrak-Tadel, UCSD/CMS,\
 Axel Naumann, CERN/SFT,\
 Vincenzo Eduardo Padulano, CERN/SFT and UPV,\
 Danilo Piparo, CERN/SFT,\
 Fons Rademakers, CERN/SFT,\
 Jonas Rembser, CERN/SFT,\
 Andrea Sciandra, SCIPP-UCSC/Atlas, \
 Oksana Shadura, UNL/CMS,\
 Enric Tejedor Saavedra, CERN/SFT,\
 Christian Tacke, GSI, \
 Matevz Tadel, UCSD/CMS,\
 Vassil Vassilev, Princeton/CMS,\
 Wouter Verkerke, NIKHEF/Atlas,\
 Stefan Wunsch, CERN/SFT

## General

### Deprecation and Removal

- [`RooAbsReal::evaluateBatch()`](https://root.cern/doc/v624/classRooAbsReal.html#a261580dfe94f2b107f9b9a77cad78a62) has been removed in favour of the faster evaluateSpan(). See section "RooFit Libraries" for instructions on how to use [`RooAbsReal::evaluateSpan()`](https://root.cern/doc/v624/classRooAbsReal.html#a1e5129ffbc63bfd04c01511fd354b1b8).
- `TTreeProcessorMT::SetMaxTasksPerFilePerWorker` has been deprecated in favour of `TTreeProcessorMT::SetTasksPerWorkerHint`.

### Header Dependency Reduction

As always, ROOT tries to reduce the amount of code exposed through its headers.
To that end, `#include`s were replaced by forward declarations in several headers.
This might cause compilation errors ("missing definition of type...") in your code, if that code was relying on indirect includes, instead of including the required headers itself. Please correct that simply by including the required header directly.

## Core Libraries

Due to internal changes required to comply with the deprecation of Intel TBB's `task_scheduler_init` and related
interfaces in recent TBB versions, as of v6.24 ROOT will not honor a maximum concurrency level set with
`tbb::task_scheduler_init` but will require instead the usage of `tbb::global_control`:

```cpp
  //tbb::task_scheduler_init init(2); // does not affect the number of threads ROOT will use anymore

  tbb::global_control c(tbb::global_control::max_allowed_parallelism, 2);
  ROOT::TThreadExecutor p1;  // will use 2 threads
  ROOT::TThreadExecutor p2(/*nThreads=*/8); // will still use 2 threads
```

Note that the preferred way to steer ROOT's concurrency level is still through
[`ROOT::EnableImplicitMT`](https://root.cern/doc/master/namespaceROOT.html#a06f2b8b216b615e5abbc872c9feff40f)
or by passing the appropriate parameter to executors' constructors, as in
[`TThreadExecutor::TThreadExecutor`](https://root.cern/doc/master/classROOT_1_1TThreadExecutor.html#ac7783d52c56cc7875d3954cf212247bb).

See the discussion at [ROOT-11014](https://sft.its.cern.ch/jira/browse/ROOT-11014) for more context.

### Dynamic Path: `ROOT_LIBRARY_PATH`

A new way to set ROOT's "Dynamic Path" was added: the
environment variable `ROOT_LIBRARY_PATH`.  On Unix it should contain a colon
separated list of paths, on Windows a semicolon separated list. It is
intended to be cross platform and to be specific to ROOT (and thus not
interfere with the system's shared linker).
The final "Dynamic Path" is now composed of these sources in order:

1. `ROOT_LIBRARY_PATH` environment variable
2. System specific shared linker environment variables like
   `LD_LIBRARY_PATH`, `LIBPATH`, or `PATH`.
3. Setting from rootrc
4. ROOT's builtin library directory

### Interpreter

- cling's LLVM is upgraded to version 9.0
- New interface to enable/disable optional cling features. Currently, it can be used to enable/disable support for redefinitions. See [this](https://github.com/root-project/cling/issues/360) issue for more information.

### Multithreading

- Fix an uninitialized variable in global read-write lock which could have caused deadlocks or crashes in some rare cases.
- Default global read-write lock transitioned to new implementation based on TBB thread local storage when TBB is available on supported platforms (all except Windows).  This gives an O(10%) performance improvement for some typical RDataFrame scenarios with 256 threads due to reduced lock contention.

## I/O Libraries

- Exclusive use of the global lock is reduced or migrated to finer grained read and write locks in a few hotspots that occur during file opening/closing or task initialization in RDataFrame.  This can lead to O(100x) improvements for some typical RDataFrame scenarios with 256 threads due to massively reduced lock contention.

## TTree Libraries

- `TTree` now supports the inclusion of leaves of types `long` and `unsigned long` (and therefore also `std::size_t` on most systems) also for branches in "leaflist mode". The corresponding leaflist letters are 'G' and 'g'.
- when looping over a `TTree` with a friend with a larger number of entries, `TTreeReader` now ends the event loop when the entries in the _main_ `TTree` are exhausted, consistently with other interfaces. See [#6518](https://github.com/root-project/root/issues/6518) for more details.
- `TTreeProcessorMT::SetMaxTasksPerFilePerWorker` is now deprecated in favor of the more flexible and newly introduced `TTreeProcessorMT::SetTasksPerWorkerHint`. See the relevant entries in our reference guide for more information.
- The name of the sub-branches of a split collection no longer have 2 consecutive dots if the top level branche name has a trailing dot.  The name of the collection's index leaf also no longer include the dot. For example for "t." the names where "t._" and "t..fValue" and are now "t_" and "t.fValue".

## RDataFrame

### New features

- Introduce `ROOT::RDF::RunGraphs`, which allows to compute the results of multiple `RDataFrame`s (or better, multiple independent computation graphs) concurrently while sharing the same thread pool. The computation may be more efficient than running the `RDataFrame`s sequentially if an analysis consists of several computation graphs that individually do not fully utilize the available resources. See e.g. [this tutorial](https://root.cern/doc/master/df104__HiggsToTwoPhotons_8py.html) for an example usage.
- `RDataFrame` now supports reading friend `TTree`s with a `TTreeIndex`, aka "indexed friends". More details at [ROOT-9559](https://sft.its.cern.ch/jira/browse/ROOT-9559).
- Experimental logging capabilities have been added to `RDataFrame`. To activate logging, define the following variable before creating the `RDataFrame` object: `auto verbosity = ROOT::Experimental::RLogScopedVerbosity(ROOT::Detail::RDF::RDFLogChannel(), ROOT::Experimental::ELogLevel.kInfo);`.
- With [ROOT-10023](https://sft.its.cern.ch/jira/browse/ROOT-10023) fixed, `RDataFrame` can now read and write certain branches containing unsplit objects, i.e. `TBranchObjects`. More information is available at [ROOT-10022](https://sft.its.cern.ch/jira/browse/ROOT-10022).
- CSV files can now be opened and processed from remote HTTP(S) locations
- `RDataFrame` results produced by the same action in different computation graphs can now be merged thanks to the new interface provided by `ROOT::Detail::RDF::RMergeableValue`, introduced in [#5552](https://github.com/root-project/root/pull/5552). A feature originally requested with [ROOT-9869](https://sft.its.cern.ch/jira/browse/ROOT-9869), it helps streamline `RDataFrame` workflows in a distributed environment. Currently only a subset of `RDataFrame` actions have their corresponding mergeable class, but in the future it will be possible to extend it to any action through the creation of a new `RMergeableValue` derived class.

### Behavior changes

- `Snapshot` now respects the basket size and split level of the original branch when copying branches to a new `TTree`.
- `Snapshot` now writes branches coming from friend `TTree`s even if they have the same name as branches in the main tree (`friendname_` is prepended to the name of the output branches). More details at [#7181](https://github.com/root-project/root/issues/7181).
- Just-in-time compilation of string expressions passed to `Filter` and `Define` now generates functions that take fundamental types by const value (rather than by non-const reference as before). This will break code that was assigning to column values in string expressions: this is an intended side effect as we want to prevent non-expert users from performing assignments (`=`) rather than comparisons (`==`). Expert users can resort to compiled callables if they absolutely have to assign to column values (not recommended). See [ROOT-11009](https://sft.its.cern.ch/jira/browse/ROOT-11009) for further discussion.
- For some `TTrees`, `RDataFrame::GetColumnNames` might now returns multiple valid spellings for a given column. For example, leaf `"l"` under branch `"b"` might now be mentioned as `"l"` as well as `"b.l"`, while only one of the two spellings might have been recognized before.
- Certain RDF-related types in the `ROOT::Detail` and `ROOT::Internal` namespaces have been renamed, most notably `RCustomColumn` is now `RDefine`. This does not impact code that only makes use of entities in the public ROOT namespace, and should not impact downstream code unless it was patching or reusing internal `RDataFrame` types.

### Notable bug fixes and improvements

- A critical issue has been fixed that could potentially result in wrong data being silently read in multi-thread runs when an input `TChain` contained more than one `TTree` coming from the _same_ input file. More details are available at [#7143](https://github.com/root-project/root/issues/7143).
- The start-up time of event loops with large computation graphs with many just-in-time-compiled expressions (e.g. thousands of string `Filter`s and `Define`s) has been greatly reduced. See [the corresponding pull request](https://github.com/root-project/root/pull/7651) for more details.

The full list of bug fixes for this release is available below.

### Distributed computing with RDataFrame
ROOT 6.24 introduces `ROOT.RDF.Experimental.Distributed`, an experimental python package that enhances RDataFrame with distributed computing capabilities. The new package allows distributing RDataFrame applications through one of the supported distributed backends. The package was designed so that different backends can be easily plugged in. Currently the [Apache Spark](http://spark.apache.org/) backend is supported and support for [Dask](https://dask.org/) is coming soon. The backend submodules of this package expose their own `RDataFrame` objects. The only needed change in user code is to substitute `ROOT.RDataFrame` calls with such backend-specific `RDataFrame`s. For example:

```python
import ROOT

# Point RDataFrame calls to the Spark specific RDataFrame
RDataFrame = ROOT.RDF.Experimental.Distributed.Spark.RDataFrame

# It still accepts the same constructor arguments as traditional RDataFrame
df = RDataFrame("mytree","myfile.root")

# Continue the application with the traditional RDataFrame API
```

The main goal of this package is to support running any RDataFrame application distributedly. Nonetheless, not all RDataFrame operations currently work with this package. The subset that is currently available is:

- AsNumpy
- Count
- Define
- Fill
- Filter
- Graph
- Histo[1,2,3]D
- Max
- Mean
- Min
- Profile[1,2,3]D
- Snapshot
- Sum

with support for more operations coming in the future.

Any distributed RDataFrame backend inherits the dependencies of the underlying software needed to distribute the applications. The Spark backend for example has the following runtime dependencies (ROOT will build just fine without, but the feature will be unavailable without these packages):

- [pyspark](https://spark.apache.org/docs/latest/api/python/index.html), that in turn has its own set of dependencies:
- [Java](https://www.java.com/en/)
- [py4j](https://www.py4j.org/)

Tests for the Spark backend can be turned ON/OFF with the new build option `test_distrdf_pyspark` (OFF by default).

## Histogram Libraries

- Add a new `THistRange` class for defining a generic bin range and iteration in a 1d and multi-dimensional histogram
- Fix a memory leak in `TF1::Copy` and `TFormula::Copy`
- Enable using automatic differentiation when computing parameter gradient in formula based TF1
- Add several fixes and improvements to the `TKDE` class using kernel estimation for estimating a density from data.
- Improve `TF1::GetRandom`, `TH1::GetRandom` and `TH1::FillRandom`  (and same for TF2,TF3, TH2 and TH3 functions) to pass optionally a random number generator instance. This allows to use these
function with a user provided random number generator instead of using the default `gRandom`.  

## Math Libraries

- Update the definitions of the physical constants using the recommended 2018 values from NIST.
- Use also the new SI definition of base units from 2019, where the Planck constant, the Boltzmann constant, the elementary electric charge and the Avogadro constant are exact numerical values. See <https://en.wikipedia.org/wiki/2019_redefinition_of_the_SI_base_units>. Note that with this new definition the functions `TMath::HUncertainty()`, `TMath::KUncertainty()`, `TMath::QeUncertainty()` and `TMath::NaUncertainty()` all return a  `0.0` value.
- Due to some planned major improvements to `RVec`, the layout of `RVec` objects will change in a backward-incompatible way between v6.24 and v6.26.
  Because of this, we now print a warning if an application is reading or writing a `ROOT::RVec` object from/to a ROOT file. We assume this is an
  exceedingly rare case, as the ROOT interface typically used to manipulate `RVec`s is `RDataFrame`, and `RDataFrame` performs an on-the-fly
  `RVec <-> std::vector` conversion rather than writing `RVec`s to disk. Note that, currently, `RVecs` written e. Enable autig. in a `TTree` cannot be read back
  using certain ROOT interfaces (e.g. `TTreeReaderArray`, `RDataFrame` and the experimental `RNTuple`). All these limitations will be lifted in v6.26.
- Portable implementation of the RANLUX++ generator, see [RanluxppEngine](https://root.cern/doc/master/classROOT_1_1Math_1_1RanluxppEngine.html) and [our blog post](https://root.cern/blog/ranluxpp/).
- Change `TRandom3::GetSeed` to return the current state element in the contained seed vector of TRandom3. The return value will now change after every call of `TRandom3::Rndm` (when generating a random number). Before the function was returning the first element of the state, which was changing only after 624 calls to `Rndm()`.
- Fix a bug in `ROOT::Fit::BinData` copy constructor
- Fix a bug in applying a correction factor used for the computation of the fit confidence level in `ROOT::Fit::FitResult`.
- TMatrix: optimize implementation of `TPrincipal::AddRow` that is heavily used by CMS.

### Minuit2

- Add a new improved message logging system. Debug message now can be enabled in Minuit2 when using maximum print level.
- When using external provided gradient, compute in MnSeed still numerical gradients to obtain correct step sizes and  initial estimate of covariance matrix. This allows to start with a good first
	state estimation, reducing significantly the number of  subsequent iterations.


## TMVA

- Introducing TMVA PyTorch Interface, a method to use PyTorch internally with TMVA for deep learning. This can be used as an alternative to PyKeras Interface for complex models providing more
flexibility and power.
- Add support in the TMVA Keras interface for Tensorflow.Keras (the version embedded in Tensorflow) and for standalone Keras versions up to it latest 2.3. For using  Tensorflow.Keras one needs to use
the booking option `tf.keras=True`.
- Update the TMVA Keras tutorials to use now tensorflow.keras.
- Deprecate the MethodDNN in favour of MethodDL supporting both CNN and RNN
- Add possibility to customize all relevant minimizer parameters used for training in MethodDL
- Add support in MethodDL for the Cudnn version 8 when using the Cuda implementation for CNN and RNN Minuit2
- Implement the missing support for MethodCategory for multiclass classifiers.
- Add possibility to retrieve a ROC curve made with the training dataset instead of the default test dataset.


## RooFit Libraries

- Extension / updates of the doxygen reference guide.
- Allow for removing RooPlot from global directory management, see [RooPlot::AddDirectory](https://root.cern/doc/v624/classRooPlot.html#a47f7ba71dcaca30ad9ee295dee89c9b8)
  and [RooPlot::SetDirectory](https://root.cern/doc/v624/classRooPlot.html#a5938bc6d5c47d94c2f04fdcc10c1c026)
- Hash-assisted finding of elements in RooWorkspace. Large RooWorkspace objects were slow in finding elements.
  This was improved using a hash map.
- Stabilise RooStats::HypoTestInverter. It can now tolerate a few failed fits when conducting hypothesis tests.
  This is relevant when a few points in a parameter scan don't converge due to numerical or model instabilities.
  These points will be skipped, and HypoTestInverter can continue.
- Tweak pull / residual plots. ROOT automatically zoomed out a bit when a pull / residual plot is created. Now, the
  axis range of the original plot is transferred to the residual plot, so the pulls can be drawn below the main plot.
- Improve plotting of `RooBinSamplingPdf`
- Print a Warning message when the `RooAddPdf` is evaluated without passing a normalization set and the class has not a normalization set defined.
Without a normalization set the `RooAddPdf` is not properly defined and its shape will be different depending on which normalization range is used.


### Massive speed up of RooFit's `BatchMode` on CPUs with vector extensions

RooFit's [`BatchMode`](https://root.cern/doc/master/classRooAbsPdf.html#a8f802a3a93467d5b7b089e3ccaec0fa8) has been around
[since ROOT 6.20](https://root.cern/doc/v620/release-notes.html#fast-function-evaluation-and-vectorisation), but to
fully use vector extensions of modern CPUs, a manual compilation of ROOT was necessary, setting the required compiler flags.

Now, RooFit comes with dedicated computation libraries, each compiled for a specific CPU architecture. When RooFit is loaded for the
first time, ROOT inspects the CPU capabilities, and loads the fastest supported version of this computation library.
This means that RooFit can now use vector extensions such as AVX2 without being recompiled, which enables a speed up of up to 4x for certain computations.
Combined with better data access patterns (~3x speed up, ROOT 6.20), computations with optimised PDFs speed up between 4x and 16x.

The fast `BatchMode` now also works in combination with multi processing (`NumCPU`) and with binned data (`RooDataHist`).

See [Demo notebook in SWAN](https://github.com/hageboeck/rootNotebooks),
[EPJ Web Conf. 245 (2020) 06007](https://www.epj-conferences.org/articles/epjconf/abs/2020/21/epjconf_chep2020_06007/epjconf_chep2020_06007.html),
[arxiv:2012.02746](https://arxiv.org/abs/2012.02746).

#### RooBatchCompute Library

The library that contains the optimised computation functions is called `RooBatchCompute`. The PDFs contained in this library are highly optimized, and there is currently work in progress for further optimization using CUDA and multi-threaded computations. If you use PDFs that are not part of the official RooFit, you are very well invited to add them to RooFit by [submitting a ticket](https://github.com/root-project/root/issues/new) or a [pull request](https://github.com/root-project/root/pulls).

#### Benefiting from batch computations by overriding `evaluateSpan()`

For PDFs that are not part of RooFit, it is possible to benefit from batch computations without vector extensions. To do so, consult the [RooBatchCompute readme](https://github.com/root-project/root/tree/v6-24-00-patches/roofit/batchcompute).


#### Migrating PDFs that override the deprecated `evaluateBatch()`

In case you have created a custom PDF which overrides `evaluateBatch()`, please follow these steps to update your code to the newest version:

1. Change the signature of the function both in the source and header file:
```diff
- RooSpan<double> RooGaussian::evaluateBatch(std::size_t begin, std::size_t batchSize) const
+ RooSpan<double> evaluateSpan(RooBatchCompute::RunContext& evalData, const RooArgSet* normSet) const
```
2. Include `RunContext.h` and `BracketAdapter.h`.
3. Use `getValues()` instead of `getValBatch()` to retrieve a RooSpan for the data of every value.
```diff
- auto xData = x.getValBatch(begin, batchSize);
+ auto xData = x->getValues(evalData,normSet);
```
4. Retrieve the number of events by getting the maximum size of the input spans.
```c++
  size_t nEvents=0;
  for (auto& i:{xData,meanData,sigmaData})
    nEvents = std::max(nEvents,i.size());
```
5. Create the output batch by calling `RunContext::makeBatch()`
```diff
- auto output = _batchData.makeWritableBatchUnInit(begin, batchSize);
+ auto output = evalData.makeBatch(this, nEvents);
```
6. **DO NOT use `RooSpan::isBatch()` and `RooSpan::empty()` methods!** Instead, distinguish between scalar (RooSpan of size 1) and vector (RooSpan of size>1) parameters as shown below.
```diff
- const bool batchX = !xData.empty();
+ const bool batchX = xData.size()>1;
```
7. Append `RooBatchCompute::` to the classes that have been moved to the RooBatchCompute Library: `RooSpan`,`BracketAdapterWithMask`, `BracketAdapter`, `RunContext`. Alternatively, you can write
```c++
using namespace RooBatchCompute;
```
8. Replace `_rf_fast_<function>` with `RooBatchCompute::fast_<function>` and include `RooVDTHeaders.h` (if applicable).
```diff
- output[i] = _rf_fast_exp(arg*arg * halfBySigmaSq);
+ output[i] = RooBatchCompute::fast_exp(arg*arg * halfBySigmaSq);
```

### Unbiased binned fits

When RooFit performs binned fits, it takes the probability density at the bin centre as a proxy for the probability in the bin. This can lead to a bias.
To alleviate this, the new class [RooBinSamplingPdf](https://root.cern/doc/v624/classRooBinSamplingPdf.html) has been added to RooFit.
Also see [arxiv:2012.02746](https://arxiv.org/abs/2012.02746).

### More accurate residual and pull distributions

When making residual or pull distributions with `RooPlot::residHist` or `RooPlot::pullHist`, the histogram is now compared with the curve's average values within a given bin by default, ensuring that residual and pull distributions are valid for strongly curved distributions.
The old default behaviour was to interpolate the curve at the bin centres, which can still be enabled by setting the `useAverage` parameter of `RooPlot::residHist` or `RooPlot::pullHist` to `false`.

### Improved recovery from invalid parameters

When a function in RooFit is undefined (Poisson with negative mean, PDF with negative values, etc), RooFit can now pass information about the
"badness" of the violation to the minimiser. The minimiser can use this to compute a gradient to find its way out of the undefined region.
This can drastically improve its ability to recover when unstable fit models are used, for example RooPolynomial.

For details, see the RooFit tutorial [rf612_recoverFromInvalidParameters.C](https://root.cern/doc/v624/rf612__recoverFromInvalidParameters_8C.html)
and [arxiv:2012.02746](https://arxiv.org/abs/2012.02746).

### Modernised RooDataHist

RooDataHist was partially modernised to improve const-correctness, to reduce side effects as well as its memory footprint, and to make
it ready for RooFit's faster batch evaluations.
Derived classes that directly access protected members might need to be updated. This holds especially for direct accesses to `_curWeight`,
`_curWeightErrLo`, etc, which have been removed. (It doesn't make sense to write to these members from const functions when the same information
can be retrieved using an index access operator of an array.) All similar accesses in derived classes should be replaced by the getters `get_curWeight()`
or better `get_wgt(i)`, which were also supported in ROOT \<v6.24. More details on what happened:

- Reduced side effects. This code produces undefined behaviour because the side effect of `get(i)`, i.e., loading the new weight into `_curWeight`
  is not guaranteed to happen before `weight()` is called:
```
  processEvent(dataHist.get(i), dataHist.weight()); // Dangerous! Order of evaluation is not guaranteed.
```
  With the modernised interface, one would use:
```
  processEvent(dataHist.get(i), dataHist.weight(i));
```
  To modernise old code, one should replace patterns like `h.get(i); h.func()` by `h.func(i);`. One may `#define R__SUGGEST_NEW_INTERFACE` to switch on
  deprecation warnings for the functions in question.
  Similarly, the bin content can now be set using an index, making prior loading of a certain coordinate unnecessary:
```diff
   for (int i=0 ; i<hist->numEntries() ; i++) {
-    hist->get(i) ;
-    hist->set(hist->weight() / sum);
+    hist->set(i, hist->weight(i) / sum, 0.);
   }
```
- More const correctness. `calcTreeIndex()` doesn't rely on side effects, any more. Instead of overwriting the internal
  coordinates with new values:
```
  // In a RooDataHist subclass:
  _vars = externalCoordinates;
  auto index = calcTreeIndex();

  // Or from the outside:
  auto index = dataHist.getIndex(externalCoordinates); // Side effect: Active bin is now `index`.
```
  coordinates are now passed into calcTreeIndex without side effects:
```
  // In a subclass:
  auto index = calcTreeIndex(externalCoordinates, fast=<true/false>); // No side effect

  // From the outside:
  auto index = dataHist.getIndex(externalCoordinates); // No side effect
```
  This will allow for marking more functions const, or for lying less about const correctness.

- RooDataHist now supports fits with RooFit's faster `BatchMode()`.
- Lower memory footprint. If weight errors are not needed, RooDataHist now allocates only 40% of the memory that the old implementation used.

#### Fix bin volume correction logic in `RooDataHist::sum()`

The public member function `RooDataHist::sum()` has three overloads.
Two of these overloads accept a `sumSet` parameter to not sum over all variables.
These two overloads previously behaved inconsistently when the `correctForBinSize` or `inverseBinCor` flags were set.
If you use the `RooDataHist::sum()` function in you own classes, please check that it can still be used with its new logic.
The new and corrected bin correction behaviour is:

  - `correctForBinSize`: multiply counts in each bin by the bin volume corresponding to the variables in `sumSet`
  - `inverseBinCor`: divide counts in each bin by the bin volume corresponding to the variables *not* in `sumSet`

### New fully parametrised Crystal Ball shape class

So far, the Crystal Ball distribution has been represented in RooFit only by the `RooCBShape` class, which has a Gaussian core and a single power-law tail on one side.
This release introduces [`RooCrystalBall`](https://root.cern/doc/v624/classRooCrystalBall.html), which implements some common generalizations of the Crystal Ball shape:

  - symmetric or asymmetric power-law tails on both sides
  - different width parameters for the left and right sides of the Gaussian core

The new `RooCrystalBall` class can substitute the `RooDSCBShape` and `RooSDSCBShape`, which were passed around in the community.

## 2D Graphics Libraries

- Add the method `AddPoint`to `TGraph(x,y)` and `TGraph2D(x,y,z)`, equivalent to `SetPoint(g->GetN(),x,y)`and `SetPoint(g->GetN(),x,y,z)`
- Option `E0` draws error bars and markers are drawn for bins with 0 contents. Now, combined
with options E1 and E2, it avoids error bars clipping.
- Fix `TAxis::ChangeLabel` for vertical axes and 3D plots

## 3D Graphics Libraries


## Geometry Libraries


## Database Libraries


## Networking Libraries

### Multithreaded support for FastCGI

Now when THttpServer creates FastCGI engine, 10 worker threads used to process requests
received via FastCGI channel. This significantly increase a performance, especially when
several clients are connected.

### Better security for THttpServer with webgui

If THttpServer created for use with webgui widgets (RBrowser, RCanvas, REve), it only will
provide access to the widgets via websocket connection - any other kind of requests like root.json
or exe.json will be refused completely. Combined with connection tokens and https protocol,
this makes usage of webgui components in public networks more secure.

### Enabled WLCG Bearer Tokens support in RDavix

Bearer tokens are part of WLCG capability-based infrastructure with capability-based scheme which uses an infrastructure that describes what the bearer is allowed to do as opposed to who that bearer is. Token discovery procedure are developed according WLCG Bearer Token Discovery specification document (https://github.com/WLCG-AuthZ-WG/bearer-token-discovery/blob/master/specification.md). Short overview:

   1. If the `BEARER_TOKEN` environment variable is set, then the value is taken to be the token contents.
   2. If the `BEARER_TOKEN_FILE` environment variable is set, then its value is interpreted as a filename. The contents of the specified file are taken to be the token contents.
   3. If the `XDG_RUNTIME_DIR` environment variable is set, then take the token from the contents of `$XDG_RUNTIME_DIR/bt_u$ID`(this additional location is intended to provide improved security for shared login environments as `$XDG_RUNTIME_DIR` is defined to be user-specific as opposed to a system-wide directory.).
   4. Otherwise, take the token from `/tmp/bt_u$ID`.

### Xrootd client support

ROOT can now be built with Xrootd 5 client libraries.

## GUI Libraries

### RBrowser improvements

- central factory methods to handle browsing, editing and drawing of different classes
- simple possibility to extend RBrowser on user-defined classes
- support of web-based geometry viewer
- better support of TTree drawing
- server-side handling of code editor and image viewer widgets
- rbrowser content is fully recovered when web-browser is reloaded
- load of widgets code only when really required (shorter startup time for RBrowser)


## Montecarlo Libraries


## PROOF Libraries


## Language Bindings


## JavaScript ROOT

### Major JSROOT update to version 6

- update all used libraries `d3.js`, `three.js`, `MathJax.js`, openui5
- change to Promise based interface for all async methods, remove call-back arguments
- change scripts names, core scripts name now `JSRoot.core.js`
- unify function/methods naming conventions, many changes in method names
- provide central code loader via `JSROOT.require`, supporting 4 different loading engines
- many nice features and many bug fixes; see [JSROOT v6 release notes](https://github.com/root-project/jsroot/blob/master/changes.md#changes-in-600)


## Tutorials


## Class Reference Guide

One can now select a class's documentation for a specific version.
If a class does not exist in a given version, that version is grayed out,
see for instance the documentation for [`ROOT::Experimental::RNTupleReader`](https://root.cern/doc/master/classROOT_1_1Experimental_1_1RNTupleReader.html).

## Build, Configuration and Testing Infrastructure

- a new cmake variable, `CMAKE_INSTALL_PYTHONDIR`, has been added: it allows customization of the installation directory of ROOT's python modules
- the developer build option `asserts` is introduced to enable/disable asserts via the `NDEBUG` C/CXX flag. Asserts are always enabled for `CMAKE_BUILD_TYPE=Debug` and `dev=ON`. The previous behavior of the builds set via the `CMAKE_BUILD_TYPE` variable has not changed.
- `CMAKE_CXX_STANDARD`, i.e. the C++ standard ROOT is built with, now defaults to the compiler default (or C++11 if the compiler default is older than that) rather than always defaulting to C++11. In turn this means that v6.24 is the first ROOT release for which ROOT's pre-compiled binaries are not compiled with C++11 but with the default standard in use by the default system compiler. On Ubuntu 20.04, for example, the v6.24 pre-compiled binaries are now compiled with C++14 rather than C++11 as it happened for previous ROOT versions. Also see [ROOT-10692](https://sft.its.cern.ch/jira/browse/ROOT-10692).

The following builtins have been updated:

- VecCore 0.7.0
- LZ4 1.9.3
- openui5
- Xrootd 4.12.8
- Zstd   1.4.8


## PyROOT

- Deprecate `TTree.AsMatrix` in this release and mark for removal in v6.26. Please use instead `RDataFrame.AsNumpy`.
