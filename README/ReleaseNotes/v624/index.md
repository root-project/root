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
 Jakob Blomer, CERN/SFT,\
 Rene Brun, CERN/SFT,\
 Philippe Canal, FNAL,\
 Olivier Couet, CERN/SFT,\
 Massimiliano Galli, CERN/SFT,\
 Andrei Gheata, CERN/SFT,\
 Hadrien Grasland, IJCLab/LAL,\
 Enrico Guiraud, CERN/SFT,\
 Claire Guyot, CERN/SFT,\
 Stephan Hageboeck, CERN/SFT,\
 Sergey Linev, GSI,\
 Pere Mato, CERN/SFT,\
 Lorenzo Moneta, CERN/SFT,\
 Alja Mrak-Tadel, UCSD/CMS,\
 Axel Naumann, CERN/SFT,\
 Vincenzo Eduardo Padulano, CERN/SFT and UPV,\
 Danilo Piparo, CERN/SFT,\
 Fons Rademakers, CERN/SFT,\
 Andrea Sciandra, SCIPP-UCSC/Atlas, \
 Oksana Shadura, UNL/CMS,\
 Enric Tejedor Saavedra, CERN/SFT,\
 Matevz Tadel, UCSD/CMS,\
 Vassil Vassilev, Princeton/CMS,\
 Wouter Verkerke, NIKHEF/Atlas,\
 Stefan Wunsch, CERN/SFT

## Deprecation and Removal


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

Note that the preferred way to steer ROOT's concurrency level is still through `[ROOT::EnableImplicitMT](https://root.cern/doc/master/namespaceROOT.html#a06f2b8b216b615e5abbc872c9feff40f)` or by passing the appropriate parameter to executors' constructors, as in `[TThreadExecutor::TThreadExecutor](https://root.cern/doc/master/classROOT_1_1TThreadExecutor.html#ac7783d52c56cc7875d3954cf212247bb)`.

See the discussion at [ROOT-11014](https://sft.its.cern.ch/jira/browse/ROOT-11014) for more context.

### Interpreter
- cling's LLVM is upgraded to version 9.0

## I/O Libraries


## TTree Libraries

- TTree now supports the inclusion of leaves of types `long` and `unsigned long` (and therefore also `std::size_t` on most systems) also for branches in "leaflist mode". The corresponding leaflist letters are 'G' and 'g'.

## RDataFrame

- With [ROOT-10023](https://sft.its.cern.ch/jira/browse/ROOT-10023) fixed, RDataFrame can now read and write certain branches containing unsplit objects, i.e. TBranchObjects. More information is available at [ROOT-10022](https://sft.its.cern.ch/jira/browse/ROOT-10022).
- Snapshot now respects the basket size and split level of the original branch when copying branches to a new TTree.
- For some `TTrees`, RDataFrame's `GetColumnNames` method returns multiple valid spellings for a given column. For example, leaf `"l"` under branch `"b"` might now be mentioned as `"l"` as well as `"b.l"`, while only one of the two spellings might have been recognized before.
- Introduce `ROOT::RDF::RunGraphs`, which allows to compute the results of multiple RDataFrames concurrently while sharing the same thread pool. The computation may be more efficient than running the RDataFrames sequentially if an analysis consists of many RDataFrames, which don't have enough data to fully utilize the available resources.
- CSV files can now be opened and processed from HTTP(S) locations
- Certain RDF-related types in the ROOT::Detail and ROOT::Internal namespaces have been renamed, most notably `RCustomColumn` is now `RDefine`. This does not impact code that only makes use of entities in the public ROOT namespace, and should not impact downstream code unless it was patching or reusing internal RDataFrame types.
- Just-in-time compilation of string expressions passed to `Filter` and `Define` now generates functions that take fundamental types by const value (rather than by non-const reference as before). This will break code that was assigning to column values in string expressions: this is an intended side effect as we want to prevent non-expert users from performing assignments (=) rather than comparisons (==). Expert users can resort to compiled callables if they absolutely have to assign to column values (not recommended). See [ROOT-11009](https://sft.its.cern.ch/jira/browse/ROOT-11009) for further discussion.
- RDataFrame action results are now automatically mergeable thanks to the new interface provided by `ROOT::Detail::RDF::RMergeableValue` and derived, introduced in [#5552](https://github.com/root-project/root/pull/5552). A feature originally requested with [ROOT-9869](https://sft.its.cern.ch/jira/browse/ROOT-9869), it helps streamline RDataFrame workflow in a distributed environment. Currently only a subset of RDataFrame actions have their corresponding mergeable class, but in the future it will be possible to extend it to any action through the creation of a new `RMergeableValue` derived class.

## Histogram Libraries


## Math Libraries

- Update the definitions of the physical constants using the recommended 2018 values from NIST.
- Use also the new SI definition of base units from 2019, where the Planck constant, the Boltzmann constant, the elementary electric charge and the Avogadro constant are exact numerical values. See <https://en.wikipedia.org/wiki/2019_redefinition_of_the_SI_base_units>. Note that with this new definition the functions `TMath::HUncertainty()`, `TMath::KUncertainty()`, `TMath::QeUncertainty()` and `TMath::NaUncertainty()` all return a  `0.0` value.
- Due to some planned major improvements to `RVec`, the layout of `RVec` objects will change in a backward-incompatible way between v6.24 and v6.26.
  Because of this, we now print a warning if an application is reading or writing a `ROOT::RVec` object from/to a ROOT file. We assume this is an
  exceedingly rare case, as the ROOT interface typically used to manipulate `RVec`s is `RDataFrame`, and `RDataFrame` performs an on-the-fly
  `RVec <-> std::vector` conversion rather than writing `RVec`s to disk. Note that, currently, `RVecs` written e.g. in a `TTree` cannot be read back
  using certain ROOT interfaces (e.g. `TTreeReaderArray`, `RDataFrame` and the experimental `RNTuple`). All these limitations will be lifted in v6.26.


## RooFit Libraries

- Extension / updates of the doxygen reference guide.
- Allow for removing RooPlot from global directory management, see [RooPlot::AddDirectory](https://root.cern/doc/v624/classRooPlot.html#a47f7ba71dcaca30ad9ee295dee89c9b8)
  and [RooPlot::SetDirectory](https://root.cern/doc/v624/classRooPlot.html#a5938bc6d5c47d94c2f04fdcc10c1c026)
- Hash-assisted finding of elements in RooWorkspace. Large RooWorkspace objects were slow in finding elements.
  This was improved using a hash map.
- Stabilise RooStats::HypoTestInverter. It can now tolerate a few failed fits when conducting hypothesis tests.
  This is relevant when a few points in a parameter scan don't converge due to numerical or model instabilities.
  These points will be skipped, and HypoTestInverter can continue.
- New PDFs
  - [RooDSCBShape](https://root.cern/doc/v624/classRooDSCBShape.html)
  - [RooSDSCBShape](https://root.cern/doc/v624/classRooSDSCBShape.html)
- Tweak pull / residual plots. ROOT automatically zoomed out a bit when a pull / residual plot is created. Now, the
  axis range of the original plot is transferred to the residual plot, so the pulls can be drawn below the main plot.


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
```
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

### Fix bin volume correction logic in `RooDataHist::sum()`

The public member function `RooDataHist::sum()` has three overloads.
Two of these overloads accept a `sumSet` parameter to not sum over all variables.
These two overloads previously behaved inconsistently when the `correctForBinSize` or `inverseBinCor` flags were set.
If you use the `RooDataHist::sum()` function in you own classes, please check that it can still be used with its new logic.
The new and corrected bin correction behaviour is:
  - `correctForBinSize`: multiply counts in each bin by the bin volume corresponding to the variables in `sumSet`
  - `inverseBinCor`: divide counts in each bin by the bin volume corresponding to the variables *not* in `sumSet`
  - 
### New fully parametrised Crystal Ball shape class

So far, the Crystal Ball distribution has been represented in RooFit only by the `RooCBShape` class, which has a Gaussian core and a single power-law tail on one side.
This release introduces the `RooCrystalBall` class, which implements some commom generalizations of the Crystal Ball shape:
  - symmetric or asymmetric power-law tails on both sides
  - different width parameters for the left and right sides of the Gaussian core

The new `RooCrystalBall` class is meant to substitute the `RooDSCBShape` and `RooSDSCBShape` classes that were never part of RooFit but passed around in its user community.

## 2D Graphics Libraries

- Add the method `AddPoint`to `TGraph(x,y)` and `TGraph2D(x,y,z)`, equivalent to `SetPoint(g->GetN(),x,y)`and `SetPoint(g->GetN(),x,y,z)`
- Option `E0` draws error bars and markers are drawn for bins with 0 contents. Now, combined
  with options E1 and E2, it avoids error bars clipping.

## 3D Graphics Libraries


## Geometry Libraries


## Database Libraries


## Networking Libraries

### Multithreaded support for FastCGI
Now when THttpServer creates FastCGI engine, 10 worker threds used to process requests
received via FastCGI channel. This significantly increase a performance, especially when
several clients are connected.

### Better security for THttpServer with webgui
If THttpServer created for use with webgui widgets (RBrowser, RCanvas, REve), it only will
provide access to the widgets via websocket connection - any other kind of requests like root.json
or exe.json will be refused completely. Cobined with connection tokens and https protocol,
this makes usage of webgui components in public networks more secure.

### Enabled WLCG Bearer Tokens support in RDavix
Bearer tokens are part of WLCG capability-based infrastructure with capability-based scheme which uses an infrastructure that describes what the bearer is allowed to do as opposed to who that bearer is. Token discovery procedure are developed according WLCG Bearer Token Discovery specification document (https://github.com/WLCG-AuthZ-WG/bearer-token-discovery/blob/master/specification.md). Short overview:
   1. If the `BEARER_TOKEN` environment variable is set, then the value is taken to be the token contents.
   2. If the `BEARER_TOKEN_FILE` environment variable is set, then its value is interpreted as a filename. The contents of the specified file are taken to be the token contents.
   3. If the `XDG_RUNTIME_DIR` environment variable is set, then take the token from the contents of `$XDG_RUNTIME_DIR/bt_u$ID`(this additional location is intended to provide improved security for shared login environments as `$XDG_RUNTIME_DIR` is defined to be user-specific as opposed to a system-wide directory.).
   4. Otherwise, take the token from `/tmp/bt_u$ID`.

## GUI Libraries

### RBrowser improvments
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
- many nice features and many bug fixes; see JSROOT v6 release notes


## Tutorials


## Class Reference Guide


## Build, Configuration and Testing Infrastructure

- a new cmake variable, `CMAKE_INSTALL_PYTHONDIR`, has been added: it allows customization of the installation directory of ROOT's python modules
- The developer build option `asserts` is introduced to enable/disable asserts via the `NDEBUG` C/CXX flag. Asserts are always enabled for `CMAKE_BUILD_TYPE=Debug` and `dev=ON`. The previous behavior of the builds set via the `CMAKE_BUILD_TYPE` variable has not changed.

The following builtins have been updated:

- VecCore 0.7.0

## PyROOT

- Deprecate `TTree.AsMatrix` in this release and mark for removal in v6.26. Please use instead `RDataFrame.AsNumpy`.
