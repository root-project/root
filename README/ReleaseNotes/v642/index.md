% ROOT Version 6.42 Release Notes
% 2026-11-15
<a name="TopOfPage"></a>

## Introduction

ROOT version 6.42.00 is scheduled for release in November 2026.

For more information, see:

[http://root.cern](http://root.cern)

The following people have contributed to this new version:

The following people have contributed to this new version:

 Bertrand Bellenot, CERN/EP-SFT,\
 Jakob Blomer, CERN/EP-SFT,\
 Lukas Breitwieser, CERN/EP-SFT,\
 Philippe Canal, FNAL,\
 Olivier Couet, CERN/EP-SFT,\
 Marta Czurylo, CERN/EP-SFT,\
 Florine de Geus, CERN/EP-SFT and University of Twente,\
 Andrei Gheata, CERN/EP-SFT,\
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
 Silia Taider, CERN/EP-SFT,\
 Devajith Valaparambil Sreeramaswamy, CERN/EP-SFT,\
 Vassil Vassilev, Princeton,\
 Sandro Wenzel, CERN/EP-ALICE,\
 Tristan Wenzel, ETHZ,\

## Deprecation and Removal

* The build options `vc`, `veccore`, `builtin_vc`, `builtin_veccore` and `rpath` that were deprecated are now removed and will result in configuration errors if used.
* The option `fail-on-missing=OFF` is no longer honored for opt-in (ie OFF by default) build options requiring external dependencies such as `arrow`, `cocoa`, `daos`, `daos_mock`, `dcache`, `experimental_adaptivecpp`, `fcgi`, `fortran`, `gviz`, `mpi`, `pythia8`, `qt6web`, `tmva-cudnn`, `tmva-pymva`, `tmva-sofie`, `uring` or `vecgeom`. If the respective associated package dependency is not installed, ROOT will always raise a configuration error independent of the value of `fail-on-missing`. The user has to take action by either providing the dependency or manually disabling that option via `-Darrow=OFF`.
  Note that `all=ON` enables several of these options, so building with `-Dall=ON` now requires all of their dependencies to be installed, or the unwanted ones to be disabled explicitly.
  Build options that are enabled by default, such as `pyroot`, `opengl`, `xml`, `sqlite`, `davix`, `curl`, `tmva-cpu` or `tpython` are not affected: they are still disabled automatically when their dependencies are missing.
* The option `fail-on-missing=OFF` will no longer be honored for CMake ROOT build options that have easy-to-install dependencies (e.g. via homebrew or apt-get), such as those required by options `cfitsio`, `civetweb`, `fftw3`, `imt`, `mathmore`, `nlohmann_json`, `tmva-cpu`, `unuran`, `vdt` or `xrootd`. Before, associated `builtin_option` was automatically turned ON (or the opt-in feature turned to OFF), now, user has to install system package or manually set `builtin_option` to `ON` or opt-in feature to `OFF`.
* The legacy evaluation backend of RooFit and the related `RooFit::BatchMode()` command argument are deprecated and will be removed in ROOT 6.44. See the RooFit section below for details.
* The method `RooRealVar::removeRange()` and the corresponding method in `RooErrorVar` that were deprecated in ROOT 6.40 are now removed.
* The overloads of `RooAbsReal::createChi2()` and `RooAbsReal::chi2FitTo()` that take unbinned **RooDataSet** data objects were deprecated in ROOT 6.40 and are now removed.
* The **RooStats::HybridPlot** class and the related **HybridResult::GetPlot** method were deprecated in ROOT 6.40 and are now removed.
* The `builtin_zeromq` and `builtin_cppzmq` build options that were deprecated in ROOT 6.40 are now removed.
* The ROOT **auth** package together with `TVirtualAuth` and `TROOT::GetListOfSecContexts()`, and the **authenticated sockets** (`TSocket::CreateAuthSocket()`) feature are now removed following deprecation in ROOT 6.40.
* The `TSSLSocket` class is now removed following deprecation in ROOT 6.40.
* The bindings to the R programming language that are enabled with the `r=ON` or `tmva-rmva=ON` build options (`TRInterface`, RMVA, and friends) are removed, following deprecation in ROOT 6.40. Their maintenance is no longer justified, given the broader adoption of the scientific Python ecosystem. Users who still rely on R from C++ are encouraged to call R directly via https://cran.r-project.org/package=RInside, which is what the ROOT bindings were using internally.
* Several enums that are redundant with `ROOT::ESTLType` are deprecated and will be removed in ROOT 6.44: `TClassEdit::ESTLType`, `TDictionary::ESTLType`, `TStreamerElement::ESTLType`. Please use `ROOT::ESTLType` instead.
* The inclusion by external projects of Makefile templates contained within ROOT is deprecated in 6.42, a warning will be raised if you use them. These files will be removed in ROOT 7.
* The conversion from Python set to **RooArgSet** is deprecated and won't work anymore in ROOT 6.44. The problem is that Python sets are unordered while RooArgSets are ordered, and this mismatch can lead to subtle problems later on. Prefer conversion from Python lists or tuples, which are ordered too.
* The **TMPIFile** class and the `mpi` build option (not to be confused with `minuit2_mpi`, which is unaffected) are deprecated and will be removed in ROOT 6.44.
* The ROOT IO capability for the `TMVA::Experimental::SOFIE::RModel` has been removed. Users should not be encouraged to serialize models in experimental classes. For the serialization of ONNX models one can already use ONNX directly, and even serialize the ONNX bytes to a ROOT file if required.
* The Keras and PyTorch parsers for SOFIE (`TMVA::Experimental::SOFIE::PyKeras` and `PyTorch`) are now removed, so `RSofieReader` only accepts ONNX files.
These parsers relied on private implementation details of Keras and PyTorch, which change faster than is appropriate for ROOT's stability standards.
Users are encouraged to export their models to ONNX and use the retained ONNX parser instead.
* **PyMVA**, the TMVA interface to Python machine-learning libraries (the `PyKeras`, `PyTorch`, `PyRandomForest`, `PyGTB` and `PyAdaBoost` methods), and the corresponding `tmva-pymva` build option are deprecated and will be removed in ROOT 6.44. Like the SOFIE Keras and PyTorch parsers, PyMVA relies on implementation details of the underlying Python libraries that change faster than is appropriate for ROOT's stability standards. Users are encouraged to train and evaluate their models directly with the Python machine-learning libraries, which integrate well with ROOT via the `ROOT::Experimental::ML::DataLoader`. For high-performance inference in C++, models can be exported to ONNX and evaluated with SOFIE (see `RSofieReader`).
* The ROOT IO capability for the `TMVA::Experimental::RBDT` class has been removed, along with the `TMVA.Experimental.SaveXGBoost` Python function. Experimental classes should not be persistified since their on-disk layout is not guaranteed to be stable. An `RBDT` is now built directly from an XGBoost model in its native JSON serialization with the new `TMVA::Experimental::RBDT::LoadXGBoost(jsonPath)`, which works both from C++ and Python. To convert a trained model, save it first with XGBoost's `Booster.save_model("model.json")` and then load it with `LoadXGBoost`.
* The **JsMVA** feature for interactive TMVA training in Jupyter notebooks is now removed. It was not functional for years and was therefore already excluded from ROOT 6.38. This also removes the `TMVA::IPythonInteractive` class and the related interactive-training interfaces from the TMVA method and fitter classes, such as `MethodBase::ExitFromTraining()` or `FitterBase::SetIPythonInteractive()`.
* The **RooStats::DebuggingSampler** and **RooStats::DebuggingTestStat** classes are removed. They were mock implementations of the `TestStatSampler` and `TestStatistic` interfaces that returned uniform random numbers independent of the data, only meant for debugging the RooStats framework itself during its initial development.
* The `RooTrace` class is deprecated and will be removed in ROOT 6.44. It was a RooFit-specific memory tracer whose instrumentation hooks are compiled out by default, so it has been inert and untested for years. For memory debugging, please use general-purpose tools such as AddressSanitizer or Valgrind instead.
* Support for the AIX operating system has been removed from the codebase. This support has not been tested since the late v5 releases and the LLVM JIT is not yet supporting AIX.
* The headers Htypes.h and Gtypes.h that were deprecated in ROOT 6.20 will now emit warnings and will be fully removed in ROOT 6.44.
* The header GLConstants.h is no longer part of ROOT installed headers.

## Build System

### Moving from builtin dependencies to system-provided packages

* The general direction of the ROOT project is to become more and more reliant on system packages. It is *recommended* to make the packages required by ROOT available on the system, e.g. via a package manager, and not with the builtin mechanism. This allows for timely updates and reduces the size of the installed binaries.
* The previously vendored builtins `ftgl`, `gl2ps`, `gtest`, `nlohmann_json`, `unuran`, `civetweb`, `xxhash`, `pcre2`, should be installed in the system if possible (e.g. via `apt-get` or `homebrew` package managers). ROOT will not automatically fall-back to their builtin versions if these are not found: the user is informed of that with a helpful message. If installing these dependencies in the system is not possible, the CMake option `-Dbuiltin_XYZ=ON` has to be consciously chosen by the user.
* For the builtin versions of `ftgl`, `gl2ps`, `gtest`, `nlohmann_json`, `unuran`, `civetweb`, `xxhash`, `pcre2`, the source tarballs are now fetched from [SPI](https://spi.web.cern.ch)'s [website](https://lcgpackages.web.cern.ch/), as for the vast majority of ROOT's builtins.

## Python Interface

### Connecting Python callables to signals

`TQObject::Connect()` now directly accepts a Python callable as the slot, for
example `button.Connect("Clicked()", on_clicked)`. The arguments emitted by the
signal are forwarded to the callable, as far as its signature accepts them, and
the connection keeps the callable alive. Use `Disconnect(signal, callable)` to
undo the connection. Signals of any signature are supported, no longer only
those covered by the `TPyDispatcher::Dispatch()` overloads.

The `TPyDispatcher` class and its `ROOT/TPyDispatcher.h` header are removed:
it required the user to create and keep alive a dispatcher object manually,
and it was broken in recent releases anyway, since the interpreter could not
resolve its symbols from the `libROOTPythonizations` Python extension module.
Replace `obj.Connect(signal, "TPyDispatcher", disp, "Dispatch()")` with
`obj.Connect(signal, callable)`.

## I/O

* Reading a collection without its dictionary no longer crashes when the elements hold a `std::string` or a `TString`. The emulated collection proxy relocated its elements with a raw memory copy when its buffer had to grow, which corrupts an object that points into itself, such as a `std::string` using the small string optimization; the invalid pointer was then freed when the object was destroyed. Such elements are now destroyed and reconstructed at the new location instead. This affected for instance a `std::vector<std::pair<std::string,double>>` read back without a dictionary.

## Core

* `TClass::IsTriviallyRelocatable()` reports whether an object of a class can be moved to a new address with a raw memory copy, i.e. without running a move or copy constructor (trivial relocatability in the C++26 sense). It is backed by the new `kClassIsTriviallyRelocatable` class property, which `TInterpreter::ClassInfo_ClassProperty()` now fills in. A class the interpreter does not know about, in particular an emulated one, is conservatively reported as not relocatable.

## Histograms

### Cumulative histograms in more than one dimension

`TH1::GetCumulative()` now computes a true multi-dimensional cumulative for 2D
(`TH2`) and 3D (`TH3`) histograms, using the inclusion-exclusion principle: each
bin of the result holds the sum of all bins whose indices are no greater than
(forward) or no less than (backward) those of the target bin along *every* axis.
Previously the method accumulated a single running sum over the flattened bin
iteration, which did not correspond to a meaningful cumulative distribution in
more than one dimension.

The behavior for one-dimensional histograms is unchanged. Code that relied on
the previous 2D/3D output (for example to build per-axis selection efficiency
maps) will now obtain different, mathematically consistent values.

## Math

## RDataFrame

* Added `RedefinePerSample` transformation. Works similarly to `DefinePerSample`, but allows to redefine existing values
  of a column on a per-sample basis. This operation is supported in local and distributed mode.

## Trees

### Behavior change: `sqrt()` of negative arguments in TTreeFormula now returns NaN

Since its introduction in 1995, the formula engine used by `TTree::Draw()`, `TTree::Scan()` and `TTreeFormula`
silently evaluated `sqrt(x)` as `sqrt(abs(x))` for negative arguments (or as `0` in the optimized evaluation path
of the legacy `ROOT::v5::TFormula`). This could produce silently wrong results, e.g. in selections involving
`sqrt` of an expression that can become negative. `sqrt()` now returns NaN for negative arguments, consistent
with `TMath::Sqrt()`, the standard C `sqrt()`, and the modern `TFormula` used by `TF1`.
Note that in a selection, a NaN evaluates as `false`, so entries where the `sqrt` argument is negative now fail
the cut instead of being selected based on `sqrt(abs(x))`.

## RooFit

### Small changes

* The `RooMinimizer::Strategy` enum has been removed. It named the Minuit strategies that are usually referred to just by integers, but caused confusion because it didn't include the unnamed "Strategy 3". Since people usually set the strategy with integer values anyway, it was decided that the simplest solution to avoid the confusion was simply to remove the `RooMinimizer::Strategy` enum

### Deprecation of the legacy evaluation backend

The `legacy` evaluation backend for likelihood and chi-square fits is deprecated and will be removed in ROOT 6.44.
It was superseded by the vectorized `cpu` backend, which is the default since ROOT 6.32.
After the removal of the constant term optimization (see below), the legacy backend also has no performance-relevant feature left that would justify its continued maintenance.

Selecting the legacy backend with `RooFit::EvalBackend("legacy")` now prints a deprecation warning whenever a likelihood or chi-square object is created with it, and the `RooFit::EvalBackend::Legacy()` factory function is marked as deprecated, resulting in compiler warnings.

The **RooFit::BatchMode()** command argument, which was superseded by `RooFit::EvalBackend()` in ROOT 6.28, is deprecated at the same time and will also be removed in ROOT 6.44.
Note that the C++ declarations of `RooFit::BatchMode()` had been unintentionally absent since ROOT 6.30; they are restored in this release, marked as deprecated, to give downstream code a proper migration window.

The removal in ROOT 6.44 will also include:

  * the implementation classes of the legacy test statistics: **RooNLLVar**, **RooChi2Var**, **RooAbsOptTestStatistic** and **RooAbsTestStatistic** (their headers are not part of the public interface anymore since ROOT 6.32, but they are still installed),
  * the old multiprocessing mechanism of the legacy backend, consisting of the **RooRealMPFE** class and the underlying **BidirMMapPipe**,
  * the `nll::name[pdf,data]` and `chi2::name[pdf,data]` expressions in the `RooWorkspace::factory()` language, which instantiate the removed classes directly.

Users are strongly encouraged to switch to the default `cpu` evaluation backend, i.e., to simply not pass any `EvalBackend()` or `BatchMode()` command argument.
If the default backend does not work for a given use case, **please report it by opening an issue on the ROOT GitHub repository**.

### Removal of the constant term optimization for legacy test statistic classes

The **RooFit::Optimize()** option (constant term optimization) was deprecated in ROOT 6.40, and its functionality is now removed.
The `RooFit::Optimize()` and `RooMinimizer::optimizeConst()` methods are kept for API consistency across ROOT versions, but they have no effect anymore.

In practice this option only affected the `legacy` evaluation backend.

Together with the mechanism itself, the public interfaces that only existed to
drive it were removed. Code that called or overrode any of the following needs
to be adapted:

  * `RooAbsArg::constOptimizeTestStatistic()`, `RooAbsArg::findConstantNodes()`,
    `RooAbsArg::setCacheAndTrackHints()`, `RooAbsArg::canNodeBeCached()` and the
    `RooAbsArg::ConstOpCode` and `RooAbsArg::CacheMode` enums.
  * `RooAbsData::cacheArgs()`, `resetCache()`, `setArgStatus()`, `attachCache()`,
    `optimizeReadingWithCaching()`, `allClientsCached()` and `hasFilledCache()`,
    together with the corresponding `RooAbsDataStore` interface (`cacheArgs()`,
    `cacheOwner()`, `attachCache()`, `setArgStatus()`, `resetCache()`,
    `cachedVars()`, `recalculateCache()` and `forceCacheUpdate()`). Classes
    deriving from `RooAbsDataStore` no longer need to implement them.
  * `RooFit::TestStatistics::RooAbsL::constOptimizeTestStatistic()` and
    `RooFit::TestStatistics::LikelihoodWrapper::constOptimizeTestStatistic()`.

Since the cache-and-track hints are gone, the `"CacheAndTrack"`,
`"NOCacheAndTrack"` and `"NeverConstant"` attributes no longer have any effect.

The default vectorized CPU evaluation backend (introduced in ROOT 6.32) already performs these optimizations automatically and is not affected by this change.
Users are strongly encouraged to switch to the vectorized CPU backend if they are still using the legacy backend.

If the vectorized backend does not work for a given use case, **please report it by opening an issue on the ROOT GitHub repository**.

### Default binning of RooFit variables changed to zero bins

A freshly-constructed `RooRealVar` (or `RooErrorVar`) no longer has a default binning of 100 bins.
Instead, `RooAbsRealLValue::getBins()` now returns `0` until a binning is explicitly set (e.g. via `RooRealVar::setBins()`).
This makes it possible to distinguish a variable whose binning was deliberately chosen from one that was left at the default, which avoids
writing redundant `nbins` fields when serializing workspaces to HS3 JSON.

For the cases that previously relied on the default of 100 bins, that value is now injected by the relevant routine:
unbinned-dataset plotting (`RooAbsRealLValue::frame()`), `RooAbsPdf::generateBinned()`, and `RooAbsRealLValue::createHistogram()`
all fall back to `RooAbsRealLValue::DefaultNBins` (100) when the variable has no binning set.
As a result, plotting, generating binned data and creating histograms from a default-constructed variable behave exactly as before.

Code that reads `getBins()`/`numBins()` of a bare variable and expected the value `100` should either set the binning explicitly,
or read the bin count from the relevant histogram or plot frame instead.

### Resolution models are no longer imported into the workspace as standalone objects

When a `RooResolutionModel` (like `RooGaussModel` or `RooTruthModel`) is used wi th a
`RooAbsAnaConvPdf` (like `RooDecay`, `RooBDecay`, etc.), it acts as a *configuration* object that specifies which model to convolve the basis functions with, rather than as a nod
e of the pdf's computation graph.
The `RooAbsAnaConvPdf` builds its own internal basis-function convolutions from it and evaluates *those*.

Until now, the resolution model was nevertheless kept as a (non-value, non-shape) server of the `RooAbsAnaConvPdf`.
As a side effect, importing such a pdf into a `RooWorkspace` also imported the original resolution model as a standalone workspace object, and it leaked into HS3/JSON exports, even though it played no role in the computation.

Starting with ROOT 6.42, a resolution model that is only used as the configuration of a `RooAbsAnaConvPdf` is no longer a server of that pdf, and is therefore not imported into the
workspace on its own anymore. The model remains accessible via `RooAbsAnaConvPdf::getModel()`.

This is not expected to affect typical usage, since the resolution model was never part of the actual likelihood. Workspaces written with older ROOT versions are read back correctly via schema evolution.

## Graphics and GUI

### New POLF and POLN draw options for TH2

Since ROOT 6.36 implementation of "POL" draw option was changed. Angle and radius range automatically scaled to visible histogram range filling full 2*Pi angle and full radius range.
This let display TH2 in polar coordinates for any provided range settings, but produced plots are not intuitive. Therefore two new options for polar coordinates were introduced.

"POLF" - fixed polar coordinates. In such case full histogram X range mapped to -PI .. +Pi.
And Y axis mapped to radius. In case of histogram zooming angle and radius of each bin remains
the same (therefore name "fixed"), just number of displayed bins are reduced.

"POLN" - natural polar coordinates. In this case X axis directly represents angle value in radians (therefore name "natural") and Y axis is just radius.

### Store canvas as HTML file

Now canvas (or several canvases) can be stored in portable HTML file.
Just call `c1->SaveAs("canvas.html")` or invoke correspondent menu item.
To store several canvases in single HTML file one can use:
```cpp
   auto c1 = new TCanvas("c1", "c1", 4);
   auto c2 = new TCanvas("c2", "c2", 4);
   auto c3 = new TCanvas("c3", "c3", 4);
   TCanvas::SaveAll({c1, c2, c3}, "canvases.html");
```
Produced HTML file will include canvas JSON data and JavaScript code to load and display canvas.
Such file can be loaded locally in any web browser or send as attachment in email to colleagues.

## Geometry

### Improved multithreaded `TGeo` navigation

Multithreaded `TGeo` navigation is now faster and more scalable, with improved thread-local state management that avoids
false sharing, releases temporary memory during geometry cleanup, and correctly supports concurrent navigation of
multiple geometries.

For ALICE material-budget lookup-table generation on 28 cores, these changes reduced the runtime from 139 s to 72 s
and improved scaling from 12x to 23x.

The [TGeometry](https://root.cern/doc/master/classTGeometry.html) classes (Geant 3 shapes) have been moved out of Graf3D into their own library.
To link to these classes, use the cmake target `TGeometry` (preferred), `root-config --libs`, or link with `-lTGeometry`.
When ROOT is configured with `-Dgeom=Off`, these classes are now off as well.

The header X3DBuffer.h is no longer part of the installed ROOT headers.

## Documentation and Examples

## Build, Configuration and Testing

### Building the CUDA backend of RooFit separately

RooFit evaluates its models with backend libraries that `libRooBatchCompute`
loads at runtime, one of which, `libRooBatchCompute_CUDA`, is the only part of
ROOT that requires the CUDA toolkit. It is now possible to build that backend
on its own against an already installed ROOT, so that distributions can ship a
CUDA-free ROOT and provide the GPU backend as a separate package:

```bash
cmake -S <root-source-dir>/roofit/batchcompute -B build -DCMAKE_PREFIX_PATH=<root-install-prefix>
cmake --build build
cmake --install build
```

`roofit/batchcompute/CMakeLists.txt` doubles as the top-level `CMakeLists.txt`
of that standalone project, so no CMake code has to be maintained downstream.
By default the library is installed into the library directory of the ROOT
installation it was configured against, which is where RooFit looks for it. See
`roofit/batchcompute/README.md` for the details. Nothing changes for the regular
ROOT build: `-Dcuda=ON` still builds the CUDA backend together with everything
else.

## Versions of built-in packages

The version of the following packages has been updated:

 - xrootd: 5.9.5
