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

## Deprecation and Removal

* The build options `vc`, `veccore`, `builtin_vc`, `builtin_veccore` and `rpath` that were deprecated are now removed and will result in configuration errors if used.
* The option `fail-on-missing=OFF` is no longer honored for opt-in (ie OFF by default) build options requiring external dependencies such as `arrow`, `cocoa`, `daos`, `daos_mock`, `dcache`, `experimental_adaptivecpp`, `fcgi`, `fortran`, `gviz`, `mpi`, `pythia8`, `qt6web`, `tmva-cudnn`, `tmva-pymva`, `tmva-sofie`, `uring` or `vecgeom`. If the respective associated package dependency is not installed, ROOT will always raise a configuration error independent of the value of `fail-on-missing`. The user has to take action by either providing the dependency or manually disabling that option via `-Darrow=OFF`.
  Note that `all=ON` enables several of these options, so building with `-Dall=ON` now requires all of their dependencies to be installed, or the unwanted ones to be disabled explicitly.
  Build options that are enabled by default, such as `pyroot`, `opengl`, `xml`, `sqlite`, `davix`, `curl`, `tmva-cpu` or `tpython` are not affected: they are still disabled automatically when their dependencies are missing.
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
* The ROOT IO capability for the `TMVA::Experimental::SOFIE::RModel` has been removed. Users should not be encouraged to serialize models in experimental classes. For the serialization of ONNX models one can already use ONNX directly, and even serialize the ONNX bytes to a ROOT file if required.
* The Keras and PyTorch parsers for SOFIE (`TMVA::Experimental::SOFIE::PyKeras` and `PyTorch`) are now removed, so `RSofieReader` only accepts ONNX files.
These parsers relied on private implementation details of Keras and PyTorch, which change faster than is appropriate for ROOT's stability standards.
Users are encouraged to export their models to ONNX and use the retained ONNX parser instead.
* The ROOT IO capability for the `TMVA::Experimental::RBDT` class has been removed, along with the `TMVA.Experimental.SaveXGBoost` Python function. Experimental classes should not be persistified since their on-disk layout is not guaranteed to be stable. An `RBDT` is now built directly from an XGBoost model in its native JSON serialization with the new `TMVA::Experimental::RBDT::LoadXGBoost(jsonPath)`, which works both from C++ and Python. To convert a trained model, save it first with XGBoost's `Booster.save_model("model.json")` and then load it with `LoadXGBoost`.
* The **JsMVA** feature for interactive TMVA training in Jupyter notebooks is now removed. It was not functional for years and was therefore already excluded from ROOT 6.38. This also removes the `TMVA::IPythonInteractive` class and the related interactive-training interfaces from the TMVA method and fitter classes, such as `MethodBase::ExitFromTraining()` or `FitterBase::SetIPythonInteractive()`.
* The **RooStats::DebuggingSampler** and **RooStats::DebuggingTestStat** classes are removed. They were mock implementations of the `TestStatSampler` and `TestStatistic` interfaces that returned uniform random numbers independent of the data, only meant for debugging the RooStats framework itself during its initial development.
* The `RooTrace` class is deprecated and will be removed in ROOT 6.44. It was a RooFit-specific memory tracer whose instrumentation hooks are compiled out by default, so it has been inert and untested for years. For memory debugging, please use general-purpose tools such as AddressSanitizer or Valgrind instead.

## Python Interface

## I/O

## Core

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

## RooFit

### Small changes

* The `RooMinimizer::Strategy` enum has been removed. It named the Minuit strategies that are usually referred to just by integers, but caused confusion because it didn't include the unnamed "Strategy 3". Since people usually set the strategy with integer values anyway, it was decided that the simplest solution to avoid the confusion was simply to remove the `RooMinimizer::Strategy` enum

### Removal of the the constant term optimization for legacy test statistic classes

The **RooFit::Optimize()** option (constant term optimization) has been deprecated in ROOT 6.40 its functionality was now removed.
The `RooFit::Optimize()` and `RooMinimizer::optimizeConst()` methods are kept for API consistency across ROOT versions, but they have no effect anymore.

This option only affected the `legacy` evaluation backend.

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

The [TGeometry](https://root.cern/doc/master/classTGeometry.html) classes (Geant 3 shapes) have been moved out of Graf3D into their own library.
To link to these classes, use the cmake target `TGeometry` (preferred), `root-config --libs`, or link with `-lTGeometry`.
When ROOT is configured with `-Dgeom=Off`, these classes are now off as well.

## Documentation and Examples

## Build, Configuration and Testing

## Versions of built-in packages

The version of the following packages has been updated:

 - xrootd: 5.9.5
