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
* The **JsMVA** feature for interactive TMVA training in Jupyter notebooks is now removed. It was not functional for years and was therefore already excluded from ROOT 6.38. This also removes the `TMVA::IPythonInteractive` class and the related interactive-training interfaces from the TMVA method and fitter classes, such as `MethodBase::ExitFromTraining()` or `FitterBase::SetIPythonInteractive()`.

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

## RooFit

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

## Graphics and GUI

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

## Documentation and Examples

## Build, Configuration and Testing

## Versions of built-in packages

The version of the following packages has been updated:

 - xrootd: 5.9.5
