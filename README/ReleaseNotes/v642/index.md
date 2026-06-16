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

## Python Interface

## I/O

## Core

## Histograms

## Math

## RooFit

### Removal of the the constant term optimization for legacy test statistic classes

The **RooFit::Optimize()** option (constant term optimization) has been deprecated in ROOT 6.40 its functionality was now removed.
The `RooFit::Optimize()` and `RooMinimizer::optimizeConst()` methods are kept for API consistency across ROOT versions, but they have no effect anymore.

This option only affected the `legacy` evaluation backend.

The default vectorized CPU evaluation backend (introduced in ROOT 6.32) already performs these optimizations automatically and is not affected by this change.
Users are strongly encouraged to switch to the vectorized CPU backend if they are still using the legacy backend.

If the vectorized backend does not work for a given use case, **please report it by opening an issue on the ROOT GitHub repository**.

### Resolution models are no longer imported into the workspace as standalone objects

When a `RooResolutionModel` (like `RooGaussModel` or `RooTruthModel`) is used with a
`RooAbsAnaConvPdf` (like `RooDecay`, `RooBDecay`, etc.), it acts as a *configuration* object that
specifies which model to convolve the basis functions with, rather than as a node of the pdf's
computation graph. The `RooAbsAnaConvPdf` builds its own internal basis-function convolutions from
it and evaluates *those*.

Until now, the resolution model was nevertheless kept as a (non-value, non-shape) server of the
`RooAbsAnaConvPdf`. As a side effect, importing such a pdf into a `RooWorkspace` also imported the
original resolution model as a standalone workspace object, and it leaked into HS3/JSON exports,
even though it played no role in the computation.

Starting with ROOT 6.42, a resolution model that is only used as the configuration of a
`RooAbsAnaConvPdf` is no longer a server of that pdf, and is therefore not imported into the
workspace on its own anymore. The model remains accessible via `RooAbsAnaConvPdf::getModel()`.

This is not expected to affect typical usage, since the resolution model was never part of the
actual likelihood. Workspaces written with older ROOT versions are read back correctly via schema
evolution.

## Graphics and GUI

## Geometry

## Documentation and Examples

## Build, Configuration and Testing

## Versions of built-in packages

The version of the following packages has been updated:

 - xrootd: 5.9.5