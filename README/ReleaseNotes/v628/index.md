% ROOT Version 6.28 Release Notes
% 2022-01-05
<a name="TopOfPage"></a>

## Introduction

ROOT version 6.28/00 is scheduled for release in May 2022.

For more information, see:

[http://root.cern](http://root.cern)

The following people have contributed to this new version:

 Bertrand Bellenot, CERN/SFT,\
 Jakob Blomer, CERN/SFT,\
 Rene Brun, CERN/SFT,\
 Philippe Canal, FNAL,\
 Olivier Couet, CERN/SFT,\
 Gerri Ganis, CERN/SFT,\
 Andrei Gheata, CERN/SFT,\
 Enrico Guiraud, CERN/SFT,\
 Sergey Linev, GSI,\
 Javier Lopez-Gomez, CERN/SFT,\
 Pere Mato, CERN/SFT,\
 Lorenzo Moneta, CERN/SFT,\
 Axel Naumann, CERN/SFT,\
 Vincenzo Eduardo Padulano, CERN/SFT and UPV,\
 Danilo Piparo, CERN/SFT,\
 Fons Rademakers, CERN/SFT,\
 Jonas Rembser, CERN/SFT,\
 Enric Tejedor Saavedra, CERN/SFT,\
 Matevz Tadel, UCSD/CMS,\
 Vassil Vassilev, Princeton/CMS,\
 Wouter Verkerke, NIKHEF/Atlas,\
 Ivan Kabadzhov, CERN/SFT,\
 David Poulton, Wits/SFT

## Deprecation and Removal

- The deprecated types `ROOT::Experimental::TBufferMerger` and `ROOT::Experimental::TBufferMergerFile` are removed.
Please use their non-experimental counterparts `ROOT::TBufferMerger` and `ROOT::TBufferMergerFile` instead.
- `ROOT::RVec::shrink_to_fit()` has now been removed after deprecation; it is not needed.
- `ROOT::RVec::emplace()` has now been removed after deprecation; please use `ROOT::RVec::insert()` instead.
- The deprecated function `ROOT::Detail::RDF::RActionImpl<Helper>::GetDataBlockCallback()` is removed; please use `GetSampleCallback()` instead.
- The deprecated RooFit containers `RooHashTable`, `RooNameSet`, `RooSetPair`, and `RooList` are removed. Please use STL container classes instead, like `std::unordered_map`, `std::set`, and `std::vector`.
- The `RooFit::FitOptions(const char*)` command to steer [RooAbsPdf::fitTo()](https://root.cern.ch/doc/v628/classRooAbsPdf.html) with an option string was removed. This way of configuring the fit was deprecated since at least since ROOT 5.02.
  Subsequently, the `RooMinimizer::fit(const char*)` function and the [RooMCStudy](https://root.cern.ch/doc/v628/classRooMCStudy.html) constructor that takes an option string were removed as well.
- The overload of `RooAbsData::createHistogram` that takes integer parameters for the bin numbers is now deprecated and will be removed in ROOT 6.30.
  This was done to avoid confusion with inconsistent behavior when compared to other `createHistogram` overloads.
  Please use the verson of `createHistogram` that takes RooFit command arguments.
- The `RooAbsData::valid()` method to cache valid entries in the variable range
  was removed. It was not implemented in RooDataSet, so it never worked as
  intended. Related to it was the `RooDataHist::cacheValidEntries()` function, which is removed as well.
  The preferred way to reduce RooFit datasets to subranges is [RooAbsData::reduce()](https://root.cern.ch/doc/v628/classRooAbsData.html#acfa7b31e5cd751eec1bc4e95d2796390).
- The longtime-deprecated `RooStats::HistFactory::EstimateSummary` class is removed, including the functions that use it. The information that it was meant to store is managed by the `RooStats::HistFactory::Measurement` object since many years.
- The `RooSuperCategory::MakeIterator()` function that was deprecated since 6.22 is now removed. Please use range-based loops to iterate over the category states.


## rootreadspeed

This version adds the new `rootreadspeed` CLI tool. This tool can be used to help identify bottlenecks in analysis runtimes, by providing time and throughput measurements when reading ROOT files via file systems or XRootD. More detailed information can be found in the tool's help information.

To see help information, install and source a recent enough version of ROOT, and run the command `rootreadspeed --help` in your terminal.

### Example usage of the tool:

```console
$ rootreadspeed --files <local-folder>/File1.root xrootd://<url-folder>/File2.root --trees Events --all-branches --threads 8
```

## Core Libraries


## I/O Libraries


## TTree Libraries

## RDataFrame

### New features

- Add [`GraphAsymmErrors`](https://root.cern/doc/master/classROOT_1_1RDF_1_1RInterface.html#acea30792eef607489d498bf6547a00a6) action that fills a TGraphAsymmErrors object.

### Notable bug fixes and improvements

- Fix the node counter of [`SaveGraph`](https://root.cern/doc/master/namespaceROOT_1_1RDF.html#ac06a36e745255fb8744b1e0a563074c9), where previously `cling` was getting wrong static initialization.
- Fix [`Graph`](https://root.cern/doc/master/classROOT_1_1RDF_1_1RInterface.html#a1ca9a94bece4767cac82968910afa02e) action (that fills a TGraph object) to properly handle containers and non-container types.
- The [`RCsvDS`](https://root.cern.ch/doc/master/classROOT_1_1RDF_1_1RCsvDS.html) class now allows users to specify column types, and can properly read empty entries of csv files.

## Histogram Libraries

- New class `TGraph2DAsymmErrors` to create TGraph2D with asymmetric errors.
![TGraph2DAsymmErrors](TGraph2DAsymmErrors.png)

## Math Libraries


## RooFit Libraries

### Consistent definition of the default minimizer type for all of RooFit/RooStats

In previous releases, the default minimizer type that RooFit used was hardcoded to be the original `Minuit`, while RooStats used the default minimizer specified by `ROOT::Math::MinimizerOptions::DefaultMinimizerType()`. Now it is possible to centrally define the global minimizer for all RooFit libraries via `ROOT::Math::MinimizerOptions::SetDefaultMinimizer()`, or alternatively in the `.rootrc` file by adding for example `Root.Fitter: Minuit2` to select Minuit2.


### Code modernization by using `std::string` in RooFit interfaces

The following lesser-used RooFit functions now return a `std::string` instead of a `const char*`, potentially requiring the update of your code:

- [std::string RooCmdConfig::missingArgs() const](https://root.cern/doc/v628/classRooCmdConfig.html#aec50335293c45a507d347c604bf9651f)
### Uniquely identifying RooArgSet and RooDataSet objects

Before v6.28, it was ensured that no `RooArgSet` and `RooDataSet` objects on the heap were located at an address that had already been used for an instance of the same class before.
With v6.28, this is not guaranteed anymore.
Hence, if your code uses pointer comparisons to uniquely identify RooArgSet or RooDataSet instances, please consider using the new `RooArgSet::uniqueId()` or `RooAbsData::uniqueId()`.

### Introducing binned likelihood fit optimization in HistFactory

In a binned likelihood fit, it is possible to skip the PDF normalization when
the unnormalized binned PDF can be interpreted directly in terms of event
yields. This is now done by default for HistFactory models, which
results in great speedups for binned fits with many channels. Some RooFit users
like ATLAS were already using this for a long time.

To disable this optimization when using the `hist2workspace` executable, add the `-disable_binned_fit_optimization` command line argument.
Directly in C++, you can also set the `binnedFitOptimization` to `false` in the
HistFactory configuration as follows:
```C++
RooStats::HistFactory::MakeModelAndMeasurementFast(measurement, {.binnedFitOptimization=false});
```
If your compiler doesn't support aggregate initialization with designators, you
need to create and edit the configuration struct explicitely:
```C++
RooStats::HistFactory::HistoToWorkspaceFactoryFast::Configuration hfCfg;
hfCfg.binnedFitOptimization = false;
RooStats::HistFactory::MakeModelAndMeasurementFast(measurement, hfCfg);
```

### Disable copy assignment for RooAbsArg and derived types

Copy assignment for RooAbsArgs was implemented in an unexpected and
inconsistent way. While one would expect that the copy assignment is copying
the object, it said in the documentation of `RooAbsArg::operator=` that it will
"assign all boolean and string properties of the original bject. Transient
properties and client-server links are not assigned." This contradicted with
the implementation, where the server links were actually copied too.
Furthermore, in `RooAbsRealLValue`, the assigment operator was overloaded by a
function that only assigns the value of another `RooAbsReal`.

With all these inconsistencies, it was deemed safer to disable copy assignment
of RooAbsArgs from now on.

### Removal of deprecated HistFactory functionality

#### Removal of HistoToWorkspaceFactory (non-Fast version)

The original `HistoToWorkspaceFactory` produced models that consisted of a
Poisson term for each bin.  In this "number counting form" the dataset has one
row and the collumns corresponded to the number of events for each bin. This
led to severe performance problems in statistical tools that generated
pseudo-experiments and evaluated likelihood ratio test statistics.

Nowadays, everyone uses the faster `HistoToWorkspaceFactoryFast` implementation that
produces a model in the "standard form" where the dataset has one row for each
event, and the column corresponds to the value of the observable in the
histogram.

Therefore, the original `HistoToWorkspaceFactory` is now removed to avoid
confusion and maintainance burden.

#### Removing constant parameter flag from RooStats:HistFactory::NormFactor

As printed out by the HistFactory in a warning message for a long time already,
setting the `Const` attribute to the `<NormFactor>` tag is deprecated and it
will be ignored. Instead, add `<ParamSetting Const="True"> myparam </ParamSetting>` to your top-level XML's `<Measurement>` entry.

This deprecation implied that the constant parameter flag in the
`RooStats:HistFactory::NormFactor` class had no effect as well. To avoid
ambiguity in the future, the possibility to set and retrieve this flag with
`NormFactor::SetConst()` and `NormFactor::GetConst()` was removed, as well as the
`Sample::AddNormFactor(std::string Name, double Val, double Low, double High, bool Const)`
overload. Also, the aforementioned deprecation warning is not printed anymore.

### Removal of `RooAbsMinimizerFcn` and `RooMinimizerFcn` from the public interface

The `RooAbsMinimizerFcn` class and its implementation `RooMinimizerFcn` were removed from the public interface.
These classes are implementation details of the RooMinimizer and should not be used in your code.
In the unlikely case that this causes any problem for you, please open a GitHub issue requesting to extend the RooMinimizer by the needed functionality.

### Vectorize `RooAbsBinning` interface for bin index lookups

The `RooAbsBinning` interface for bin index lookups was changed to enable vectorized implementations.
Instead of having the override `RooAbsBinning::binNumber()`, the binning implementations now have to override the `RooAbsBinning::binNumbers()` function to evaluate the bin indices of multiple values in one function call.

## 2D Graphics Libraries

- Implement the option "File": The current file name is painted on the bottom right of each plot
  if the option `File` is set on via `gStyle->SetOptFile()`.

- In matplolib one can use the "Default X-Points" feature to plot X/Y graphs: If one doesn't
  specify the points in the x-axis, they will get the default values 0, 1, 2, 3, (etc. depending
  on the length of the y-points). The matplotlib script will be:
```
   import matplotlib.pyplot as plt
   import numpy as np
   points = np.array([3, 8, 1, 10, 5, 7])
   plt.plot(ypoints)
   plt. show()
```
It is now possible to do the same with the ROOT TGraph:
```
   double y[6] = {3, 8, 1, 10, 5, 7};
   auto g = new TGraph(6,y);
   g->Draw();
```

So, if we take the same example as above, and leave out the x-points, the diagram will look like this:

## 3D Graphics Libraries


## Geometry Libraries


## Database Libraries


## Networking Libraries

### THttpServer

- upgrade civetweb code to version 1.15, supports SSL version 3.0
- resolve problem with symbolic links usage on Windows
- let disable/enable directory files listing via THttpServer (default is off)


## GUI Libraries


## Montecarlo Libraries


## PROOF Libraries


## Language Bindings


## JavaScript ROOT


## Tutorials


## Class Reference Guide


## Build, Configuration and Testing Infrastructure


