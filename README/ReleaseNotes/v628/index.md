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
 Ivan Kabadzhov, CERN/SFT

## Deprecation and Removal

- The deprecated types `ROOT::Experimental::TBufferMerger` and `ROOT::Experimental::TBufferMergerFile` are removed.
Please use their non-experimental counterparts `ROOT::TBufferMerger` and `ROOT::TBufferMergerFile` instead.
- `ROOT::RVec::shrink_to_fit()` has now been removed after deprecation; it is not needed.
- `ROOT::RVec::emplace()` has now been removed after deprecation; please use `ROOT::RVec::insert()` instead.
- The deprecated function `ROOT::Detail::RDF::RActionImpl<Helper>::GetDataBlockCallback()` is removed; please use `GetSampleCallback()` instead.
- The deprecated RooFit containers `RooHashTable`, `RooNameSet`, `RooSetPair`, and `RooList` are removed. Please use STL container classes instead, like `std::unordered_map`, `std::set`, and `std::vector`.
- The `RooFit::FitOptions(const char*)` command to steer [RooAbsPdf::fitTo()](https://root.cern.ch/doc/v628/classRooAbsPdf.html) with an option string was removed. This way of configuring the fit was deprecated since at least since ROOT 5.02.
  Subsequently, the `RooMinimizer::fit(const char*)` function and the [RooMCStudy](https://root.cern.ch/doc/v628/classRooMCStudy.html) constructor that takes an option string was removed as well.

## Core Libraries


## I/O Libraries


## TTree Libraries

## RDataFrame

### New features

- Add [`GraphAsymmErrors`](https://root.cern/doc/master/classROOT_1_1RDF_1_1RInterface.html#acea30792eef607489d498bf6547a00a6) action that fills a TGraphAsymmErrors object.

### Notable bug fixes and improvements

- Fix the node counter of [`SaveGraph`](https://root.cern/doc/master/namespaceROOT_1_1RDF.html#ac06a36e745255fb8744b1e0a563074c9), where previously `cling` was getting wrong static initialization.
- Fix [`Graph`](https://root.cern/doc/master/classROOT_1_1RDF_1_1RInterface.html#a1ca9a94bece4767cac82968910afa02e) action (that fills a TGraph object) to properly handle containers and non-container types.
- The [`RCsvDS`](https://root.cern.ch/doc/master/classROOT_1_1RDF_1_1RCsvDS.html) class can have custom column types, and can properly ready empty entries of csv files.

## Histogram Libraries


## Math Libraries


## RooFit Libraries

### Code modernization by using `std::string` in RooFit interfaces

The following lesser-used RooFit functions now return a `std::string` instead of a `const char*`, potentially requiring the update of your code:

- [std::string RooCmdConfig::missingArgs() const](https://root.cern/doc/v628/classRooCmdConfig.html#aec50335293c45a507d347c604bf9651f)


## 2D Graphics Libraries


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


