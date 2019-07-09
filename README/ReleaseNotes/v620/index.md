% ROOT Version 6.20 Release Notes
% 2019-05-29
<a name="TopOfPage"></a>

## Introduction

ROOT version 6.20/00 is scheduled for release in November 2019.

For more information, see:

[http://root.cern](http://root.cern)

The following people have contributed to this new version:

 Kim Albertsson, CERN/ATLAS,\
 Guilherme Amadio, CERN/SFT,\
 Bertrand Bellenot, CERN/SFT,\
 Iliana Betsou, CERN/SFT,\
 Jakob Blomer, CERN/SFT,\
 Brian Bockelman, Nebraska,\
 Rene Brun, CERN/SFT,\
 Philippe Canal, FNAL,\
 Javier Cervantes Villanueva, CERN/SFT,\
 Olivier Couet, CERN/SFT,\
 Alexandra Dobrescu, CERN/SFT,\
 Giulio Eulisse, CERN/ALICE,\
 Gerri Ganis, CERN/SFT,\
 Andrei Gheata, CERN/SFT,\
 Enrico Guiraud, CERN/SFT,\
 Stephan Hageboeck, CERN/SFT,\
 Jan Knedlik, GSI,\
 Sergey Linev, GSI,\
 Pere Mato, CERN/SFT,\
 Lorenzo Moneta, CERN/SFT,\
 Alja Mrak-Tadel, UCSD/CMS,\
 Axel Naumann, CERN/SFT,\
 Vincenzo Eduardo Padulano, Bicocca/SFT,\
 Danilo Piparo, CERN/SFT,\
 Fons Rademakers, CERN/SFT,\
 Henry Schreiner, Princeton,\
 Oksana Shadura, Nebraska,\
 Simon Spies, GSI,\
 Yuka Takahashi, Princeton and CERN/SFT,\
 Enric Tejedor Saavedra, CERN/SFT,\
 Matevz Tadel, UCSD/CMS,\
 Vassil Vassilev, Princeton/CMS,\
 Wouter Verkerke, NIKHEF/Atlas,\
 Zhe Zhang, Nebraska,\
 Stefan Wunsch, CERN/SFT

## Deprecation and Removal

### Deprecated packages

### Removed packages

## Core Libraries


## I/O Libraries


## TTree Libraries


## Histogram Libraries


## Math Libraries


## RooFit Libraries


## 2D Graphics Libraries


## 3D Graphics Libraries


## Geometry Libraries


## Database Libraries


## Networking Libraries


## GUI Libraries


## Montecarlo Libraries


## PROOF Libraries


## Language Bindings


## JavaScript ROOT
- Provide monitoring capabilities for TGeoManager object. Now geomtry with some tracks can be displayed and
  updated in web browser, using THttpServer monitoring capability like histogram objects.


## Tutorials
- Add the "Legacy" category collecting the old tutorials which do not represent any more best practices


## Class Reference Guide
- Images in tutorials can now be displayed à JavaScript thanks to the (js) option
  added next to the directive `\macro_image`
- As the tutorial `palettes.C` is often hit when searching the keyword `palette`
  in the reference guide, a direct link from this example to the full list of
  predefined palettes given in `TColor` has been added.


## Build, Configuration and Testing Infrastructure

- Make MLP optional via the `-Dmlp={OFF,ON}` switch for CMake
- Make Spectrum optional via the `-Dspectrum={OFF,ON}` switch for CMake
- ROOT now fails to configure when any package is missing
  when `-Dfail-on-missing=ON` is passed to CMake
- The `-Dall=ON` now switches the default value of all optional packages to `ON`
- The options `astiff`, `cling`, `pch`, `thread`, and `explicitlink` have been
  removed and are now ignored. They either had no effect (their value was not
  being used in the build system), or could not be disabled (like `cling` and
  `explicitlink`).
- ROOT library targets now export which C++ standard they were built with via
  the target compile features `cxx_std_11`, `cxx_std_14`, and `cxx_std_17`.

The following builtins have been updated:

- Intel TBB 2019 U7
- OpenSSL 1.0.2s
- XRootD 4.9.1
