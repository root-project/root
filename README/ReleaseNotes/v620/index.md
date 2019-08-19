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

## ROOT

### Splash screen

The venerable splash screen is now disabled by default to make ROOT's startup
faster. Many users already use `root -l` to start ROOT, but this also hides the
useful text banner with version information along with the splash screen. With
this new default, starting up ROOT as just `root` will show only the text banner
instead of the splash screen. The splash screen can still be seen with `root -a`
or in `TBrowser` by opening `Browser Help → About ROOT`.

## Deprecation and Removal
 * rootcling flags `-cint`, `-reflex` and `-gccxml` have no effect and will be
   removed. Please remove them from the rootcling invocations.
 * genreflex flag `--deep` has no effect and will be removed. Please remove it
   from the genreflex invocation.

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
- Revisited the TSpectrum2 documentation. All the static images have been replaced
  by macros generating images at reference guide build time. These macros have
  been added in the tutorial section of the reference guide.


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
- The file `RootNewMacros.cmake` has been renamed to `RootMacros.cmake`.
  Including the old file by name is deprecated and will generate a warning.
  Including `RootMacros.cmake` is not necessary, as now it is already included
  when calling `find_package(ROOT)`. If you still need to inherit ROOT's compile
  options, however, you may use `include(${ROOT_USE_FILE})` as before.
- ROOT's internal CMake modules (e.g. CheckCompiler.cmake, SetUpLinux.cmake, etc)
  are no longer installed with `make install`. Only the necessary files by
  dependent projects are installed by default now, and they are installed
  directly into the cmake/ directory, not cmake/modules/ as before.
- The macro `ROOT_GENERATE_DICTIONARY()` can now attach the generated source
  file directly to a library target by using the options `NOTARGET MODULE
  <library>`, where `<library>` is an existing library target. This then allows
  the dictionary to inherit target properties such as compile options and
  include directories directly from the library target, even when they are added
  after the call to `ROOT_GENERATE_DICTIONARY()`.
- The macros `REFLEX_GENERATE_DICTIONARY()` and `ROOT_GENERATE_DICTIONARY()` can
  now have custom extra dependencies added with the options `DEPENDS` and
  `EXTRA_DEPENDENCIES`, respectively.

The following builtins have been updated:

- FFTW3 3.3.8
- GSL 2.5
- Intel TBB 2019 U8
- PCRE 8.43
- OpenSSL 1.0.2s
- Vdt 0.4.3
- VecCore 0.6.0
- XRootD 4.10.0
