% ROOT Version 6.08 Release Notes
% 2015-11-12
<a name="TopOfPage"></a>

## Introduction

ROOT version 6.08/00 is scheduled for release in May, 2016.

For more information, see:

[http://root.cern.ch](http://root.cern.ch)

The following people have contributed to this new version:

 David Abdurachmanov, CERN, CMS,\
 Bertrand Bellenot, CERN/SFT,\
 Rene Brun, CERN/SFT,\
 Philippe Canal, FNAL,\
 Cristina Cristescu, CERN/SFT,\
 Olivier Couet, CERN/SFT,\
 Gerri Ganis, CERN/SFT,\
 Andrei Gheata, CERN/Alice,\
 Lukasz Janyst, CERN/IT,\
 Christopher Jones, Fermilab, CMS,\
 Wim Lavrijsen, LBNL, PyRoot,\
 Sergey Linev, GSI, http,\
 Pere Mato, CERN/SFT,\
 Lorenzo Moneta, CERN/SFT,\
 Axel Naumann, CERN/SFT,\
 Danilo Piparo, CERN/SFT,\
 Timur Pocheptsov, CERN/SFT,\
 Fons Rademakers, CERN/SFT,\
 Enric Tejedor Saavedra, CERN/SFT,\
 Liza Sakellari, CERN/SFT,\
 Manuel Tobias Schiller,\
 David Smith, CERN/IT,\
 Matevz Tadel, UCSD/CMS, Eve,\
 Vassil Vassilev, Fermilab/CMS,\
 Wouter Verkerke, NIKHEF/Atlas, RooFit

<a name="core-libs"></a>
## Core Libraries

ROOT prepares for [cxx modules](http://clang.llvm.org/docs/Modules.html). One of
the first requirements is its header files to be self-contained (section "Missing
Includes"). ROOT header files were cleaned up from extra includes and the missing
includes were added.

This could be considered as backward incompatibility (for good however). User
code may need to add extra includes, which were previously resolved indirectly
by including a ROOT header. For example:

  * TBuffer.h - TObject.h doesn't include TBuffer.h anymore. Third party code,
    replying on the definition of TBufer will need to include TBuffer.h, along
    with TObject.h.
  * TSystem.h - for some uses of gSystem.
  * GeneticMinimizer.h
  * ...

### Dictionaries

Fix ROOT-7760: fully allow the usage of the dylib extension on OSx.

## Histogram Libraries


## Math Libraries


## RooFit Libraries


## TTree Libraries


## 2D Graphics Libraries


## 3D Graphics Libraries


## Geometry Libraries


## I/O Libraries
Custom streamers need to #include TBuffer.h explicitly (see
[section Core Libraries](#core-libs)


## Database Libraries


## Networking Libraries


## GUI Libraries


## Montecarlo Libraries


## PROOF Libraries


## Language Bindings


## JavaScript ROOT


## Tutorials


## Class Reference Guide


## Build, Configuration and Testing Infrastructure


