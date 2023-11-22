% ROOT Version 6.32 Release Notes
% 2023-10-10
<a name="TopOfPage"></a>

## Introduction

ROOT version 6.32/00 is scheduled for release around May 2024.

For more information, see:

[http://root.cern](http://root.cern)

The following people have contributed to this new version:

 Guilherme Amadio, CERN/IT,\
 Bertrand Bellenot, CERN/EP-SFT,\
 Jakob Blomer, CERN/EP-SFT,\
 Rene Brun,\
 Philippe Canal, FNAL,\
 Olivier Couet, CERN/EP-SFT,\
 Gerri Ganis, CERN/EP-SFT,\
 Florine de Geus, CERN/ATLAS,\
 Andrei Gheata, CERN/EP-SFT,\
 Enrico Guiraud, CERN/EP-SFT and Princeton,\
 Jonas Hahnfeld, CERN/EP-SFT,\
 Akeem Hart, Queen Mary University of London/DUNE and MINERvA,\
 Sergey Linev, GSI,\
 Pere Mato, CERN/EP-SFT,\
 Lorenzo Moneta, CERN/EP-SFT,\
 Alja Mrak Tadel, UCSD/CMS,\
 Axel Naumann, CERN/EP-SFT,\
 Vincenzo Padulano, CERN/EP-SFT,\
 Danilo Piparo, CERN/EP-SFT,\
 Fons Rademakers, CERN/IT,\
 Jonas Rembser, CERN/EP-SFT,\
 Matevz Tadel, UCSD/CMS,\
 Vassil Vassilev, Princeton/CMS,\
 Wouter Verkerke, NIKHEF/Atlas,

## Deprecation and Removal
- The RooFit legacy iterators are deprecated and will be removed in ROOT 6.34 (see section "RooFit libraries")

## Core Libraries


## I/O Libraries


## TTree Libraries


## Histogram Libraries


## Math Libraries


## RooFit Libraries

### Deprecation of legacy iterators

The following methods related to the RooFit legacy iterators are deprecated and will be removed in ROOT 6.34.
They should be replaced with the suitable STL-compatible interfaces, or you can just use range-based loops:

- `RooAbsArg::clientIterator()`: use `clients()` and `begin()`, `end()` or range-based loops instead
- `RooAbsArg::valueClientIterator()`: use `valueClients()`
- `RooAbsArg::shapeClientIterator()`: use `shapeClients()`
- `RooAbsArg::serverIterator()`: use `servers()`
- `RooAbsArg::valueClientMIterator()`: use `valueClients()`
- `RooAbsArg::shapeClientMIterator()`: use `shapeClients()`
- `RooAbsArg::serverMIterator()`: use `servers()`

- `RooAbsCollection::createIterator()`: use `begin()`, `end()` and range-based for loops
- `RooAbsCollection::iterator()`: same
- `RooAbsCollection::fwdIterator()`: same

- `RooWorkspace::componentIterator()`: use `RooWorkspace::components()` with range-based loop

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


## Tutorials


## Class Reference Guide


## Build, Configuration and Testing Infrastructure


