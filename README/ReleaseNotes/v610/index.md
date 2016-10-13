% ROOT Version 6.10 Release Notes
% 2016-09-30
<a name="TopOfPage"></a>

## Introduction

ROOT version 6.10/00 is scheduled for release in 2017.

For more information, see:

[http://root.cern.ch](http://root.cern.ch)

The following people have contributed to this new version:

 Bertrand Bellenot, CERN/SFT,\
 Rene Brun, CERN/SFT,\
 Philippe Canal, FNAL,\
 Olivier Couet, CERN/SFT,\
 Gerri Ganis, CERN/SFT,\
 Andrei Gheata, CERN/SFT,\
 Sergey Linev, GSI, http,\
 Pere Mato, CERN/SFT,\
 Lorenzo Moneta, CERN/SFT,\
 Axel Naumann, CERN/SFT,\
 Danilo Piparo, CERN/SFT,\
 Fons Rademakers, CERN/SFT,\
 Enric Tejedor Saavedra, CERN/SFT,\
 Vassil Vassilev, Fermilab/CMS,\
 Wouter Verkerke, NIKHEF/Atlas, RooFit

## Removed interfaces

The following interfaces have been removed, after deprecation in v6.08.

### CINT remnants, dysfunctional for ROOT 6
- `TInterpreter`'s `Getgvp()`, `Getp2f2funcname(void*)`, `Setgvp(Long_t)`, `SetRTLD_NOW()`, `SetRTLD_LAZY()`.
- `SetFCN(void*)` from TVirtualFitter, TFitter, TBackCompFitter, TMinuit
- `TFoam::SetRhoInt(void*)`


## Core Libraries

## Histogram Libraries


## Math Libraries


## RooFit Libraries


## TTree Libraries


## 2D Graphics Libraries
- If one used "col2" or "colz2", the value of `TH1::fMaximum` got modified.
  This deviated from the behavior of "col" or "colz". This is now fixed as
  requested [here](https://sft.its.cern.ch/jira/browse/ROOT-8389).
- With option BOX on 2D histos: do not draw the empty bins as requested
  [here](https://sft.its.cern.ch/jira/browse/ROOT-8385)

## 3D Graphics Libraries


## Geometry Libraries


## I/O Libraries


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


