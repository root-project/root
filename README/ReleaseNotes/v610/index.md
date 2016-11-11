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
- When the option SAME (or "SAMES") is used with the option COL, the boxes' color
  are computing taking the previous plots into account. The range along the Z axis
  is imposed by the first plot (the one without option SAME); therefore the order
  in which the plots are done is relevant.
- With option BOX on 2D histos with negative content:
    - do not draw the empty bins as requested [here](https://sft.its.cern.ch/jira/browse/ROOT-8385).
    - fix the issue mentioned [here](https://sft.its.cern.ch/jira/browse/ROOT-*402).
- When several histogram were drawn on top of each other with the option
  `BOX SAME` and if the log scale along Z was on, the plot showed only the
  first histogram. This can be reproduce by using the documentation example
  illustrating `BOX SAME`and turning the canvas into log scale along Z.
- In TLatex: Do not paint the text when the text size is <= 0. This fixes
  the problem mentioned [here](https://sft.its.cern.ch/jira/browse/ROOT-8305)
- From: Sergey Linev: In `TPad::SaveAs` method json file extension is now handled
- Because of some precision issue some data points exactly on the plot limits of
  a `TGraph2D` were not drawn (option `P`).
  The problem was reported [here](https://sft.its.cern.ch/jira/browse/ROOT-8447).

## 3D Graphics Libraries
- In `TMarker3DBox::PaintH3` the boxes' sizes was not correct.
- The option `BOX`and `GLBOX` now draw boxes with a volume proportional to the
  bin content to be conform to the 2D case where the surface of the boxes is
  proportional to the bin content.

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


