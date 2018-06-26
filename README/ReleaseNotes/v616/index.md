% ROOT Version 6.16 Release Notes
% 2018-06-25
<a name="TopOfPage"></a>

## Introduction

ROOT version 6.16/00 is scheduled for release end of 2018.

For more information, see:

[http://root.cern](http://root.cern)

The following people have contributed to this new version:

 Bertrand Bellenot, CERN/SFT,\
 Rene Brun, CERN/SFT,\
 Philippe Canal, FNAL,\
 Olivier Couet, CERN/SFT,\
 Gerri Ganis, CERN/SFT,\
 Andrei Gheata, CERN/SFT,\
 Sergey Linev, GSI,\
 Pere Mato, CERN/SFT,\
 Lorenzo Moneta, CERN/SFT,\
 Axel Naumann, CERN/SFT,\
 Danilo Piparo, CERN/SFT,\
 Fons Rademakers, CERN/SFT,\
 Enric Tejedor Saavedra, CERN/SFT,\
 Matevz Tadel, UCSD/CMS,\
 Vassil Vassilev, Princeton/CMS,\
 Wouter Verkerke, NIKHEF/Atlas, \
 Jan Musinsky, SAS Kosice

## Deprecation and Removal

### Ruby bindings

The ruby binding has been unmaintained for several years; it does not build with current ruby versions.
Given that this effectively meant that Ruby was dysfunctional and given that nobody (but package maintainers) has complained, we decided to remove it.

### Removal of previously deprecated or disabled packages

The packages `afs`, `chirp`, `glite`, `sapdb`, `srp` and `ios` have been removed from ROOT.
They were deprecated before, or never ported from configure, make to CMake.


## Core Libraries

### Fish support for thisroot script

`. bin/thisroot.fish` sets up the needed ROOT environment variables for one of the ROOT team's favorite shells, the [fish shell](https://fishshell.com/).


## I/O Libraries


## TTree Libraries
### RDataFrame
  - Optimise the creation of the set of branches names of an input dataset,
  doing the work once and caching it in the RInterface.

## Histogram Libraries


## Math Libraries


## RooFit Libraries


## 2D Graphics Libraries

  - Highlight mode is implemented for `TH1` and for `TGraph` classes. When
    highlight mode is on, mouse movement over the bin will be represented
    graphically. Histograms bins or graph points will be highlighted. Moreover,
    any highlight emits signal `TCanvas::Highlighted()` which allows the user to
    react and call their own function. For a better understanding see also
    the tutorials `$ROOTSYS/tutorials/hist/hlHisto*.C` and
    `$ROOTSYS/tutorials/graphs/hlGraph*.C` .
  - Implement fonts embedding for PDF output. The "EmbedFonts" option allows to
    embed the fonts used in a PDF file inside that file. This option relies on
    the "gs" command (https://ghostscript.com).

    Example:

~~~ {.cpp}
   canvas->Print("example.pdf","EmbedFonts");
~~~
  - In TAttAxis::SaveAttributes` take into account the new default value for `TitleOffset`.
  - When the histograms' title's font was set in pixel the position of the
    `TPaveText` containing the title was not correct. This problem was reported
    [here](https://root-forum.cern.ch/t/titles-disappear-for-font-precision-3/).

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


