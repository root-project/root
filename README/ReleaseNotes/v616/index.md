% ROOT Version 6.16 Release Notes
% 2018-06-25
<a name="TopOfPage"></a>

## Introduction

ROOT version 6.16/00 is scheduled for release end of 2018.

For more information, see:

[http://root.cern](http://root.cern)

The following people have contributed to this new version:

 Bertrand Bellenot, CERN/SFT,\
 Brian Bockelman, University of Nebraska-Lincoln,\
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
 Wouter Verkerke, NIKHEF/Atlas,\
 Jan Musinsky, SAS Kosice,\
 Enrico Guiraud, CERN,\
 Massimo Tumolo, Politecnico di Torino

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

### Change of setting the compression algorithm in `rootrc`

The previous setting called `ROOT.ZipMode` is now unused and ignored.
Instead, use `Root.CompressionAlgorithm` which sets the compression algorithm according to the values of [ECompression](https://root.cern/doc/master/Compression_8h.html#a0a7df9754a3b7be2b437f357254a771c):

* 0: use the default value of `R__ZipMode` (currently selecting LZ4)
* 1: use zlib (the default until 6.12)
* 2: use lzma
* 3: legacy, please don't use
* 4: LZ4 (the current default)

### TRef

* Improve thread scalability of TRef. Creating and looking up a lot of TRef from the same processID now has practically perfect weak scaling.

## I/O Libraries


## TTree Libraries
### RDataFrame
  - Optimise the creation of the set of branches names of an input dataset,
  doing the work once and caching it in the RInterface.
  - Add [StdDev](https://root.cern/doc/master/classROOT_1_1RDF_1_1RInterface.html#a482c4e4f81fe1e421c016f89cd281572) action.
  - Add [Display](https://root.cern/doc/master/classROOT_1_1RDF_1_1RInterface.html#aee68f4411f16f00a1d46eccb6d296f01) action and [tutorial](https://github.com/root-project/root/blob/master/tutorials/dataframe/df024_Display.C).
  - Add [Graph](https://root.cern/doc/master/classROOT_1_1RDF_1_1RInterface.html#a804b466ebdbddef5c7e3400cc6b89301) action and [tutorial](https://github.com/root-project/root/blob/master/tutorials/dataframe/df021_createTGraph.C).
  - Improve [GetColumnNames](https://root.cern/doc/master/classROOT_1_1RDF_1_1RInterface.html#a951fe60b74d3a9fda37df59fd1dac186) to have no redundancy in the returned names.
  - Add [Kahan Summation tutorial](https://github.com/root-project/root/blob/master/tutorials/dataframe/df022_useKahan.C) to subscribe a Kahan summation action to the RDataFrame.
  - Add [Aggregate tutorial](https://github.com/root-project/root/blob/master/tutorials/dataframe/df023_aggregate.C).
  - Fix ambiguous call on Cache() with one or two columns as parameters.
  - Add [GetFilterNames](https://root.cern/doc/master/classROOT_1_1RDF_1_1RInterface.html#a25026681111897058299161a70ad9bb2).

### TTree
  - TTrees can be forced to only create new baskets at event cluster boundaries.
    This simplifies file layout and I/O at the cost of memory.  Recommended for
    simple file formats such as ntuples but not more complex data types.  To
    enable, invoke `tree->SetBit(TTree::kOnlyFlushAtCluster)`.

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
  - In `TGraph2D` when the points were all in the same plane (along X or Y) at a
    negative coordinate, the computed axis limits were not correct. This was reported
    [here](https://root-forum.cern.ch/t/potential-bug-in-tgraph2d/29700/5).
  - Implemented the drawing of filled polygons in NDC space as requested
    [here](https://sft.its.cern.ch/jira/browse/ROOT-9523)
  - Implement the drawing of filled polygons in NDC space.
  - When drawing a histogram with the automatic coloring options (PMC, PLC etc ...)
    it was easy to forget to add a drawing option. This is now fixed. If no drawing
    option is specified the default drawing option for histogram is added.
  - When drawing a `TGraph` if `TH1::SetDefaultSumw2()` was on, then the underlying
    histogram used to draw the `TGraph` axis was created with errors and therefore
    the histogram painter painted it with errors which was a non sense in that
    particular case. This is now fixed. It was discussed
    [here](https://root-forum.cern.ch/t/horizontal-line-at-0-on-y-axis/30244/26)

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


