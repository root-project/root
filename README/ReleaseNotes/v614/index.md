% ROOT Version 6.14 Release Notes
% 2017-11-19

<a name="TopOfPage"></a>

## Introduction

ROOT version 6.14/00 is scheduled for release in 2018.

For more information, see:

[http://root.cern.ch](http://root.cern.ch)

The following people have contributed to this new version:

 Guilherme Amadio, CERN/SFT,\
 Bertrand Bellenot, CERN/SFT,\
 Brian Bockelman, UNL,\
 Rene Brun, CERN/SFT,\
 Philippe Canal, FNAL,\
 David Clark, ANL (SULI),\
 Olivier Couet, CERN/SFT,\
 Gerri Ganis, CERN/SFT,\
 Andrei Gheata, CERN/SFT,\
 Enrico Guiraud, CERN/SFT,\
 Sergey Linev, GSI,\
 Timur Pocheptsov, CERN/SFT,\
 Pere Mato, CERN/SFT,\
 Lorenzo Moneta, CERN/SFT,\
 Axel Naumann, CERN/SFT,\
 Danilo Piparo, CERN/SFT,\
 Fons Rademakers, CERN/SFT,\
 Enric Tejedor Saavedra, CERN/SFT,\
 Peter van Gemmeren, ANL,\
 Vassil Vassilev, Princeton/CMS,\
 Wouter Verkerke, NIKHEF/Atlas, RooFit

## Removed interfaces

## Core Libraries
   - Optimize away redundant deserialization of template specializations. This reduces the memory footprint for hsimple
     by around 22% while improving the runtime performance for various cases by around 15%.
   - When ROOT is signaled with a SIGUSR2 (i.e. on Linux and MacOS X) it will now print a backtrace.

## I/O Libraries
   - Implement reading of objects data from JSON
   - Provide TBufferJSON::ToJSON() and TBufferJSON::FromJSON() methods
   - Provide TBufferXML::ToXML() and TBufferXML::FromXML() methods

## TTree Libraries

### TDataFrame

## Histogram Libraries

## Math Libraries

## RooFit Libraries

## 2D Graphics Libraries
   - `TMultiGraph::GetHistogram` now works even if the multigraph is not drawn. Make sure
     it never returns a null pointer.
   - X11 line `width = 0` doesn't work on OpenSuSE Thumbleweed for non solid lines. Now fixed.
   - TCanvas::SetWindowsSize has been changed to get the same window size in interactive mode…and batch mode.
   - Change the `TGraph` default fill color to white to avoid black box in legend
     when `gPad->BuildLegend()` is called.
   - Auto-coloring for TF1 (drawing options PFC, PLC and PMC) is implemented.
   - Auto-coloring for TH1::DrawCopy (drawing options PFC, PLC and PMC) is implemented.
   - Improve the option management in `TF1::Draw` to allow to combine the option
     `SAME` with other drawing options.
   - `TGraph::Draw("AB")` was malfunctioning when using `TAxis::SetRangeUser`.
      It was reported [here](https://sft.its.cern.ch/jira/browse/ROOT-9144).
   - The errors end-caps size in `TLegend` follows the value set by `gStyle->SetEndErrorSize()`.
     For instance setting it to 0 allows to remove the end-caps both on the graph and the legend.
     It was requested [here](https://sft.its.cern.ch/jira/browse/ROOT-9184)
   - New color palette "cividis"implemented by Sven Augustin.
     This colormap aims to solve problems that people with color vision deficiency have
     with the common colormaps. For more details see:
     Nuñez J, Anderton C, and Renslow R. Optimizing colormaps with consideration
     for color vision deficiency to enable accurate interpretation of scientific data.
     See the article [here](https://arxiv.org/abs/1712.01662)

## 3D Graphics Libraries
  - When a LEGO plot was drawn with Theta=90, the X and Y axis were misplaced.

## Geometry Libraries

## Database Libraries

## Networking Libraries

Changes in websockets handling in THttpServer.
   - New THttpWSHandler class should be used to work with websockets.
     It includes all necessary methods to handle multiple connections correctly.
     See in tutorials/http/ws.C how it can be used.
   - Interface of THttpWSEngine class was changed, all its instances handled internally in THttpWSHandler.

## GUI Libraries

## Montecarlo Libraries

## Parallelism

## Language Bindings

## JavaScript ROOT

## Tutorials

## Class Reference Guide

## Build, Configuration and Testing Infrastructure


