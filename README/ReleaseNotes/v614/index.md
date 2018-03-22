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
 Oksana Shadura, UNL,\
 Wouter Verkerke, NIKHEF/Atlas, RooFit

## Removed interfaces

## Core Libraries
   - Optimize away redundant deserialization of template specializations. This reduces the memory footprint for hsimple by around 30% while improving the runtime performance for various cases by around 15%.
   - When ROOT is signaled with a SIGUSR2 (i.e. on Linux and MacOS X) it will now print a backtrace.
   - Move RStringView.h to ROOT/RStringView.hxx and always include ROOT/RStringView.hxx instead of RStringView.h for backward compatibility
   - In `TClingCallFunc`, support r-value reference parameters. This paves the way for the corresponding support in PyROOT (implemented now in the latest Cppyy).

## I/O Libraries
   - LZ4 (with compression level 4) is now the default compression algorithm for new ROOT files (LZ4 is lossless data compression algorithm that is focused on compression and decompression speed, while in ROOT case providing benefit in faster decompression at the price of a bit worse compression ratio comparing to ZLIB)
   - Implement reading of objects data from JSON
   - Provide TBufferJSON::ToJSON() and TBufferJSON::FromJSON() methods
   - Provide TBufferXML::ToXML() and TBufferXML::FromXML() methods

## TTree Libraries

### TDataFrame
   - Histograms and profiles returned by TDataFrame (e.g. by a Histo1D action) are now not associated to a ROOT directory (their fDirectory is a nullptr).
     The following still works as expected:
     ```
     auto h = tdf.Histo1D("x");
     TFile f(fname, "RECREATE");
     h->Write(); // event loop is run here and h is written to the TFile f
     ```

#### New features
   - The TLazyDS data source has been added. It allows to create a source starting from ResultProxies to vectors.
   - `TDataFrameInterface<T>::Report` returns a `TCutflowReport` object which can be inspected programmatically.
   - Add `Aggregate` action and implement `Reduce` in terms of it.
   - Add support for a more general leafname syntax such as "myBranch.myLeaf" - not available in jitted strings but only usable in lists of columns names.
   - Add the `ROOT::Experimental::TAdoptAllocator<T>`, an allocator which allows to adopt existing memory. If memory is adopted, upon allocation a copy is performed in the new, potentially more extended, memory region.
   - Add `ROOT::Experimental::ExperimentalTVec<T>` a class which represents a contiguous array, inspired by Numpy arrays. `TVec` offer a convenient interface, almost identical to the one of `std::vector`. It can own or adopt its memory. As well as a set of tools which make analysis of collections easier, avoiding to loop over the individual elements of the collections. Basic arithmetic operations such as +,-,*,/,% between TVecs and scalars and TVecs are supported. Most popular math functions which act on TVecs are provided. Helpers to calculate basic quantities such as sum, mean, variance or standard deviation of TVecs are provided.
   A powerful and concise syntax for expressing cuts is available:
```
  // mu_pts_tvec and mu_etas_tvec are two equally sized TVecs holding kinematic properties of muons
  // a filter on muons pseudorapidities is applied considering a range in pseudo rapidity.
  filtered_mu_pts_tvec = mu_pts_tvec[abs(mu_etas_tvec) < 2)];
```
   - The `TArrayBranch` class has been removed and replaced by the more powerful `TVec`.
   - Columns on disk stored as C arrays should be read as `TVec`s, `std::vector` columns can be read as `TVec`s if requested. Jitted transformations and actions consider `std::vector` columns as well as C array columns `TVec`s.
   - In jitted transformations and actions, `std::vector` and C array columns are read as `TVec`s.
   - When snapshotting, columns read from trees which are of type `std::vector` or C array and read as TVecs are persistified on disk as a `std::vector` or C arrays respectively - no transformation happens. `TVec` columns, for example coming from `Define`s, are written as `std::vector<T, TAdoptAllocator<T>>`.

#### Fixes
   - Do not alphabetically order columns before snapshotting to avoid issues when writing C arrays the size of which varies and is stored in a separate branch.
   - Validate columns before writing datasets on disk.
   - Check the type of the columns via type info in CSV, ROOT and trivial data source.
   - Allow to snapshot a dataset read from a `TCsvDS`.
   - Snapshot and Cache now properly trigger column definitions.
   - Correctly deduce type of Float_t branches when jitting.
   - Do not rely on branches' titles for runtime type inference.
   - Do not loose an entry when using Range and multiple actions.

#### Other changes
   - Throw an exception if the type of a branch cannot be deduced.


## Histogram Libraries
   - Per object statsoverflow flag has been added. This change is required to prevent non reproducible behaviours in a multithreaded environments. For example, if several threads change the `TH1::fgStatOverflows` flag and fill histograms, the behaviour will be undefined.

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
   - New graphics style "ATLAS" from M.Sutton.
   - In `TGraphPainter` the fit parameters were painted too early. [In some cases graph's
     error bars overlapped the stat box](https://root-forum.cern.ch/t/hide-error-bars-behind-tpavestats/27996).


## 3D Graphics Libraries
  - When a LEGO plot was drawn with Theta=90, the X and Y axis were misplaced.

## Geometry Libraries

## Database Libraries
  - Fix issue related to time stamps manipulation done by `TPgSQLStatement` as suggested [here](https://root-forum.cern.ch/t/please-correct-bug-reading-date-time-from-postgresql-tpgsqlstatement).

## Networking Libraries

Changes in websockets handling in THttpServer.
   - New THttpWSHandler class should be used to work with websockets.
     It includes all necessary methods to handle multiple connections correctly.
     See in tutorials/http/ws.C how it can be used.
   - Interface of THttpWSEngine class was changed, all its instances handled internally in THttpWSHandler.

## GUI Libraries

## Montecarlo Libraries

## Parallelism
   - `TTree::GetEntry`: if IMT is enabled, run work in tasks if we have at least more than one top level branch.
   - Make EnableImplicitMT no-op if IMT is already on
   - Decompress `TTreeCache` in parallel if IMT is on (upgrade of the `TTreeCacheUnzip` class).
   - In `TTreeProcessorMT` delete friend chains after the main chain to avoid double deletes.


## Language Bindings

### Notebook integration
   - In the ROOT kernel, avoid import of unnecessary components.
   - In the ROOT kernel, optimise regexes involved in tab-completion which could take up to minutes to be executed

## JavaScript ROOT

## Code Examples

  - New graphics tutorial AtlasExample.C illustrating the ATLAS style.
  - New TLazyDS tutorial added tdf015_LazyDataSource.C.
  - Show how to inspect a `TCutFlowReport` object.

## Class Reference Guide
  - Replace low resolution images with bigger ones more suited for modern screens.

## Build, Configuration and Testing Infrastructure

The `python3` option to CMake has been removed. Python support is enabled by
default. To configure ROOT to use specific Python versions, there is a new
option called `python_version`. This is how to configure ROOT and Python for
the common use cases:

* Use the default Python interpreter:
  - `-Dpython=ON` (default)
* Search only for Python 2.x or only 3.x:
  - `-Dpython_version=2` or `-Dpython_version=3`
* Use a specific version of Python from `$PATH`:
  - `-Dpython_version=2.7` or `-Dpython_version=3.5`
* Use a specific Python interpreter, whatever the version:
  - `-DPYTHON_EXECUTABLE=/usr/local/bin/python`

Note: The use of `PYTHON_EXECUTABLE` requires the full path to the interpreter.

   - Reduce time taken by tests which takes too long to run ({One,Two}SidedFrequentistUpperLimitWithBands.C)
   - Disable PyROOT SQL tutorials (the C++ counterparts are since several releases).
