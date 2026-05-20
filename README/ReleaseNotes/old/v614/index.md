% ROOT Version 6.14 Release Notes
% 2017-11-19

<a name="TopOfPage"></a>

## Introduction

ROOT version 6.14/00 is scheduled for release in 2018.

For more information, see:

[http://root.cern.ch](http://root.cern.ch)

The following people have contributed to this new version:

 Kim Albertsson, CERN/EP-ADP-OS,\
 Guilherme Amadio, CERN/SFT,\
 Bertrand Bellenot, CERN/SFT,\
 Brian Bockelman, UNL,\
 Rene Brun, CERN/SFT,\
 Philippe Canal, FNAL,\
 Olivier Couet, CERN/SFT,\
 Gerri Ganis, CERN/SFT,\
 Andrei Gheata, CERN/SFT,\
 Enrico Guiraud, CERN/SFT,\
 Raphael Isemann, Chalmers Univ. of Tech.,\
 Vladimir Ilievski, GSOC 2017,\
 Sergey Linev, GSI,\
 Pere Mato, CERN/SFT,\
 Lorenzo Moneta, CERN/SFT,\
 Axel Naumann, CERN/SFT,\
 Danilo Piparo, CERN/SFT,\
 Fons Rademakers, CERN/SFT,\
 Enric Tejedor Saavedra, CERN/SFT,\
 Oksana Shadura, UNL,\
 Saurav Shekhar, GSOC 2017,\
 Xavier Valls Pla, UJI, CERN/SFT,\
 Vassil Vassilev, Princeton/CMS,\
 Wouter Verkerke, NIKHEF/Atlas, RooFit,\
 Stefan Wunsch, CERN/SFT, \
 Zhe Zhang, UNL

## Important Notice

The default compression algorithm used when writing ROOT files has been updated to use LZ4 in particular to improve read (decompression) performance.  You can change this default for each file through (for example) the `TFile constructor` or `TFile::SetCompressionAlgorithm`.

It should be noted that ROOT files written with LZ4 compression can not be read with older release of ROOT.  Support for LZ4 was however back-ported to the patch branches of previous releases and the following tags (and later release in the same patch series) can read ROOT files written with LZ4 compression:

* v5.34/38
* v6.08/06 [not yet released]
* v6.10/08
* v6.12/02


## Removed interfaces

## Core Libraries
   - Optimize away redundant deserialization of template specializations. This reduces the memory footprint for hsimple by around 30% while improving the runtime performance for various cases by around 15%.
   - When ROOT is signaled with a SIGUSR2 (i.e. on Linux and MacOS X) it will now print a backtrace.
   - Move RStringView.h to ROOT/RStringView.hxx and always include ROOT/RStringView.hxx instead of RStringView.h for backward compatibility
   - In `TClingCallFunc`, support r-value reference parameters. This paves the way for the corresponding support in PyROOT (implemented now in the latest Cppyy).
   - Included the new TSequentialExecutor in ROOT, sharing the interfaces of TExecutor.This should improve code economy when providing a fallback for TThreadExecutor/TProcessExecutor.

### Thread safety
   - Resolved several race conditions, dead-locks, performance and order of initialization/destruction issues still lingering because of or despite the new read-write lock mechanism.

## Interpreter

   - Enabled use of multi-threaded code from the interpreter.
      - Previouslyl multi-threaded code could be run from the interpreter as long as the call starting the threada was the same code that initialized the ROOT global lock, any other uses, including attempting to run the same code a second time in the same session would lead to a dead lock (if any other thread attempted to take on the ROOT lock).
      - The interpreter now suspend the ROOT lock (which is taken to protect the interpreter global state) during user code execution.

## I/O Libraries
   - LZ4 (with compression level 4) is now the default compression algorithm for new ROOT files (LZ4 is lossless data compression algorithm that is focused on compression and decompression speed, while in ROOT case providing benefit in faster decompression at the price of a bit worse compression ratio comparing to ZLIB)
   - If two or more files have an identical streamer info record, this is only treated once therewith avoiding to take the global lock.
   - Allow writing temporary objects (with same address) in the same TBuffer(s). A new flag to TBuffer*::WriteObject allows to skip the mechanism that prevent the 2nd streaming of an object.  This allows the (re)use of temporary objects to store different data in the same buffer.
   - Reuse branch proxies internally used by TTreeReader{Value,Array} therewith increasing performance when having multiple readers pointing to the same branch.
   - Implement reading of objects data from JSON
   - Provide TBufferJSON::ToJSON() and TBufferJSON::FromJSON() methods
   - Provide TBufferXML::ToXML() and TBufferXML::FromXML() methods
   - Converts NaN and Infinity values into null in JSON, there are no other direct equivalent

## TTree Libraries
   - Enable the TTreeCache by default of `TTree::Draw`, `TTreeReader` and `RDataFrame`
   - Significant enhancement in the `TTreeCache` filling algorithm to increase robustness in case of oddly clustered `TTree` and under provisioned cache size.  See the [merge request](https://github.com/root-project/root/pull/1960) for more details.
   - Proxies are now properly re-used when multiple TTreeReader{Value,Array}s are associated to a single branch. Deserialisation is therefore performed once. This is an advantage for complex TDataFrame graphs.
   - Add TBranch::BackFill to allow the addition of new branches to an existing tree and keep the new basket clustered in the same way as the rest of the TTree.  Use with the following pattern,
   make sure to to call BackFill for the same entry for all the branches consecutively:
```
  for(auto e = 0; e < tree->GetEntries(); ++e) { // loop over entries.
    for(auto branch : branchCollection) {
      ... Make change to the data associated with the branch ...
      branch->BackFill();
    }
  }
```
Since we loop over all the branches for each new entry all the baskets for a cluster are consecutive in the file.

### RDataFrame (formerly TDataFrame)
#### Behaviour, interface and naming changes
   - `TDataFrame` and `TDataSource` together with their federation of classes have been renamed according to the coding conventions for new interfaces and extracted from the `Experimental` namespace: they can now be found in the ROOT namespace and they are called `ROOT::RDataFrame` and `ROOT::RDataSource`.
   - `ROOT::Experimental::TDF::TResultProxy` has been renamed to `ROOT::RDF::RResultPtr`.
   - `Report` now behaves identically to all other actions: it executes lazily and returns a `RResultPtr` (see the `New features` section for more details).
   - `Snapshot` now returns a `RResultPtr` like all other actions: specifically, this is a pointer to a new `RDataFrame` which will run on the snapshotted dataset.
   - `RDataFrame` has been removed from tree/treeplayer and put in its own package, tree/dataframe. The library where this code can be found is `libROOTDataFrame`. This new library is included in the list provided by `root-config --libs`.
   - The `TArrayBranch` class has been removed and replaced by the more powerful `RVec` (see the `New features` section for more details).
   - All `RDataFrame` tutorials are now prefixed with `df` rather than `tdf`.
   - Histograms and profiles returned by RDataFrame (e.g. by a Histo1D action) are now not associated to a ROOT directory (their fDirectory is a nullptr).
     The following still works as expected:
     ```
     auto h = tdf.Histo1D("x");
     TFile f(fname, "RECREATE");
     h->Write(); // event loop is run here and h is written to the TFile f
     ```

#### New features
   - The TDataSource interface changed. The `TDataSource::SetEntry` method now returns a boolean. If true the entry is processed within the event loop managed by the tdf, skipped otherwise.
   - The TLazyDS data source has been added. It allows to create a source starting from ResultProxies to vectors.
   - `TDataFrameInterface<T>::Report` returns a `TCutflowReport` object which can be inspected programmatically.
   - Add `Aggregate` action and implement `Reduce` in terms of it.
   - Add support for a more general leafname syntax that includes pathnames with multiple dots, such as "myBranch.mySubBranch.myLeaf". This is available both for jitted expressions and for lists of column names.
   - The CSV data source (TCsvDS) can now be constructed with a chunk size parameter, and as a result the CSV file will be read progressively, in chunks of the specified size. This can be used to prevent the whole CSV file from being read into memory at once, thus reducing the memory footprint of this data source.
   - Add the `ROOT::Experimental::TAdoptAllocator<T>`, an allocator which allows to adopt existing memory. If memory is adopted, upon allocation a copy is performed in the new, potentially more extended, memory region.
   - Add `ROOT::Experimental::VecOps::TVec<T>` a class which represents a contiguous array, inspired by Numpy arrays. `TVec` offer a convenient interface, almost identical to the one of `std::vector`. It can own or adopt its memory. As well as a set of tools which make analysis of collections easier, avoiding to loop over the individual elements of the collections. Basic arithmetic operations such as +,-,*,/,% between TVecs and scalars and TVecs are supported. Most popular math functions which act on TVecs are provided. Helpers to calculate basic quantities such as sum, mean, variance or standard deviation of TVecs are provided.
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
   - Per object statsoverflow flag has been added. This change is required to prevent non reproducible behaviours in a multithreaded environments. For example, if several threads change the
   `TH1::fgStatOverflows` flag and fill histograms, the behaviour will be undefined.
   - A fix has been added in resetting the statistics of histograms with label. The bug was causing the histogram entries to be  set as zero and this was making failing the merging of those
     histogram (see ROOT-9336). 

## Math Libraries


## RooFit Libraries

- A fix has been added in the component selection, which is used for plotting simultaneous models. See [PR #2033](https://github.com/root-project/root/pull/2033).

## TMVA Library

#### New Deep Learning Module

 - TMVA contains a new set of Deep Learning classes ( `MethodDL` ), with support, in addition to dense layer, also convolutional and recurrent layer. 

#### Other New TMVA Features

- Support for Parallelization of BDT using Multi-Threads
- Several improvements in Cross Validation including support for Multi-Process cross-validation running. 


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
   - New color palette "cividis" implemented by Sven Augustin.
     This colormap aims to solve problems that people with color vision deficiency have
     with the common colormaps. For more details see:
     Nuñez J, Anderton C, and Renslow R. Optimizing colormaps with consideration
     for color vision deficiency to enable accurate interpretation of scientific data.
     See the article [here](https://arxiv.org/abs/1712.01662)
   - New graphics style "ATLAS" from M.Sutton.
   - In `TGraphPainter` the fit parameters were painted too early. [In some cases graph's
     error bars overlapped the stat box](https://root-forum.cern.ch/t/hide-error-bars-behind-tpavestats/27996).
   - Implement the possibility to generate high definition bitmap pictures in `TImageDump`.
     This done via `gStyle->SetImageScaling(x);` `x` being a multiplication factor.
     This new feature is now used to generate the reference guide with `x=3`.
     Pictures in the reference guide are now much shaper and in particular the text.

## 3D Graphics Libraries
  - When a LEGO plot was drawn with Theta=90, the X and Y axis were misplaced.

## Geometry Libraries

## Database Libraries
  - Fix issue related to time stamps manipulation done by `TPgSQLStatement` as suggested [here](https://root-forum.cern.ch/t/please-correct-bug-reading-date-time-from-postgresql-tpgsqlstatement).

## Networking Libraries
   - New THttpWSHandler class should be used to work with websockets. It includes all necessary methods to handle multiple connections correctly. See in tutorials/http/ws.C how it can be used.
   - Interface of THttpWSEngine class was changed, all its instances handled internally in THttpWSHandler.

## GUI Libraries

## Montecarlo Libraries

## Parallelism
   - `TTree::GetEntry`: if IMT is enabled, run work in tasks if we have at least more than one top level branch.
   - Make EnableImplicitMT no-op if IMT is already on
   - Decompress `TTreeCache` in parallel if IMT is on (upgrade of the `TTreeCacheUnzip` class).
   - In `TTreeProcessorMT` delete friend chains after the main chain to avoid double deletes.
   - If IMT is enabled, the multithreaded execution of the fit respects the number of threads IMT has been initialized with.


## Language Bindings

### Notebook integration
   - In the ROOT kernel, avoid import of unnecessary components.
   - In the ROOT kernel, optimise regexes involved in tab-completion which could take up to minutes to be executed

## JavaScript ROOT
 
Upgrade JSROOT to v5.4.1. Following new features implemented:

* New supported classes:
   - TDiamond
   - TArc
   - TCurlyLine
   - TCurlyArc
   - TCrown
* New draw options:
   - "RX" and "RY" for TGraph to reverse axis
   - "noopt" for TGraph to disable drawing optimization
   - "CPN" for TCanvas to create color palette from N last colors
   - "line" for TGraph2D
* New features:
   - support LZ4 compression
   - tooltips and zooming in TGraphPolar drawings
   - TPavesText with multiple underlying paves
   - implement all fill styles
   - draw borders for TWbox
   - draw all objects from TList/TObjArray as they appear in list of primitives
   - let enable/disable highlight of extra objects in geometry viewer
   - draw axis labels on both sides when pad.fTick[x/y] > 1
   - make drawing of TCanvas with many primitives smoother
   - add fOptTitle, fOptLogx/y/z fields in JSROOT.gStyle
* Behavior changes:
   - disable automatic frame adjustment, can be enabled with "&adjframe" parameter in URL
   - when drawing TH2/TH3 scatter plots, always generate same "random" pattern
   - use barwidth/baroffset parameters in lego plots
* Bug fixes:
   - use same number of points to draw lines and markers on the TGraph
   - correctly draw filled TArrow endings
   - let combine "L" or "C" TGraph draw option with others
   - correct positioning of custom axis labels
   - correctly toggle lin/log axes in lego plot
   - let correctly change marker attributes interactively 
   - monitoring mode in draw.htm page
   - zooming in colz palette
   - support both 9.x and 10.x jsdom version in Node.js (#149)
   - draw axis main line with appropriate attributes (#150)
   - use axis color when drawing grids lines (#150)
   - when set pad logx/logy, reset existing user ranges in pad
   - avoid too deep calling stack when drawing many graphs or histos (#154)
   - correctly (re)draw tooltips on canvas with many subpads


## Code Examples

  - New graphics tutorial AtlasExample.C illustrating the ATLAS style.
  - New TLazyDS tutorial added tdf015_LazyDataSource.C.
  - Show how to inspect a `TCutFlowReport` object.

## Class Reference Guide

  - Replace low resolution images with bigger ones more suited for modern screens.

## Build System and Configuration

  - ROOT can now be built against an externally built llvm and clang (llvm can be used unpatched, clang still require ROOT specific patches).  The options are builtin_llvm and builtin_clang both defaulting to ON.
  - Update RConfigure.h with R__HAS__VDT if the package is found/builtin
  - CMake exported targets now have the `INTERFACE_INCLUDE_DIRECTORIES` property set ([ROOT-8062](https://sft.its.cern.ch/jira/browse/ROOT-8062)).
  - The `-fPIC` compile flag is no longer propagated to dependent projects via `CMAKE_CXX_FLAGS` ([ROOT-9212](https://sft.its.cern.ch/jira/browse/ROOT-9212)).
  - Several builtins have updated versions:
     - OpenSSL was updated from 1.0.2d to 1.0.2.o (latest lts release, [ROOT-9359](https://sft.its.cern.ch/jira/browse/ROOT-9359))
     - Davix was updated from 0.6.4 to 0.6.7 (support for OpenSSL 1.1, [ROOT-9353](https://sft.its.cern.ch/jira/browse/ROOT-9353))
     - Vdt has been updated from 0.3.9 to 0.4.1 (includes new atan function)
     - XRootd has been updated from 4.6.1 to 4.8.2 (for GCC 8.x support)
     - Builtin TBB can now be used on Windows
     - xxHash and LZ4 have been separated so that a system version of LZ4 can be used even if it does not include xxHash headers ([ROOT-9099](https://sft.its.cern.ch/jira/browse/ROOT-9099))
  - In addition, several updates have been made to fix minor build system issues, such as not checking for external packages if their builtin is turned off, or checking for packages even when the respective option is disabled ([ROOT-8806](https://sft.its.cern.ch/jira/browse/ROOT-8806), [ROOT-9190](https://sft.its.cern.ch/jira/browse/ROOT-9190), [ROOT-9315](https://sft.its.cern.ch/jira/browse/ROOT-9315), [ROOT-9385](https://sft.its.cern.ch/jira/browse/ROOT-9385)).
  - The `python3` option to CMake has been removed ([ROOT-9033](https://sft.its.cern.ch/jira/browse/ROOT-9033), [ROOT-9143](https://sft.its.cern.ch/jira/browse/ROOT-9143)). Python support is enabled by default. To configure ROOT to use specific Python versions, there is a new option called `python_version`. This is how to configure ROOT and Python for the common use cases:

    * Use the default Python interpreter:
      - `-Dpython=ON` (default)
    * Search only for Python 2.x or only 3.x:
      - `-Dpython_version=2` or `-Dpython_version=3`
    * Use a specific version of Python from `$PATH`:
      - `-Dpython_version=2.7` or `-Dpython_version=3.5`
    * Use a specific Python interpreter, whatever the version:
      - `-DPYTHON_EXECUTABLE=/usr/local/bin/python`

Note: The use of `PYTHON_EXECUTABLE` requires the full path to the interpreter.

## Infrastructure and Testing

   - Reduce time taken by tests which takes too long to run ({One,Two}SidedFrequentistUpperLimitWithBands.C)
   - Disable PyROOT SQL tutorials (the C++ counterparts are since several releases).
