% ROOT Version 6.14 Release Notes
% 2018-07-27

<a name="TopOfPage"></a>

## Introduction

ROOT version 6.14/00 has been released on June 13, 2018.

For more information, see:

[http://root.cern](http://root.cern)

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

The default compression algorithm was LZ4 for v6.14/00 and v6.14/02, and reverted to zlib for v6.14/06 due to a lack of compression efficiency for certain kinds of data. LZ4 will likely become the default again once this behavior is repaired.

The default compression algorithm used when writing ROOT files has been updated to use LZ4 in particular to improve read (decompression) performance.  You can change this default for each file through (for example) the `TFile constructor` or `TFile::SetCompressionAlgorithm`.

It should be noted that ROOT files written with LZ4 compression can not be read with older release of ROOT.  Support for LZ4 was however back-ported to the patch branches of previous releases and the following tags (and later release in the same patch series) can read ROOT files written with LZ4 compression:

* v5.34/38
* v6.08/06 [not yet released]
* v6.10/08
* v6.12/02


## Removed interfaces

## Core Libraries
   - Dictionary generation for classes with a deleted default ctor is now supported.
   - Optimize away redundant deserialization of template specializations. This reduces the memory footprint for hsimple by around 30% while improving the runtime performance for various cases by around 15%.
   - When ROOT is signaled with a SIGUSR2 (i.e. on Linux and MacOS X) it will now print a backtrace.
   - Move RStringView.h to ROOT/RStringView.hxx and always include ROOT/RStringView.hxx instead of RStringView.h for backward compatibility
   - In `TClingCallFunc`, support r-value reference parameters. This paves the way for the corresponding support in PyROOT (implemented now in the latest Cppyy).

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
   - Add `ROOT::VecOps::RVec<T>`, a class which represents a contiguous array, inspired by Numpy arrays. `RVec` offer a convenient interface which extends the one of `std::vector`. It can own or adopt its memory. `RVec` comes with a set of tools which make analysis of collections easier, avoiding to loop over the individual elements of the collections. Basic arithmetic operations such as +,-,*,/,% between RVecs and scalars and RVecs are supported. Most popular math functions which act on RVecs are provided. Helpers to calculate basic quantities such as sum, mean, variance or standard deviation of RVecs are provided.
   A powerful and concise syntax for expressing cuts is available:
```
  // mu_pts_tvec and mu_etas_tvec are two equally sized RVecs holding kinematic properties of muons
  // a filter on muons pseudorapidities is applied considering a range in pseudo rapidity.
  filtered_mu_pts_tvec = mu_pts_tvec[abs(mu_etas_tvec) < 2)];
```
   - `RDataFrame` can now write multiple output files concurrently: `Snapshot` can be made lazy specifying the appropriate flag in the `RSnapshotOptions`
   - `RDataFrame` now supports custom actions, i.e. actions which are provided by the user. A tutorial has been added to illustrate this functionality: `tutorials/dataframe/df018_customActions.C`.
   - The `RArrowDS` class has been added which allows to read arrow tables in RDataFrame. This source can be activated with the configuration switch `-D arrow=ON`.
   - The RDataSource interface changed. The `RDataSource::SetEntry` method now returns a boolean. If true the entry is processed within the event loop managed by the tdf, skipped otherwise.
   - The RLazyDS data source has been added. It allows to create a source starting from ResultProxies to vectors.
   - `RDataFrameInterface<T>::Report` returns a `RCutflowReport` object which can be inspected programmatically.
   - Add `Aggregate` action and implement `Reduce` in terms of it.
   - Add support for a more general leafname syntax that includes pathnames with multiple dots, such as "myBranch.mySubBranch.myLeaf". This is available both for jitted expressions and for lists of column names.
   - The CSV data source (TCsvDS) can now be constructed with a chunk size parameter, and as a result the CSV file will be read progressively, in chunks of the specified size. This can be used to prevent the whole CSV file from being read into memory at once, thus reducing the memory footprint of this data source.
   - Add the `ROOT::Details::RAdoptAllocator<T>`, an allocator which allows to adopt existing memory. If memory is adopted, upon allocation a copy is performed in the new, potentially more extended, memory region.
   - Columns on disk stored as C arrays should be read as `RVec`s, `std::vector` columns can be read as `RVec`s if requested. Jitted transformations and actions consider `std::vector` columns as well as C array columns `RVec`s.
   - In jitted transformations and actions, `std::vector` and C array columns are read as `RVec`s.
   - When snapshotting, columns read from trees which are of type `std::vector` or C array and read as RVecs are persistified on disk as a `std::vector` or C arrays respectively - no transformation happens. `RVec` columns, for example coming from `Define`s, are written as `std::vector<T, RAdoptAllocator<T>>`.
   - Support aliasing of leaves.

#### Fixes
   - passing strings to `Filter` and `Define` is now much faster, and should not be a runtime bottleneck anymore.
   - TDataFrame now respects user-provided custom binnings in result histograms.
   - Do not alphabetically order columns before snapshotting to avoid issues when writing C arrays the size of which varies and is stored in a separate branch.
   - Validate columns before writing datasets on disk.
   - Check the type of the columns via type info in CSV, ROOT and trivial data source.
   - Allow to snapshot a dataset read from a `RCsvDS`.
   - Snapshot and Cache now properly trigger column definitions.
   - Correctly deduce type of Float\_t branches when jitting.
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


## Language Bindings

### PyROOT
* Interoperability between C++ objects and numpy arrays has been enhanced:
   - New pythonizations for vector classes (`std::vector`, `TVec`), which are contiguous in memory, allow to convert objects of those classes to numpy arrays without extra copies. This can be done by calling `numpy.asarray(vec_obj)` from Python.
   - A new pythonization for TTree is now offered, incarnated in the `AsMatrix` method, which allows to convert `TTree`s with columns of arithmetic types to numpy arrays.

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
  - Show how to inspect a `RCutFlowReport` object.

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

   - ROOT now uses the NEW behavior for CMake policy CMP0051 (list
     `TARGET_OBJECTS` in `SOURCES` target property). Please run
     `cmake --help-policy CMP0051` for more information.
   - The file `ROOTUseFile.cmake` used by dependent projects to load ROOT CMake
     macros now requires the same version of CMake as ROOT itself.

## Infrastructure and Testing

   - `root-config` now provides switches to link against libROOTDataFrame and libROOTVecOps
   - Reduce time taken by tests which takes too long to run ({One,Two}SidedFrequentistUpperLimitWithBands.C)
   - Disable PyROOT SQL tutorials (the C++ counterparts are since several releases).


## Bugs and Issues fixed in this release

* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9166">[ROOT-9166]</a> - TDataSource: Read only a subset of the columns
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9349">[ROOT-9349]</a> - [TDF] Only invoke the interpreter once per event loop
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9371">[ROOT-9371]</a> - [TDF] Allow users to define and execute custom actions
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9374">[ROOT-9374]</a> - Re-enable dataframe_{interface,simple} and test_stringfiltercolumn tests
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9380">[ROOT-9380]</a> - [TDF] Switch back to using Calc rather than ProcessLine when finished debugging test failures
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-5432">[ROOT-5432]</a> - Possible rounding bug in THashTable::AverageCollisions
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-6872">[ROOT-6872]</a> - crash on TMemFile deletion in multi-threaded app
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-7417">[ROOT-7417]</a> - Check for OPENGL_LIBRARIES if building on OSX without Cocoa
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-7512">[ROOT-7512]</a> - Problem in the JIT of a TFormula when it is retrieved automatically from a file  
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-8204">[ROOT-8204]</a> - rootcp does not preserve name of Key when copying
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-8591">[ROOT-8591]</a> - http files should not be installed into /etc/root/
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-8705">[ROOT-8705]</a> - Dependency issue in roottest/root/io/transient/base
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-8741">[ROOT-8741]</a> - CMake fails to check out roottest
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-8806">[ROOT-8806]</a> - CMake: libjpeg not found / ignored
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-8844">[ROOT-8844]</a> - Loss of precision in gGeoManager Export
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-8863">[ROOT-8863]</a> - Root include path not working after using gSystem->ChangeDirectory
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-8897">[ROOT-8897]</a> - cling cannot print non-fully qualified types
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-8937">[ROOT-8937]</a> - TTreeProcessorMT segfaults when constructed from a tree that is not on disk
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-8991">[ROOT-8991]</a> - Cling exports buggy include paths to AcLIC
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9030">[ROOT-9030]</a> - Several notebook tutorials show errors about `rand` redefinition
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9033">[ROOT-9033]</a> - CMake python3 flag ignored.
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9067">[ROOT-9067]</a> - lz4 not installed in lib when builtin_lz4
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9077">[ROOT-9077]</a> - etc/http not sync'ed by the build system
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9099">[ROOT-9099]</a> - System LZ4 is not found by ROOT due to lack of xxhash.h header
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9114">[ROOT-9114]</a> - garbage code generated in dictionary file for typedefs of templates
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9116">[ROOT-9116]</a> - TDF: bad interaction between multi-thread execution and separate output TFile
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9117">[ROOT-9117]</a> - TDF: Reports on Filters booked after the first event loop do not re-trigger the event loop 
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9121">[ROOT-9121]</a> - TDF: rare crash in TRootDS
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9130">[ROOT-9130]</a> - TDF: Float_t branch type is not inferred when jitting 
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9136">[ROOT-9136]</a> - TDF: failure in test-reports
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9142">[ROOT-9142]</a> - [TDF] members of leaflists (e.g. "b.v") are not recognized as valid column names
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9143">[ROOT-9143]</a> - Build system finds wrong versions of Python
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9144">[ROOT-9144]</a> - TGraph::Draw Malfunctioning when using SetRangeUser
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9146">[ROOT-9146]</a> - GetListOfFunctionOverloads() is broken
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9147">[ROOT-9147]</a> - rootcling crashes when compiling ROOT with C++17 and GCC 7.2.0
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9160">[ROOT-9160]</a> - Cannot interrupt interpreter invocation
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9173">[ROOT-9173]</a> - TH1::StatOverflows is not thread safe
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9175">[ROOT-9175]</a> - [Jenkins] Linux 32bit roottest_root_io_filemerger_make roottest_root_tree_cloning_make
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9178">[ROOT-9178]</a> - [Jenkins] testIntegration fails
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9179">[ROOT-9179]</a> - Warnings on zebra.f and hbook.f (rank-1 and scalar)
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9180">[ROOT-9180]</a> - If R components Rcpp and RInside are missing the 'r' option should be disabled
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9185">[ROOT-9185]</a> - rootcling dereferences null pointer
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9186">[ROOT-9186]</a> - cmake configure fails
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9190">[ROOT-9190]</a> - Missing check for GLEW if builtin_glew=off
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9196">[ROOT-9196]</a> - [Jenkins] MakeProject file names too long
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9199">[ROOT-9199]</a> - TDF: improper handling of branches with leaflists
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9201">[ROOT-9201]</a> - Bad placement of x-axis title when drawing histogram
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9208">[ROOT-9208]</a> - Warning: Nonexistent include directory
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9212">[ROOT-9212]</a> - ROOT should not add -fPIC to compilation flags of dependent projects
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9216">[ROOT-9216]</a> - [Jenkins] RooFit build dependency issue?
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9222">[ROOT-9222]</a> - TGraph axis titles not set 
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9223">[ROOT-9223]</a> - afterimage (builtin or auto found) not compiling - broken root compilation
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9232">[ROOT-9232]</a> - [TDF] Entry loss when using Range and multiple actions
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9233">[ROOT-9233]</a> - Simple rootmap file can not be read by ROOT
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9237">[ROOT-9237]</a> - [Jenkins] roottest_python_JupyROOT_cppcompleter_doctest fails on 32bit
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9238">[ROOT-9238]</a> - [pyROOT] Crash when value-printing empty TFile
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9241">[ROOT-9241]</a> - Address of stack memory associated with local variable returned to caller
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9255">[ROOT-9255]</a> - PyROOT crash after canvas.GetListOfPrimitives() call
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9273">[ROOT-9273]</a> - shapes.C.html not found
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9281">[ROOT-9281]</a> - [TDF] Crash when snapshotting trees with friends from multiple threads
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9292">[ROOT-9292]</a> - Minuit2 does not work with OpenMP enabled
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9306">[ROOT-9306]</a> - [Jenkins] Test roottest.root.multicore.roottest_root_multicore_tp_process_imt fails
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9308">[ROOT-9308]</a> - Error on pulling TGraphErrors with fit from root file in command line 
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9315">[ROOT-9315]</a> - cmake should not search for ftgl when opengl is switched off
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9318">[ROOT-9318]</a> - TFileCacheRead over reading
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9319">[ROOT-9319]</a> - [TTreeReader] main tree branch instead of friend tree branch accessed when branches have the same name
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9322">[ROOT-9322]</a> - TTreeReaderValue<double> reads a double[] branch by only accessing the first element with no warning
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9325">[ROOT-9325]</a> - CMAKE_BUILD_TYPE should default to RelWithDebInfo again
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9337">[ROOT-9337]</a> - TTree::Draw not able to call 'std::vector::at'
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9338">[ROOT-9338]</a> - When merging two TTree each with a single autoflush/clustering range, the cluster size of one the file is lost.
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9339">[ROOT-9339]</a> - Enhance TTree Cache loading algorithm when reading badly clustered files.
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9344">[ROOT-9344]</a> - TChain does not recognize folder ending in .root
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9359">[ROOT-9359]</a> - Update external / builtin openssl
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9363">[ROOT-9363]</a> - Enum Autoload updates in root master broke cmssw (since Dec 2017)
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9364">[ROOT-9364]</a> - Error importing gVirtualX
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9369">[ROOT-9369]</a> - A couple issues involving SetMustClean
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9393">[ROOT-9393]</a> - hadd crashes when recompressing certain input files
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9396">[ROOT-9396]</a> - Compilation error with GCC 8.1.1
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9410">[ROOT-9410]</a> - roottest_root_tree_cloning_assertBranchCount fails sporadically
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9418">[ROOT-9418]</a> - [TDF] Cannot use Range before a Define with a string expression
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9441">[ROOT-9441]</a> - Configure broken on Ubuntu 17
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9452">[ROOT-9452]</a> - [DF] Exception thrown when calling jitted snapshot on an aliased column
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9454">[ROOT-9454]</a> - [Jenkins] Frequent timeout in tp_process_imt
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9373">[ROOT-9373]</a> - unresolved while linking [cling interface function] - (with ROOT 7 graphics)
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9125">[ROOT-9125]</a> - Making TEnv::GetValue and Lookup const
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9437">[ROOT-9437]</a> - Improve the processing of a TChain with many files in TTreeProcessorMT
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-8122">[ROOT-8122]</a> - Terminate interpreter in clean way if EOT / Ctrl+D is read and current line is empty
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9270">[ROOT-9270]</a> - [TDF] If the user instructs Snapshot to write "mybranch.myleaf" chose as branch name "mybranch_myleaf""
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9231">[ROOT-9231]</a> - TMVA GUI produces identical results for variants of models used in RMVA
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-8806">[ROOT-8806]</a> - CMake: libjpeg not found / ignored
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9295">[ROOT-9295]</a> - [TDF] Variable sized bins are not respected by Histo actions
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9272">[ROOT-9272]</a> - [TDF] Entry loss executing multiple event loops on the same Range'd TDataFrame
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9305">[ROOT-9305]</a> - [Jenkins] Test root/dataframe/test_gdirectoryRestore fails on Ubuntu 14.04 with GCC 4.8
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-8062">[ROOT-8062]</a> - Add INTERFACE_INCLUDE_DIRECTORIES and namespacing to exported/imported CMake targets
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-7863">[ROOT-7863]</a> - Build real ZLIB when builtin_zlib is turned on 



## Release 6.14/02

Released on July 27, 2018

### TTree Libraries

* Enhanced the scheduling of I/O customization rules in split TTree to support many additional cases in the presence of splitting.  Rules that do not correspondant to a end branch (i.e. do not target a persistent member) are now scheduled as part of the parent branch (In the previous implementation, in many cases, thoses kind of rules were never run)

### Bugs and Issues fixed in this release

* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9456">[ROOT-9456]</a> - [Jenkins][DF] Sporadic failures in test_snapshotNFiles
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9471">[ROOT-9471]</a> - [DF] Snapshot does not write any entry if many workers have zero entries to write
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9466">[ROOT-9466]</a> - [DF] A lazy snapshot that's never triggered crashes at teardown
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9472">[ROOT-9472]</a> - GCC 8 Error: call of overloaded 'basic_string(TString)' is ambiguous
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9478">[ROOT-9478]</a> - Compilation failure of version 6.14
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9479">[ROOT-9479]</a> - Build failure: value of type 'void' is not contextually convertible to 'bool'
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9503">[ROOT-9503]</a> - $ROOTSYS/tutorials/http/httpserver.C only shows blank page
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9505">[ROOT-9505]</a> - Template forward declaration is missing in the .rootmap
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9526">[ROOT-9526]</a> - [DF] RResultPtrs cannot be copy- or move-assigned
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9532">[ROOT-9532]</a> - Built-in OpenSSL results in non-relocatable ROOT installation
* <a href="https://sft.its.cern.ch/jira/browse/ROOT-9550">[ROOT-9550]</a> - last ROOT versions don't support PostgreSQL less than 10.x



## Release 6.14/04

Released on August 23, 2018

### I/O

* **Default compression algorithm was reverted to zlib**.
* Avoid double search of elements in TBranchElement.
* Additional fix for rule scheduling in TBranchElement.
* Avoid getting confused by out-of-range request to TTreeCache

### Core

* Improve thread scability of TRef. Creating and looking up a lot of TRef from the same processID now has practically perfect weak scaling.

### RDataFrame

* Support unsigned integer types in RArrowDS.


### TGeo

* Extended return value of operator* to TGeoHMatrix. Added Multiply taking const reference.
* Added protection for ComputeLradTsaiFactor when Z=0.
* Consistency for copy constructors and assignment.

### Build system

* Recovery of correct CMake invocation of rootcling.
* Use NEW behavior for CMake policy CMP0051.
* Require same version of CMake as ROOT in ROOTUseFile.cmake.
* Clear cache and set $DAVIX_FOUND when using builtin Davix.


### Bugs and Issues fixed in this release

* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9521'>ROOT-9521</a>] -         Issue when building ROOT with Python 3.7
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9529'>ROOT-9529</a>] -         TBits::operator~() must be const
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9555'>ROOT-9555</a>] -         [DF] Compilation fails for Reduce on a bool column due to std::vector&lt;bool&gt;
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9630'>ROOT-9630</a>] -         Arrow RDS and lack of int32 support


## Release 6.14/06

Released on November 5, 2018

### Platforms

MacOS 10.14 / Xcode 10 is supported.

### I/O

* To allow for increase run-time performance and increase thread scalability the override ability of `TFile::GetStreamerInfoList` is replaced by an override of `TFile::GetStreamerInfoListImp` with updated return type and arguments.   If a class override `TFile::GetStreamerInfoList` you will now see a compilation error like:


```
include/TSQLFile.h:225:19: error: declaration of 'GetStreamerInfoList' overrides a 'final' function
virtual TList *GetStreamerInfoList();
^
include/TFile.h:231:24: note: overridden virtual function is here
virtual TList      *GetStreamerInfoList() final; // Note: to override behavior, please override GetStreamerInfoListImpl
^
```

Instead you need to override the protected method:

```
InfoListRet GetStreamerInfoListImpl(bool lookupSICache);
```

which can be implemented as

```
InfoListRet DerivedClass::GetStreamerInfoListImpl(bool /*lookupSICache*/) {
ROOT::Internal::RConcurrentHashColl::HashValue hash;
TList *infolist = nullptr;
//
// Body of the former Derived::GetStreamerInfoList with the
// return statement replaced with something like:

// The second element indicates success or failure of the load.
// (i.e. {nullptr, 0, hash} indicates the list has already been processed
//  {nullptr, 1, hash} indicates the list failed to be loaded
return {infolist, 0, hash};
}
```

See `TFile::GetStreamerInfoListImpl` implementation for an example on how to implement the caching.

### RDataFrame
* throw if name of defined column is not a valid C++ name


### Bugs and Issues fixed in this release

* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9453'>ROOT-9453</a>] -         [VecOps] Cannot instantiate RVec&lt;bool&gt;
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-5236'>ROOT-5236</a>] -         Memory Leak in RooFit/HistFactory
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9209'>ROOT-9209</a>] -         Crash in cling::Value destruction
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9214'>ROOT-9214</a>] -         [Jenkins] TMVA_DNN_Arithmetic_Cpu fails sporadically
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9331'>ROOT-9331</a>] -         rootcint doesn&#39;t return with an error code when encountering a syntax error
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9467'>ROOT-9467</a>] -         TF1: Error when reading two functions with the same formula
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9473'>ROOT-9473</a>] -         FatalRootError after updating ROOT master in CMs IBs
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9475'>ROOT-9475</a>] -         ROOT / Windows startup warnings
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9506'>ROOT-9506</a>] -         [DF] RDataFrame filters cannot be cast to a common type, TDataFrame filters could
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9513'>ROOT-9513</a>] -         Data member not saved due to no streamer or dictionary 
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9542'>ROOT-9542</a>] -         RDataFrame Sum gets confused by std::string
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9563'>ROOT-9563</a>] -         [TreeProcMT] Trees in subdirectories are not supported (and their usage lead to a crash)
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9574'>ROOT-9574</a>] -         TTree::TClusterIterator::GetEstimatedClusterSize broken for ROOT File with clustering disabled.
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9624'>ROOT-9624</a>] -         TMySQLStatement.h incompatible with MySQL 8
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9632'>ROOT-9632</a>] -         ROOT::RDataFrame::Define struggles with user-defined return types when built stand-alone
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9640'>ROOT-9640</a>] -         Segmentation violation after starting TBrowser b
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9662'>ROOT-9662</a>] -         Thread problem in I/O for ROOT master
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9670'>ROOT-9670</a>] -         Cling problems with TFormula and std::math functions
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9681'>ROOT-9681</a>] -         Typedef clash in global namespace between ROOT and Geant4
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9684'>ROOT-9684</a>] -         Wrong cast of functions in TGlobalMappedFunction
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9694'>ROOT-9694</a>] -         Infinite recursion in GetStreamerInfoList when opening corrupt file.
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9701'>ROOT-9701</a>] -         IO read rules no longer executed




## HEAD of the v6-14-00-patches branch

These changes will be part of the future 6.14/08


