% ROOT Version 6.12 Release Notes
% 2017-05-18

<a name="TopOfPage"></a>

## Introduction

ROOT version 6.12/00 is scheduled for release in 2018.

For more information, see:

[http://root.cern.ch](http://root.cern.ch)

The following people have contributed to this new version:

 Guilherme Amadio, CERN/SFT,\
 Bertrand Bellenot, CERN/SFT,\
 Brian Bockelman, UNL,\
 Rene Brun, CERN/SFT,\
 Philippe Canal, FNAL,\
 Olivier Couet, CERN/SFT,\
 Gerri Ganis, CERN/SFT,\
 Andrei Gheata, CERN/SFT,\
 Enrico Guiraud, CERN/SFT,\
 Sergey Linev, GSI,\
 Pere Mato, CERN/SFT,\
 Lorenzo Moneta, CERN/SFT,\
 Axel Naumann, CERN/SFT,\
 Danilo Piparo, CERN/SFT,\
 Fons Rademakers, CERN/SFT,\
 Enric Tejedor Saavedra, CERN/SFT,\
 Vassil Vassilev, Fermilab/CMS,\
 Wouter Verkerke, NIKHEF/Atlas, RooFit

## Removed interfaces

The following interfaces have been removed, after deprecation in v6.10.

- Remove the deprecated `TSelectorCint.h` and `TSelectorCint.cxx`.
- Remove the deprecated `Riosfwd.h` and `Rtypeinfo.h`.


### TTreeReader

`TTreeReader::SetLastEntry()` was replaced by `TTreeReader::SetEntriesRange()`.



## Core Libraries

- When invoking root with the "-t" argument, ROOT enables thread-safety and,
  if configured, implicit multithreading within ROOT.
- `NULL` is not defined by `Rtypes.h` anymore. Instead, its definition is expected to be
  provided by `Rtype.h`'s `#include` of `stddef.h`.
- ROOT now supports dictionaries, autoload and autoparse for classes with template parameter packs.
- std::make_unique has been backported


## I/O Libraries

- Introduce TKey::ReadObject<typeName>.  This is a user friendly wrapper around ReadObjectAny.  For example
```{.cpp}
auto h1 = key->ReadObject<TH1>
```
after which h1 will either be null if the key contains something that is not a TH1 (or derived class)
or will be set to the address of the histogram read from the file.

- Added a new mechanism for providing clean forward-compatibility breaks in a ``TTree`` (i.e., a newer version of ROOT writes a ``TTree`` an older version cannot read).  When future versions of ROOT utilize an IO feature that this version does not support, ROOT will provide a clear error message instead of crashing or returning garbage data.  In future ROOT6 releases, forward-compatibility breaks will only be allowed if a non-default feature is enabled via the ``ROOT::Experimental`` namespace; it is expected ROOT7 will enable forward-compatibility breaks by default.

   - When a file using an unsupported file format feature is encountered, the error message will be similar to the following:
      ```
      Error in <TBasket::Streamer>: The value of fIOBits (00000000000000000000000001111110) contains unknown flags (supported flags are 00000000000000000000000000000001), indicating this was written with a newer version of ROOT utilizing critical IO features this version of ROOT does not support.  Refusing to deserialize.
      ```
   - When an older version of ROOT, without this logic, encounters the file, the error message will be similar to the following:
      ```
      Error in <TBasket::Streamer>: The value of fNevBufSize is incorrect (-72) ; trying to recover by setting it to zero
      ```

## TTree Libraries

- Resolved O(N^2) scaling problem in ```TTree::Draw()``` observed when a branch that contains a
large TClonesArray where each element contains another small vector container.
- `TTree::TTree()` now takes the `TDirectory*` that the tree should be constructed in.
  Defaults to `gDirectory`, i.e. the default behavior did not change.

### TDataFrame
  - Improved documentation
  - Fixed race condition: concurrent deletion of TTreeReader/TTreeReaderValue
  - TDF now avoids performing virtual calls for parts of the analysis that are not jitted
  - Improved checks for column name validity (throw if column does not exist and if `Define`d column overrides an already existing column)
  - Removed "custom column" nodes from the functional graph therewith optimising the traversal
  - Added `DefineSlot`, a `Define` transformation that is aware of the multi-threading slot where the workload is executed
  - Improvements in Cling drastically enhanced scaling and performance of TDF jitted code
  - Fixed reading of c-style arrays from jitted transformations and actions
  - pyROOT users can now easily specify parameters for the TDF histograms thanks to the newly introduced tuple-initialization
  - The new TDataSource interface allows developers to pipe any kind of columnar data format into TDataFrame
  - Test coverage has been increased with the introduction of google tests
  - Users can now configure Snapshot to use different file open modes ("RECREATE" or "UPDATE"), compression level, compression algrotihm, TTree split-level and autoflush settings
  - Python tutorials show the new "tuple-initialisation" feature of PyROOT (see below)
  - The possibility to read from data sources was added. An interface for all data sources, TDataSource, is provided by ROOT. Two example data sources have been provided too: the TRootDS and the TTrivialDS. The former allows to read via the novel data source mechanism ROOT data, while the latter is a simple generator, created for testing and didactic purposes. It is therefore now possible to interface *any* kind of dataset/data format to ROOT as long as an adaptor which implements the pure virtual methods of the TDataSource interface can be written in C++.

## Histogram Libraries


## Math Libraries


## RooFit Libraries


## 2D Graphics Libraries
  - The method TColor::InvertPalette inverts the current palette. The top color becomes
    bottom and vice versa. This was [suggested by Karl Smith](https://root-forum.cern.ch/t/inverted-color-palettes/24826/2).
  - New method `TColor::SetColorThreshold(Float_t t)` to specify the color
    threshold used by GetColor to retrieve a color.
  - Improvements in candle plots:
    -  LogZ for violins
    -  scaling of candles and violins with respect to each other
    -  static functions for WhiskerRange and BoxRange
  - In some case it was not possible to zoom a 1D histogram using the mouse
    on the X axis. This was described
    [here](https://root-forum.cern.ch/t/axis-blocked-when-overlaying-two-histograms/25326)
  - When drawing an histogram with option "BOX" with log scale along the Z axis
    the bins were not visible in some case. This was described
    [here](https://root-forum.cern.ch/t/set-logscale-on-z-axis-in-2d-histo/25385).
  - When a TGraph2D was plotted with the option "PCOLZ" with a log scale along the
    Z axis, there was a mismatch between the markers' colors and the color palette
    displayed. It is now fixed. It was reported
    [here](https://sft.its.cern.ch/jira/browse/ROOT-8200).
  - It is now possible to set the titles and the axis ranges of a TMultiGraph drawn as 3D lines.
  - Implement the option "Z" (to draw the palette) for 3D histograms drawn with
    the option "BOX2".
  - With the option `HBAR` the histogram grid was painted over the stat box.
  - The `TGraph`'s options "F" and "L" respectively draw a filled polygon and
    a line plot. They can be combined when calling `TGraph::Draw`. Doing that
    produced a filled polygon only. Now it produces a filled polygon and a line plot.
  - `TH1::SetOption()` method didn't work when called from `TH3D` instance.

## 3D Graphics Libraries


## Geometry Libraries


## Database Libraries


## Networking Libraries


## GUI Libraries


## Montecarlo Libraries


## Parallelism
  - Fix issue which prevented nested TBB task execution without race conditions, e.g. in TDataFrame
  - Fix race condition in TTreeProcessorMT due to TBB nested task execution
  - The TTaskGroup class has been added to the ROOT::Experimental namespace. It allows to submit to the runtime
  item of work which are dealt with in parallel;
  - The Async template function has been added the ROOT::Experimental namespace. The template function is analogous
  to *std::async* but without the possibility of specifying the execution policy and without creating a thread but
  directly submitting the work to the runtime in order to use the same pool as any other item of work spawned by ROOT.
  - The TFuture template has been added to the ROOT::Experimental namespace. It represents a future and is compatible
  with the ROOT::Experimental::Async function. It has the same properties of an STL future and can be initialised by
  one of these classes. For example, *TFuture<int> = std::async(myfunc,a,b,c);*


## Language Bindings
  - PyROOT now supports list initialisation with tuples. For example, suppose to have a function `void f(const TH1F& h)`. In C++, this can be invoked with this syntax: `f({"name", "title", 64, -4, 4})`. In PyROOT this translates too `f(('name', 'title', 64, -4, 4))`.


## JavaScript ROOT


## Tutorials

## Command line tools
  - `rootls` has been extended.
    - option `-l` displays the year
    - option `-t` displays all details of 'THnSparse'


## Class Reference Guide
  - The list of libraries needed by each class is displayed as a diagram.

## Build, Configuration and Testing Infrastructure


