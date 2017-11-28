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
- If a class overloads TObject::Hash, this derived class should also add
```{.cpp}
   ROOT::CallRecursiveRemoveIfNeeded(*this)
```
Otherwise, when RecursiveRemove is called (by ~TObject or example) for this
type of object, the transversal of THashList and THashTable containers will
will have to be done without call Hash (and hence be linear rather than
logarithmic complexity).  You will also see warnings like
```
   Error in <ROOT::Internal::TCheckHashRecurveRemoveConsistency::CheckRecursiveRemove>: The class SomeName overrides TObject::Hash but does not call TROOT::RecursiveRemove in its destructor.
```
- When a container relies on TObject::Hash and RecursiveRemove, for example THashTable, the container call use ```TObject::CheckedHash()``` instead of ```TObject::Hash``` during insertion operation to record in the object whether the Hash/RecursiveRemove setup is done properly (as explain above).  It this is not the case ```TObject::HasInconsistentHash()``` will return true.  This can then be used to select, in RecursiveRemove, whether the call to Hash can be trusted or if one needs to do a linear search (as was done in v6.10 and earlier).


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

- Added an experimental feature that allows the IO libraries to skip writing out redundant information for some
   split classes, resulting in disk space savings.  This is disabled by default and may be enabled by setting:
   ```
   ROOT::TIOFeatures features;
   features.Set(ROOT::Experimental::EIOFeatures::kGenerateOffsetMap);
   ttree_ref.SetIOFeatures(features);
   ```

## TTree Libraries

- Resolved O(N^2) scaling problem in ```TTree::Draw()``` observed when a branch that contains a
large TClonesArray where each element contains another small vector container.
- `TTree::TTree()` now takes the `TDirectory*` that the tree should be constructed in.
  Defaults to `gDirectory`, i.e. the default behavior did not change.
- To prepare for multi-threaded workflows, a preloading and retaining clusters feature is introduced.
  This change will prevent additional reads from occurring when reading events out of sequence.
  By setting TTree::SetClusterPrefetch(), an entire clusters will be loaded into memory, rather than single baskets.
  By setting the MaxVirtualSize of the tree to a negative value, previous clusters will be retained
  (the absolute value of MaxVirtualSize indicates how many additional clusters will be kept in memory).

### TDataFrame

#### New features
  - Add `Alias`, a facility to specify an alternative name for a given column: `auto histo = mytdf.Alias("myAlias", "myColumn").Histo1D("myAlias");`. Especially useful for pyROOT users to deal with column names that are not valid C++ identifiers (e.g. `Filter("1branch > 0") --> Alias("1branch", "branch1").Filter("branch1 > 0")`.
  - Add `Cache`, a facility to cache `TDataFrame`s in memory. All or some columns can be cached. Two versions of the method are proposed: one which allows to explicitly list the types of the columns and another one allowing to let the system infer them (the same mechanism of the `Snapshot` method). Only columns containing instances of classes which have a copy constructor can be cached.
  - Add `DefineSlot`, a `Define` transformation that is aware of the multi-threading slot where the workload is executed
  - Add `DefineSlotEntry`, a `Define` transformation that is aware of the multi-threading slot and of the current entry number
  - Add `GetColumnsNames`: users can now get the names of the available columns coming from trees, data sources or `Define`d columns
  - Add `OnPartialResult` and `OnPartialResultSlot`: users can now register one or more functions to be executed on partial results of TDF actions during the event loop.
    This mechanism is meant to be used to inspect partial results of the analysis or print useful debug information.
    For example, both in single- and multi-thread event loops, one can draw a result histogram and update the canvas every 100 entries like this:
    ```c++
    auto h = tdf.Histo1D("x");
    TCanvas c("c","x hist");
    h.OnPartialResult(100, [&c](TH1D &h_) { c.cd(); h_.Draw(); c.Update(); });
    ```
    See the tutorials for more examples.
  - Add `Sum`, an action that sums all values of a column for the processed entries
  - The new TDataSource interface allows developers to pipe any kind of columnar data format into TDataFrame. Two example data sources have been provided: the TRootDS and the TTrivialDS. The former allows to read via the novel data source mechanism ROOT data, while the latter is a simple generator, created for testing and didactic purposes. It is therefore now possible to interface *any* kind of dataset/data format to ROOT as long as an adaptor which implements the pure virtual methods of the TDataSource interface can be written in C++.
  - TDF can now read CSV files through a specialized TDataSource. Just create the TDF with `MakeCsvDataFrame("f.csv")`. Just create the TDF with MakeCsvDataFrame("f.csv"). The data types of the CSV columns are automatically inferred. You can also specify if you want to use a different delimiter or if your file does not have headers.
  - Users can now configure Snapshot to use different file open modes ("RECREATE" or "UPDATE"), compression level, compression algorithm, TTree split-level and autoflush settings
  - Users can now access multi-threading slot and entry number as pre-defined columns "tdfslot_" and "tdfentry_". Especially useful for pyROOT users.
  - Users can now specify filters and definitions as strings containing multiple C++ expressions, e.g. "static int a = 0; return ++a". Especially useful for pyROOT users.
  - pyROOT users can now easily specify parameters for the TDF histograms and profiles thanks to the newly introduced tuple-initialization
  - Add support for friend trees and chains. Just add the friends before passing the tree/chain to TDataFrame's constructor and refer to friend branches as usual.

#### Fixes
  - Fixed race condition: concurrent deletion of TTreeReader/TTreeReaderValue
  - Fixed reading of c-style arrays from jitted transformations and actions
  - Fixed writing of c-style arrays with `Snapshot`
  - Improved checks for column name validity (throw if column does not exist and if `Define`d column overrides an already existing column)

#### Other changes
  - Improved documentation
  - TDF now avoids performing virtual calls for parts of the analysis that are not jitted
  - Removed "custom column" nodes from the internal functional graph therewith optimising its traversal
  - Improvements in Cling drastically enhanced scaling and performance of TDF jitted code
  - Test coverage has been increased with the introduction of google tests
  - Interface change: users must now use TDF::TArrayBranch rather than std::array\_view to specify that the column being read is a c-style array TTree branch
  - Interface change: `Min` and `Max` now return results as the same type specified as template parameter, or double if no template parameter was specified


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
  - With the Cocoa backend on Mac the png image were truncated when ROOT was running
    in a small screen attached to the Mac with the graphics window on the Mac
    display. It was reported
    [here](https://root-forum.cern.ch/t/tcanvas-print-png-outputs-fraction-of-figure-when-canvas-size-is-declared/26011/44).
  - Fix an issue with `TGraph2D` drawn as lines (reported [here](https://sft.its.cern.ch/jira/browse/ROOT-9046)).
  - ROOT Cocoa: fix rendering into bitmaps on high-dpi display. With retina display
    the rendering of polylines was slow.
  - Fix a precision issue in `TGraph2D`. It was reported [here](https://root-forum.cern.ch/t/tgraph2d-plotting-issues/26562)
  - New method `TGraph::InsertPointBefore(Int_t ipoint, Double_t x, Double_t y)`
    to insert a new point with coordinates (x,y) before the point number `ipoint`.
  - When a 2D histogram was drawn with error bars and has a function in its list
    it was impossible to rotate it interactively. This problem was reported
    [here](https://root-forum.cern.ch/t/2d-histogram-fit-draws-to-wrong-scale/26369).
  - As more and more people are using `TGraph2D` for random cloud of points, the default
    drawing option implying Delaunay triangulation was not appropriate. The default
    drawing option is now change to `P0`.
  - It is now possible to set the value of `MaxDigits` on individual axis as
    requested [here](https://sft.its.cern.ch/jira/browse/ROOT-35).
    For example, to accept 6 digits number like 900000 on the X axis of the
    histogram `h` call:
```{.cpp}
    h->GetXaxis()->SetMaxDigits(6);
```

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


