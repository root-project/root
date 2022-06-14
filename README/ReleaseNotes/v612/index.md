% ROOT Version 6.12 Release Notes
% 2017-05-18

<a name="TopOfPage"></a>

## Introduction

ROOT version 6.12/00 is scheduled for release in 2018.

For more information, see:

[http://root.cern.ch](http://root.cern.ch)

The following people have contributed to this new version:

 Kim Albertsson, CERN,\
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
 Raphael Isemann, Chalmers Univ. of Tech.,\
 Sergey Linev, GSI,\
 Timur Pocheptsov, CERN/SFT,\
 Pere Mato, CERN/SFT,\
 Lorenzo Moneta, CERN/SFT,\
 Axel Naumann, CERN/SFT,\
 Simon Pfreundschuh,\
 Danilo Piparo, CERN/SFT,\
 Fons Rademakers, CERN/SFT,\
 Enric Tejedor Saavedra, CERN/SFT,\
 Oksana Shadura, UNL,\
 Arthur Tsang, CERN/SFT, \
 Peter van Gemmeren, ANL,\
 Vassil Vassilev, Princeton Univ./CMS,\
 Xavier Valls Pla, CERN/UJI, \
 Wouter Verkerke, NIKHEF/Atlas, RooFit,\
 Stefan Wunsch, KIT,\
 Omar Zapata

## General News

This release now supports building with C++17 enabled using either libstdc++ or
libc++. This requires Clang >= 5.0, or GCC >= 7.3.0. At the date of this
release, GCC 7.2.0 still does not provide full support to compile ROOT with C++17.

## Removed interfaces

The following interfaces have been removed, after deprecation in v6.10.

- Remove the deprecated `TSelectorCint.h` and `TSelectorCint.cxx`.
- Remove the deprecated `Riosfwd.h` and `Rtypeinfo.h`.
- `TTreeReader::SetLastEntry()` was replaced by `TTreeReader::SetEntriesRange()`.



## Core Libraries

- Added support for XCode 9 and MacOS High Sierra.
- When invoking root with the "-t" argument, ROOT enables thread-safety and,
  if configured, implicit multithreading within ROOT.
- `NULL` is not defined by `Rtypes.h` anymore. Instead, its definition is expected to be
  provided by `Rtype.h`'s `#include` of `stddef.h`.
- ROOT now supports dictionaries, autoload and autoparse for classes with template parameter packs.
- std::make_unique has been backported
- If a class overloads TObject::Hash, this derived class should also add
```
   ROOT::CallRecursiveRemoveIfNeeded(*this)
```
Otherwise, when RecursiveRemove is called (by ~TObject or example) for this
type of object, the transversal of THashList and THashTable containers will
will have to be done without call Hash (and hence be linear rather than
logarithmic complexity).  You will also see warnings like
```
   Error in <ROOT::Internal::TCheckHashRecursiveRemoveConsistency::CheckRecursiveRemove>: The class SomeName overrides TObject::Hash but does not call TROOT::RecursiveRemove in its destructor.
```
- When a container relies on TObject::Hash and RecursiveRemove, for example THashTable, the container uses ```TObject::CheckedHash()``` instead of ```TObject::Hash``` during insertion operation to record in the object whether the Hash/RecursiveRemove setup is done properly (as explain above).  It this is not the case ```TObject::HasInconsistentHash()``` will return true.  This can then be used to select, in RecursiveRemove, whether the call to Hash can be trusted or if one needs to do a linear search (as was done in v6.10 and earlier).
- In TClass::GetMissingDictionaries activate the search through the base classes.
- Added a TStatusBitsChecker to avoid Status Bits overlap in class hierarchy deriving from TObject (and resolved a handful of conflicts).
- Introduced support for type safe range-for-loop for ROOT collection. The typical use is:

```
   for(auto bcl : TRangeDynCast<TBaseClass>( * cl->GetListOfBases() )) {
     if (!bcl) continue;
     ... use bcl as a TBaseClass*
   }
   for(auto bcl : TRangeDynCast<TBaseClass>( cl->GetListOfBases() )) {
      if (!bcl) continue;
      ... use bcl as a TBaseClass*
   }
```
- ClassDefInline has been enhanced even for some compiled class (without a dictionary).  ClassDefInline can still not be used for class template instance using Double32_t or Float16_t as a template parameter or for class or class template that do not have a public default constructor.
- ROOT's backport of `std::string_view` has been updated to follow what's available in C++17, notably its `to_string` member function has been removed.


### Thread safety

Resolved the race conditions inherent to the use of the RecursiveRemove mechanism.

- Introduced ```ROOT::TReentrantRWLock```, an implementation of a reentrant read-write lock with a configurable internal mutex/lock and a condition variable to synchronize readers and writers when necessary.

   The implementation allows a single reader to take the write lock without releasing the reader lock.  It also allows the writer to take a read lock. In other word, the lock is re-entrant for both reading and writing.

   The implementation tries to make faster the scenario when readers come and go but there is no writer. In that case, readers will not pay the price of taking the internal lock.
Moreover, this RW lock tries to be fair with writers, giving them the possibility to claim the lock and wait for only the remaining readers, thus preventing starvation.

- Switched the ROOT global to be a ```ROOT::TReentrantRWLock``` and renamed it ROOT::gCoreMutex.  The old name ```gROOTMutex``` and ```gInterpreterMutex``` are deprecated and may be removed in future releases.
- Added ```TReadLockGuard```,```TWriteLockGuard```, ```R__READ_LOCKGUARD``` and```R__WRITE_LOCKGUARD``` to take advantage of the new lock.  The legacy ```TLockGuard``` and ```R__LOCKGUARD``` use the write lock.
- Improved scaling of TROOT::RecursiveRemove in the case of large collection.
- Added a thread safe mode for the following ROOT collections: THashList, THashTable, TList and TObjArray.  When ROOT's thread safe mode is enabled and the collection is set to use internal locks by calling:
```
  collection->UseRWLock();
```
all operations on the collection will take the read or write lock when needed, currently they shared the global lock (ROOT::gCoreMutex).

### Interpreter

- cling's LLVM is upgraded to version 5.0
- All of cling's patches to llvm have been upstreamed.
- The interpreter-related lock is now locking only the compilation step, not the execution step. This reduces the scope for lock contention. Most significantly, it enables the use of concurrency on the prompt!


## I/O Libraries

- Introduce TKey::ReadObject<typeName>.  This is a user friendly wrapper around ReadObjectAny.  For example
```
auto h1 = key->ReadObject<TH1>
```
after which h1 will either be null if the key contains something that is not a TH1 (or derived class)
or will be set to the address of the histogram read from the file.
- Add the ability to store the 'same' object several time (assumingly with different data) in a single buffer.  Instead of

```
  while(...) {
     TObjArray arr;
     ... update the content of "arr"
     buffer << arr;
  }
```
which would only really stream the array at the first iteration because it will be detected has having the same address and thus assumed to be the same object.  We can now do:
```
  while(...) {
     TObjArray arr;
     ... update the content of "arr"
     buffer.WriteObject(&arr, kFALSE);
  }
```
where the last argument of WriteObject tells the buffer do *not* remember this object's address and to always stream it.  This feature is also available via WriteObjectAny.

- Added a new mechanism for providing clean forward-compatibility breaks in a ``TTree`` (i.e., a newer version of ROOT writes a ``TTree`` an older version cannot read).  When future versions of ROOT utilize an IO feature that this version does not support, ROOT will provide a clear error message instead of crashing or returning garbage data.  In future ROOT6 releases, forward-compatibility breaks will only be allowed if a non-default feature is enabled via the ``ROOT::Experimental`` namespace; it is expected ROOT7 will enable forward-compatibility breaks by default.

   - When a file using an unsupported file format feature is encountered, the error message will be similar to the following:
      ```
      Error in <TBasket::Streamer>: The value of fIOBits (00000000000000000000000001111110) contains unknown flags (supported flags are 00000000000000000000000000000001), indicating this was written with a newer version of ROOT utilizing critical IO features this version of ROOT does not support.  Refusing to deserialize.
      ```
   - When an older version of ROOT, without this logic, encounters the file, the error message will be similar to the following:
      ```
      Error in <TBasket::Streamer>: The value of fNevBufSize is incorrect (-72) ; trying to recover by setting it to zero
      ```

- Added an experimental feature that allows the IO libraries to skip writing out redundant information for some split classes, resulting in disk space savings.  This is disabled by default and may be enabled by setting:

   ```
   ROOT::TIOFeatures features;
   features.Set(ROOT::Experimental::EIOFeatures::kGenerateOffsetMap);
   ttree_ref.SetIOFeatures(features);
   ```
- Added `GetAutoSave()` and `SetAutoSave()` methods to `TBufferMerger`, to allow
  it to accumulate several buffers in memory before merging, to reduce the
  amount of compression work done due to `TTree` metadata.

- Added a non-blocking callback mechanism to `TBufferMerger` to allow users to
  control the rate at which data is pushed into the merging queue. The callback
  mechanism can be used, for example, to launch tasks asynchronously whenever a
  buffer is done processing.

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
- Added ```TBranchProxy::GetEntries``` to support leaflist variable size array and added ```TBranchProxy::GetArrayLength```.
- In ```TBranch::Streamer``` insured that we never steam already basket already written to disk.

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
  - Histograms can be initialised by *models*, which allow to create histograms with the same parameters of their constructors, for example
  ```c++
  auto myHisto = myTdf.Histo1D({"histName", "histTitle", 64, 0, 128}, "myColumn");
  ```
  or
  ```c++
  auto myHistoCustomBinning = myTdf.Histo1D({"histName", "histTitle", 64, binEdges}, "myColumn");
  ```
  Models can be created as stand alone objects:
  ```c++
  TDF::TH1DModel myModel {"histName", "histTitle", 64, binEdges};
  auto myHistoCustomBinning = myTdf.Histo1D(myModel, "myColumn");
  ```
  - pyROOT users can now easily specify parameters for the TDF histograms and profiles thanks to the newly introduced tuple-initialization
  ```python
  myHisto = myTdf.Histo1D(('histName', 'histTitle', 64, 0, 128), 'myColumn')
  ```
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
- Histogram-based fits are implicitly parallelized.
- Added new options to the histogram fitting interfaces to support explicit parallelization of the fit as well.
- `TF1` gradient evaluation supports vectorization.
- Refactor of `TF1` constructors, default initialization of its data members and fixed ambiguous TF1::operator().
- Extend `TFormula` parsing capabilities.
  - The parsing of arguments for defining parametric function is improved. For example a Gaussian function in y can be defined as `gaus( y , [A], [Mean], [Sigma])`.
  - One can define the function variables or parameters using another function or an expression. Example: `gaus(x, [A], [m0]*y+[m1], [sigma])`.
  - Support for function composition in `TFormula`, i.e. a function can be composed from another function, Again, an example: `gaus( f1(x), [A],[Mean],[Sigma])`, where `f1` is a function defined
  previously.
- Facilitate using Normalized sums of TF1 objects and convolutions, by adding the `NSUM` and `CONV` operators for TF1 objects built with formula expressions
  - `TF1("model", "NSUM(gaus , expo)", xmin, xmax)`  will create a function composed of a normalized sum of a gaussian and an exponential.
  - `TF1("voigt", "CONV(breitwigner, gausn) , -15, 15)` will create a TF1 object made of a convolution between a Breit-Wigner and a Gaussian. 
- `TFormula` supports vectorization. All the `TF1` objected created with a formula expression can have a vectorized signature using `ROOT::Double_v`: `TF1::EvalPar( ROOT::Double_v * x,
double * p)`. The vectorization can then be used to speed-up fitting. It is not enabled by default, but it can be enabled by callig  `TF1::SetVectorized(true)` or using the `"VEC"` option in the
constructor of TF1, when ROOT has been built with VecCore and one vectorization library such as Vc. 
- Added new auto-binning algorithm, referred to as `power-2`, which uses power of 2 bin widths to create bins
  that are mergeable. The target use-case is support for auto-binning in multi-process or multi-thread execution,
  e.g. `TDataFrame`, without the need of a synchronization point.
  The new `power-2` algorithm is activated by setting the new `TH1::kAutoBinPTwo` status bit on the histogram.
  The tutorial `tutorials/multicore/mt304_fillHistos.C` gives an example of how to use the functionality with
  `TThreadedObject<TH1D>` . The `power-2` binning is currently available only for 1D histograms.


## Math Libraries
 - The Fitting functions now support vectorization and parallelization.
 - Added padding in the fit data classes for correct loading of SIMD arrays.


## RooFit Libraries

- Apply several fixes from the ATLAS Higgs combination branch of RooFit. These fixes include
- fix for computing the contraint normalization. This requires now the option GlobalObservables when creating the NLL.
- All the `RooAbsPdf::createNLL` used in The RooStats classes have been updated to include the `GlobalObservables` option.
- Remove the `Roo1DMomentMorphFunction` and replace  it with `RooMomentMorphFunction` and `RooMomentMorphFunctionND`

## TMVA Library

- Improvement and fixes in ROCCurve class.
- Add support for event weights in the DNN
- Add in the DNN the option to use a validation data set independent of the training/test set used for training the DNN.
- Add option to suppress correlation outputs
- Improvements in the support for multi-class classification.
- Improvements in the Gradient Boostig Trees
- Deprecate the TMVA DNN Reference Implementation. Support now only CPU and GPU implementations. 


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
   - Auto-coloring for TF1 (drawing options PFC, PLC and PMC) is implemented.

## 3D Graphics Libraries
  - When a LEGO plot was drawn with Theta=90, the X and Y axis were misplaced.

## Geometry Libraries
  - Added system of units and physical constants matching the CLHEP port to Geant4, adapted to ROOT by Marko Petric.
  - Computing radiation length and nuclear interaction length for mixtures as in Geant4 to have
    numeric matching of average properties.
  - Added support for reading region definition and production cuts for e+, e-, gamma, p
    from GDML files
  - Added support for reading/writing parts of the geometry tree to GDML (Markus Frank)

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
  - Reintroduced greedy reduction in TProcessExecutor.
  - Fix empty chunks in the result vector of TThreadExecutor::Map. If the integer partition of the data in nChunks causes the existence of empty chunks (e.g the—rounded up—division of 12 elements in 5 chunks), the number of chunks is decreased to avoid empty chunks and, as a consequence, accesses to uninitialized memory in the reduction step.

## Language Bindings
  - PyROOT now supports list initialisation with tuples. For example, suppose to have a function `void f(const TH1F& h)`. In C++, this can be invoked with this syntax: `f({"name", "title", 64, -4, 4})`. In PyROOT this translates too `f(('name', 'title', 64, -4, 4))`.


## JavaScript ROOT

Upgrade JSROOT to v5.3.1. Following new features implemented:

- New supported classes:
    - TGraphPolar
    - TGraphTime
    - TSpline3
    - TSpline5
    - TPolyLine3D
    - TPolyMarker
    - TEfficiency
    - TH1K
- New supported options:
    * "PFC" - auto fill color (histograms and graphs)
    * "PLC" - auto line color
    * "PMC" - auto marker color
    * "A"  - fully disables axes drawing for histograms painters
    * "TEXT" - for TH2Poly
    * "SAMES" - draw stat box for superimposed histograms
    * "NOCOL" - ignore stored in the TCanvas colors list
    * "NOPAL" - ignore stored in the TCanvas color palette
- Improvements in existing painters:
     - use color palette stored in the TCanvas
     - draw stats box when really required
     - let resize frames and paves in all eight directions
     - support lines, boxes and arbitrary text positions in TPaveText
     - automatic title positioning of vertical axis when fTitleOffset==0
     - when pad.fTickx/y==2 draw axes labels on opposite side
     - editing of TGraph objects - moving of the graph bins
     - draw X/Y/Z axis titles in lego plots
     - use canvas Theta/Phi angles to set initial camera position in 3D plots
- New TLatex processor supports most ROOT features, still MathJax can be used
- New X/Y projections display for TH2 histograms (aka TH2::SetShowProjectionX/Y)
- New in geometry viewer:
    - provide shape parameters in TGeo tooltips
    - let inspect selected TGeoNode
    - provide text info when geometry drawing takes too long
- Change in JSROOT.draw functionality. Now valid painter instance can be only
   obtained via call-back - forth argument of JSROOT.draw() function.
- Use latest three.js r86 with improved Projector and CanvasRenderer
   Still use own SVGRenderer which supported direct SVG text dump
- Introduce openui5 components for webgui functionality
- In all sources specify "use strict" directive
- Basic JSROOT functionality can be used in Node.js:
       var jsroot = require("path/to/JSRootCore.js");
   One could parse ROOT JSON, read binary ROOT files (local and remote) and produce SVG.
- Implement dropping of TTree object on the geometry drawing.
   This automatically invokes extract_geo_tracks() function, which
   should extract TGeoTracks from provided TTree.
   Example can be found in demo/alice_esd.js and in api.htm.
- Implement projection of geometry on given plane.
   One could reuse drawing of geometry in other div (should be drawn with main option).
   In control GUI one could change position of the projection plane
- One of the TGeo drawing can be assigned as main. When same object drawn next time,
   its drawing will be derived from the main. Useful for geometry projections.
   Also all tracks and hits will be imported from main drawing.
- Let change background color of geo drawing.
- One can change web browser title, providing &title="any string" in URL.
- Introduce event status line, which is similar to ROOT TCanvas.
   Shown information similar to output in tooltip.
   One can enable both tooltips and status line at the same time.
- Introduce JSROOT.GEO.build function to create three.js model for
   any supported TGeo class. Such model can be inserted in any three.js scene
   independent from normal JSROOT drawings.
- Improve rendering of geometries with transparency. Use EVE approach, when transparent
   objects rendered after opaque and without writing depth buffer. Provide different
   methods to produce render order for transparent objects.
- Let specify initial zoom factor for geometry like opt=zoom50.
- Support TPolyMarker3D class in geo painter.
- Implement TGeoScaledShape.
- Limit complexity of composite shape. If it has too many components, only most left is used.
- When produce canvas or pad screenshot, render 3D objects with SVGRenderer.
    Allows to combine 2D and 3D objects in same PNG image
- Improve MathJax.js output. It scales correctly in Firefox, makes correct alignment
    and works significantly faster.
- When creating image in SVG format, correctly convert url("#id") references

Bugfixes:
- Show TH2 projections also when tooltip is disabled
- use z_handle to format Z-axis labels
- Support labels on TH3 Z axis
- TH1 zooming in 3D mode
- Suppress empty {} in TLatex
- Add several math symbols for TLatex
- Font kind 1 is italic times roman
- Do not let expand parent item in hierarchy
- Use correct painter to check range
- Change proper axis attributes in context menu
- Correctly show axis labels on 3D plot
- Correctly handle circle (marker kind 24) as marker kind
- Correct circle drawing with coordinates rounding
- TLatex #frac and #splitline, adjust vertical position
- Workaround for y range when fMinimum==fMaximum!=-1111
- Correct tooltips for graph with marker drawing
- Support pow(x,n) function in formula
- Use pad.fFillColor for frame when fFrameFillColor==0
- Correctly identify horizontal TGaxis with reverse scale
- Correctly handle negative line width in exclusion
- Tooltips handling for TF1
- Potential mix-up in marker attributes handling
- Unzomming of log scale https://root-forum.cern.ch/t/25889
- Ignore not-supported options in TMultiGraph https://root-forum.cern.ch/t/25888
- Correctly use fGridColor from TStyle
- Prevent error when TPaveText includes TLine or TBox in list of lines
- Bin errors calculations in TProfile
- Correctly handle new TF1 parameter coding convention (jsroot#132)
- Check if pad name can be used as element id (jsroot#133)
- Adjust title position for vertical axis with fTitleOffset==0


## Tutorials

- xml/xmlreadfile.C shows how to read and parse any xml file, supported by TXMLEngine class.
- fit/fitNormSum.C shows building of vectorized function and fitting with TF1.
- multicore/mt303_AsyncSimple.C explains uses of `Async()` and `TFuture`.
- multicore/mt304_fillHistos.C shows the new auto-binning mechanism.
- graphs/timeSeriesFromCSV_TDF.C illustrates a time axis on a TGraph with text-data read by `TDataFrame`.
- dataframe/tdf013_InspectAnalysis.C shows how to display incremental snapshots of `TDataFrame` analysis results in a `TBrowser`
- dataframe/tdf014_CSVDataSource.C shows reading text-data (comma separated) using a `TDataFrame`
- dataframe/tdf012_DefinesAndFiltersAsStrings.C shows how to use jitted defines and filters by calculating pi
  from checking how many randomly generated points in the unit square fall inside a unit circle
- most `TDataFrame` tutorials are now provided both in C++ and python

## Command line tools
  - `rootls` has been extended.
    - option `-l` displays the year
    - option `-t` displays all details of 'THnSparse'
  - `rootcp` bug fixes ([ROOT-8528](https://sft.its.cern.ch/jira/browse/ROOT-8528))
    - Now copies only the latest version of each object instead of copying all
      versions in wrong order.

## Class Reference Guide
  - The list of libraries needed by each class is displayed as a diagram.

## Build, Configuration and Testing Infrastructure

This is the last release with the configure/make-based build system. It will
be removed; please migrate to the CMake-based build system.
