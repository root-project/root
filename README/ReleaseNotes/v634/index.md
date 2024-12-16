% ROOT Version 6.34 Release Notes
% 2024-11
<a name="TopOfPage"></a>

## Important note about this development release

6.34 is a short term support cycle not meant to be used for data taking. It will be superseded by the 6.36 cycle, which is foreseen to start with 6.36.00 in the second quarter of 2025. Patch releases of the 6.36 cycle will be provided until June 30th 2025.


## Introduction

The development ROOT version 6.34.00 is scheduled for release at the end of November 2024.

For more information, see:

[http://root.cern](http://root.cern)

The following people have contributed to this new version:

 Guilherme Amadio, CERN/IT,\
 Bertrand Bellenot, CERN/EP-SFT,\
 Jakob Blomer, CERN/EP-SFT,\
 Patrick Bos, Netherlands eScience Center,\
 Rene Brun,\
 Carsten Burgard, DESY\
 Will Buttinger, RAL,\
 Philippe Canal, FNAL,\
 Jolly Chen, CERN/EP-SFT,\
 Olivier Couet, CERN/EP-SFT,\
 Marta Czurylo, CERN/EP-SFT,\
 Monica Dessole, CERN/EP-SFT,\
 Adrian Duesselberg, TU Munchen,\
 Mattias Ellert, Uppsala University,\
 Gerri Ganis, CERN/EP-SFT,\
 Florine de Geus, CERN/University of Twente,\
 Andrei Gheata, CERN/EP-SFT,\
 Enrico Guiraud,\
 Stephan Hageboeck, CERN/EP-SFT,\
 Jonas Hahnfeld, CERN/Goethe University Frankfurt,\
 Fernando Hueso Gonzalez, University of Valencia,\
 Attila Krasznahorkay, CERN/EP-ADP-OS,\
 Wim Lavrijsen, LBL,\
 Aaron Jomy, CERN/EP-SFT,\
 Ida Kaspary, Imperial College,\
 Valerii Kholoimov, National University of Kyiv/IRIS-HEP,\
 Sergey Linev, GSI,\
 Javier Lopez-Gomez,\
 Pere Mato, CERN/EP-SFT,\
 Andrea Maria Ola Mejicanos, Berea College,\
 Alaettin Serhan Mete, Argonne,\
 Thomas Madlener, DESY,\
 Vedant Mehra, GSOC, \
 Lorenzo Moneta, CERN/EP-SFT,\
 Alja Mrak Tadel, UCSD/CMS,\
 Axel Naumann, CERN/EP-SFT,\
 Ianna Osborne, Princeton University,\
 Vincenzo Eduardo Padulano, CERN/EP-SFT,\
 Giacomo Parolini, CERN/EP-SFT,\
 Danilo Piparo, CERN/EP-SFT,\
 Kristupas Pranckietis, Vilnius University,\
 Fons Rademakers, CERN/IT,\
 Jonas Rembser, CERN/EP-SFT,\
 Andrea Rizzi, University of Pisa,\
 Andre Sailer, CERN/EP-SFT,\
 Nopphakorn Subsa-Ard, KMUTT,\
 Pavlo Svirin, National Technical University of Ukraine,\
 Robin Syring, Leibniz University Hannover, CERN/EP-SFT,\
 Maciej Szymanski, Argonne,\
 Christian Tacke, Darmstadt University,\
 Matevz Tadel, UCSD/CMS,\
 Alvaro Tolosa Delgado, CERN/RCS-PRJ-FC,\
 Devajith Valaparambil Sreeramaswamy, CERN/EP-SFT,\
 Peter Van Gemmeren, Argonne,\
 Vassil Vassilev, Princeton/CMS,\
 Wouter Verkerke, NIKHEF/ATLAS,\
 Stefan Wunsch\

## Removal and Deprecation

The following interfaces have been removed:

- The `RooAbsReal::plotSliceOn()` function that was deprecated since at least ROOT 6 was removed. Use `plotOn(frame,Slice(...))` instead.
- Multiple overloads of internal Minuit 2 constructors and functions have been removed. If your code fails to compile, you can easily change to another overload that takes a `MnUserParameterState`, which is a change backwards compatible with older ROOT versions.

The following interfaces are deprecated and will be removed in future releases:

- The `RooTemplateProxy` constructors that take a `proxyOwnsArg` parameter to manually pass ownership are deprecated and replaced by a new constructor that takes ownership via `std::unique_ptr<T>`. They will be removed in ROOT 6.36.
- Several RooFit legacy functions are deprecated and will be removed in ROOT 6.36 (see section "RooFit libraries")
- The `int ROOT::CompressionSettings(ROOT::ECompressionAlgorithm algorithm, int compressionLevel)` function is deprecated and will be removed in ROOT 6.36. Please use `int CompressionSettings(RCompressionSetting::EAlgorithm::EValues algorithm, int compressionLevel)` instead.
- The `void R__zip(int cxlevel, int *srcsize, char *src, int *tgtsize, char *tgt, int *irep)` function is deprecated and will be removed in ROOT 6.36. Please use `void R__zipMultipleAlgorithm(int cxlevel, int *srcsize, char *src, int *tgtsize, char *tgt, int *irep, ROOT::RCompressionSetting::EAlgorithm::EValues algorithm)` instead.
- The `Bool_t TGeoShape::AreOverlapping(const TGeoBBox *box1, const TGeoMatrix *mat1, const TGeoBBox *box2, const TGeoMatrix *mat2)` function is deprecated and will be removed in ROOT 6.36.
- The `TPython::Eval()` function is deprecated and scheduled for removal in ROOT 6.36.


## Core Libraries

* The Cling C++ interpreter now relies on LLVM version 18.
* The `rootcling` invocation corresponding to a `genreflex` invocation can be obtained with the new `genreflex`
  command line argument `--print-rootcling-invocation`. This can be useful when migrating from genreflex to
  rootcling.
* The `rootcling` utility now fully supports selection xml files and not only LinkDef files.

## I/O Libraries

## RNTuple Libraries

* The first version of the `RNTuple` on-disk binary format is finalized. Future versions of ROOT will be able to read back
  RNTuple data written as of this release. Please note that this version breaks compatibility with experimental RNTuple
  data written with releases up to v6.34. Please also note that the RNTuple API is not yet moving out of
  `ROOT::Experimental`.
* Support for low-precision on-disk floating point representation. This can be enabled through
  `RField<float|double>::SetTruncated()` (truncated mantissa) and `RField<float|double>::SetQuantized()`
  (scaled integer representation).
* Link RNTuple self-description to the common ROOT streamer infrastructure. As a result, `TFile::MakeProject()`
  properly creates header files for classes used in RNTuple data.
* First version of the new `RNTupleProcessor` class. The `RNTupleProcessor` will support iteration of composed RNTuple data sets (comparable to and improving upon TTree friends and chains). This release supports chained (vertically composed) RNTuples. Other types of concatenations will be added in subsequent releases.
* Support for cluster staging in the `RNTupleParallelWriter`. Cluster staging enables users to enforce a certain
  logical cluster ordering in the presence of parallel cluster writing.
* Support for Direct I/O for writing. This gives access to the peak performance of modern NVMe drives.
* Support for a "streamer field" that can wrap classic ROOT I/O serialized data for RNTuple in cases where native
  RNTuple support is not possible (e.g., recursive data structures). Use of the streamer field can be enforced
  through the LinkDef option `rntupleStreamerMode(true)`.  This features is similar to the unsplit/level-0-split branch in `TTree`.
* Naming rules have been established for the strings representing the name of an RNTuple and the name of a field. The
  allowed character set is restricted to Unicode characters encoded as UTF-8, with the following exceptions: control
  codes, full stop, space, backslash, slash. See a full description in the RNTuple specification. The naming rules are
  also enforced when creating a new RNTuple or field for writing.
* Many fixes to RNTuple merging, both through `hadd` and when using the `RNTupleMerger` class directly. Most notable
  of these fixes is the proper handling of projected fields.
* Many additional bug fixes and improvements.

## TTree Libraries
* TTreeReader can now detect whether there is a mismatched number of entries between the main trees and the friend tree
  and act accordingly in two distinct scenarios. In the first scenario, at least one of the friend trees is shorter than
  the main tree, i.e. it has less entries. When the reader is trying to load an entry from the main tree which is beyond
  the last entry of the shorter friend, this will result in an error and stop execution. In the second scenario, at
  least one friend is longer than the main tree, i.e. it has more entries. Once the reader arrives at the end of the
  main tree, it will issue a warning informing the user that there are still entries to be read from the longer friend.
* TTreeReader can now detect whether a branch, which was previously expected to exist in the dataset, has disappeared
  due to e.g. a branch missing when switching to the next file in a chain of files.
* TTreeReader can now detect whether an entry being read is incomplete due to one of the following scenarios:
  * When switching to a new tree in the chain, a branch that was expected to be found is not available.
  * When doing event matching with TTreeIndex, one or more of the friend trees did not match the index value for
    the current entry.


## RDataFrame

* The `GetColumnNames` function to retrieve the number of available column names in the RDataFrame object is now also
  usable from a node of a distributed computation graph. This makes the generation of said computation graph slightly
  less lazy than before. Notably, it used to be the case that a distributed computation graph could be defined with
  code that was not yet available on the user's local application, but that would only become available in the
  distributed worker. Now a call such as `df.Define("mycol", "return run_my_fun();")` needs to be at least declarable
  to the interpreter also locally so that the column can be properly tracked.
* The order of execution of operations within the same branch of the computation graph is now guaranteed to be top to
  bottom. For example, the following code:
  ~~~{.cpp}
  ROOT::RDataFrame df{1};
  auto df1 = df.Define("x", []{ return 11; });
  auto df2 = df1.Define("y", []{ return 22; });
  auto graph = df2.Graph<int, int>("x","y");
  ~~~
  will first execute the operation `Define` of the column `x`, then the one of the column `y`, when filling the graph.
* The `DefinePerSample` operation now works also in the case when a TTree is stored in a subdirectory of a TFile.
* The memory usage of distributed RDataFrame was drastically reduced by better managing caches of the computation graph
  artifacts. Large applications which previously had issues with killed executors due to being out of memory now show a
  minimal memory footprint. See https://github.com/root-project/root/pull/16094#issuecomment-2252273470 for more details.
* RDataFrame can now read TTree branches of type `std::array` on disk explicitly as `std::array` values in memory.
* New parts of the API were added to allow dealing with missing data in a TTree-based dataset:
  * DefaultValueFor(colname, defaultval): lets the user provide one default value for the current entry of the input
    column, in case the value is missing.
  * FilterAvailable(colname): works in the same way as the traditional Filter operation, where the "expression" is "is
    the value available?". If so, the entry is kept, if not, it is discarded.
  * FilterMissing(colname): works in the same way as the traditional Filter operation, where the "expression" is "is
    the value missing?". If so, the entry is kept, if not, it is discarded.
  The tutorials `df036_missingBranches` and `df037_TTreeEventMatching` show example usage of the new functionalities.
* The automatic conversion of `std::vector` to `ROOT::RVec` which happens in memory within a JIT-ted RDataFrame
  computation graph meant that the result of a `Snapshot` operation would implicitly change the type of the input branch.
  A new option available as the data member `fVector2RVec` of the `RSnapshotOptions` struct can be used to prevent
  RDataFrame from making this implicit conversion.
* RDataFrame does not take a lock anymore to check reading of supported types when there is a mismatch, see
  https://github.com/root-project/root/pull/16528.
* Complexity of lookups during internal checks for type matching has been made constant on average, see the discussions
  at https://github.com/root-project/root/pull/16559 and https://github.com/root-project/root/pull/16559.
* Major improvements have been brought to the experimental feature that allows lazily loading ROOT data into batches for
  machine learning model training pipelines. For a full description, see the presentation at CHEP 2024
  https://indico.cern.ch/event/1338689/contributions/6015940/.

## Histogram Libraries

* `THStack:GetMinimum()` was not correct in case of negative contents.

### Upgrade TUnfold to version 17.9

The [TUnfold package](https://www.desy.de/~sschmitt/tunfold.html) inside ROOT is upgraded from version 17.6 to version 17.9.

## Math Libraries

### Minuit2

* **Usage of `std::span<const double>`in the interface**: To avoid forcing the user to do manual memory allocations via `std::vector`, the interfaces of Minuit 2 function adapter classes like `ROOT::Minuit2::FCNBase` or `ROOT::Minuit2::FCNGradientBase` were changed to accept `std::span<const double>` arguments instead of `std::vector<double> const&`.
This should have minimal impact on users, since one should usual use Minuit 2 via the `ROOT::Math::Minimizer` interface, which is unchanged.

* **Initial error/covariance matrix values for Hessian matrix**: Initial error/covariance matrix values can be passed for initializating the Hessian matrix to be used in minimization algorithms by attaching the covariance matrix to the `ROOT::Minuit2::MnUserParameterState` instance used for seeding via the method `AddCovariance(const MnUserCovariance &);`.

## RooFit Libraries

### Error handling in MultiProcess-enabled fits

The `MultiProcess`-based fitting stack now handles errors during fits.
Error signaling in (legacy) RooFit happens through two mechanisms: `logEvalError` calls and `RooNaNPacker`-enhanced NaN doubles.
Both are now implemented and working for `MultiProcess`-based fits as well.
See [this PR](https://github.com/root-project/root/pull/15797) for more details.
This enables the latest ATLAS Higgs combination fits to complete successfully, and also other fits that encounter NaN values or other expected errors.

### Miscellaneous

* Setting `useHashMapForFind(true)` is not supported for RooArgLists anymore, since hash-assisted finding by name hash can be ambiguous: a RooArgList is allowed to have different elements with the same name. If you want to do fast lookups by name, convert your RooArgList to a RooArgSet.

* The function `RooFit::bindFunction()` now supports arbitrary many input variables when binding a Python function.

* The `ExportOnly()` attribute of the `RooStats::HistFactory::Measurement` object is now switched on by default, and the associated getter and setter functions are deprecated. They will be removed in ROOT 6.36. If you want to fit the model as well instead of just exporting it to a RooWorkspace, please do so with your own code as demonstrated in the `hf001` tutorial.

* Initial error values can be used for initializating the Hessian matrix to be used in Minuit2 minimization algorithms by setting the `RooMinimizer::Config` option `setInitialCovariance` to `true`. These values correspond to the diagonal entries of the initial covariance matrix.

* `RooFit::MultiProcess`-enabled fitting developer/advanced documentation -- [available through GitHub](https://github.com/root-project/root/blob/master/roofit/doc/developers/test_statistics.md) -- was updated. It now contains the most up to date usage instructions for optimizing load balancing (and hence run speed) using this backend.

### Deprecations

* The `RooStats::MarkovChain::GetAsDataSet` and `RooStats::MarkovChain::GetAsDataHist` functions are deprecated and will be removed in ROOT 6.36. The same functionality can be implemented by calling `RooAbsData::reduce` on the Markov Chain's `RooDataSet*` (obtained using `MarkovChain::GetAsConstDataSet`) and then obtaining its binned clone(for `RooDataHist`).

  An example in Python would be:

  ```py
  mcInt = mc.GetInterval() # Obtain the MCMCInterval from a configured MCMCCalculator
  mkc = mcInt.GetChain() # Obtain the MarkovChain
  mkcData = mkc.GetAsConstDataSet()
  mcIntParams = mcInt.GetParameters()

  chainDataset = mkcData.reduce(SelectVars=mcIntParams, EventRange=(mcInt.GetNumBurnInSteps(), mkc.Size()))
  chainDataHist = chainDataset.binnedClone()
  ```

* The following methods related to the RooAbsArg interface are deprecated and will be removed in ROOT 6.36.
They should be replaced with the suitable alternatives interfaces:

    - `RooAbsArg::getDependents()`: use `getObservables()`
    - `RooAbsArg::dependentOverlaps()`: use `observableOverlaps()`
    - `RooAbsArg::checkDependents()`: use `checkObservables()`
    - `RooAbsArg::recursiveCheckDependents()`: use `recursiveCheckObservables()`

## TMVA SOFIE
The support for new ONNX operators has been included in the SOFIE ONNX parser and in RModel in order to generate inference code for new types of models.
The full list of currently supported operators is available [here](https://github.com/root-project/root/blob/master/tmva/sofie/README.md#supported-onnx-operators)

The list of operators added for this release is the following:
   - Constant and ConstantOfShape
   - If
   - Range
   - ReduceSum
   - Split
   - Tile
   - TopK

In addition support in RModel has been added to generate the code with dynamic input shape parameter, such as the batch size. These input shape parameters can be specified at run time when evaluating the model.
Since not all ONNX operators in SOFIE support yet dynamic input parameters, it is possible to initialize a parsed dynamic model with fixed values. For this, a new member function, `RModel::Initialize(const std::map<std::string,size_t> & inputParams, bool verbose = false)` has been added.
The RModel class has been extended to support sub-graph (needed for operator `If`), dynamic tensors and constant tensors (for example those defined by the operator `Constant`).

## Graphics Backends

### Web-based TWebCanvas

Support "haxis" draw option for histograms, allows superposition of several histograms drawn on the same pad with horizontal ty axis. Add `tutorials\webcanv\haxis.cxx` macro demonstrating new feature.

Support "frame" draw option for several primitives like `TBox`, `TLine`, `TLatex`. This enforce clipping of such objects by
frame border. Provide demo in `tutorials\webcanv\inframe.cxx` macro

Provide batch mode for image production with headless browser. In such mode data for several canvases collected together (in batch) and then N images are produced with single invocation of the web browser (chrome or firefox). For instance after `TWebCanvas::BatchImageMode(100)` next 99 calls to `TCanvas::SaveAs(filename)` method will not lead to image files creation. But with following call all 100 images will be produced together. Alternatively one can use `TCanvas::SaveAll()` static method which allows to create images for several canvases at once.

Support multi-page PDF file creation with web-based canvas using `svg2pdf.js` library. Both with native and web-baed graphics one can do now:
```c++
c1->SaveAs("file.pdf[")
c2->SaveAs("file.pdf+")
c3->SaveAs("file.pdf+")
c4->SaveAs("file.pdf]")
```
Or same can be achieved with:
```c++
TCanvas::SaveAll({c1, c2, c3, c4}, "file.pdf");
```

## 2D Graphics Libraries
* In `TGraphErrors` `TGraphAsymmErrors` and `TGraphBentErrors`, the error bars were drawn inside the marker when the marker was bigger than the error bars. This produced a weird plot. This is now fixed.

* When error-bars exceeded the y range limits the end of error bars were nevertheless displayed was not correcton the x-bottom and top axis. So it looked like the total error bar while it was indeed not.

* Choosing an appropriate color scheme is essential for making results easy to understand and interpret. Factors like colorblindness and converting colors to grayscale for publications can impact accessibility. Furthermore, results should be aesthetically pleasing. The following three color schemes, recommended by M. Petroff in [arXiv:2107.02270v2](https://arxiv.org/pdf/2107.02270) and available on [GitHub](https://github.com/mpetroff/accessible-color-cycles) under the MIT License, meet these criteria.

* Implement properly the TScatter palette attributes as requested [here](https://github.com/root-project/root/issues/15922).

* Add `TStyle::SetLegendFillStyle`

## 3D Graphics Libraries

## Geometry Libraries

The geometry package is now optional and activated by default in the CMake configuration. To disable it, use the `-Dgeom=OFF` CMake option.

## Web-based GUIs

Adjust `rootssh` script to be usable on MacOS. Fixing problem to start more than one web widget on remote node.

Fix `rootbrowse` script to be able properly use it with all kinds of web widgets. Provide `--web=<type>` argument as for
regular root executable.

Update openui5 library to version 1.128.0. Requires use of modern web-browsers, skipping IE support.

## Python Interface

### Typesafe `TTree::SetBranchAddress()` for array inputs

If you call `TTree::SetBranchAddress` with NumPy array or `array.array` inputs, ROOT will now check if the array type matches with the column type.
If it doesn't, `SetBranchAddress()` will return a negative status code and print an error.
Take for example this code snippet:
```python
arr = array.array(typecode, "d")
status = t.SetBranchAddress("name", arr)
print("Status = %s" % (status, ))
```
If the branch type is also `double` (like the type of the array indicated by `"d"`), the call to `SetBranchAddress()` would succeed with status code zero.
If the type doesn't match, you now get a clear error instead of garbage values.
```txt
Error in <TTree::SetBranchAddress>: The pointer type given "Double_t" (8) does not correspond to the type needed "Float_t" (5) by the branch: a
Status = -2
```

### Deprecation of `TPython::Eval()`

The `TPython::Eval()` method is deprecated and scheduled for removal in ROOT 6.36.
Its implementation was fragile, and the same functionality can be achieved with `TPython::Exec()`, using a C++ variable that is known to the ROOT interpreter for crossing over from Python to C++.

Example:
```c++
// Before, with TPython::Eval()
std::string stringVal = static_cast<const char*>(TPython::Eval("'done'"));
std::cout << stringVal << std::endl;

// Now, with TPython::Exec(). You can set `_anyresult` to whatever std::any you want.
// It will be swapped into the return variable in the end.

std::any result;
TPython::Exec("_anyresult = ROOT.std.make_any['std::string']('done')", &result);
std::cout << std::any_cast<std::string>(result) << std::endl;
```
## JavaScript ROOT

Upgrade to JSROOT 7.8.0 with following new features and fixes:

1. Let use custom time zone for time display, support '&utc' and '&cet' in URL parameters
2. Support gStyle.fLegendFillStyle
3. Let change histogram min/max values via context menu
4. Support Z-scale zooming with `TScatter`
5. Implement "haxis" draw option for histogram to draw only axes for hbar
6. Implement "axisg" and "haxisg" to draw axes with grids
7. Support `TH1` marker, text and line drawing superimposed with "haxis"
8. Support `TBox`, `TLatex`, `TLine`, `TMarker` drawing on "frame", support drawing on swapped axes
9. Implement `TProfile` and `TProfile2D` projections https://github.com/root-project/root/issues/15851
10. Draw total histogram from `TEfficiency` when draw option starts with 'b'
11. Let redraw `TEfficiency`, `THStack` and `TMultiGraph` with different draw options via hist context menu
12. Support 'pads' draw options for `TMultiGraph`, support context menu for it
13. Let drop objects on sub-pads
14. Properly loads ES6 modules for web canvas
15. Improve performance of `TH3`/`RH3` drawing by using `THREE.InstancedMesh`
16. Implement batch mode with '&batch' URL parameter to create SVG/PNG images with default GUI
17. Adjust node.js implementation to produce identical output with normal browser
18. Create necessary infrastructure for testing with 'puppeteer'
19. Support injection of ES6 modules via '&inject=path.mjs'
20. Using importmap for 'jsroot' in all major HTML files and in demos
21. Implement `settings.CutAxisLabels` flag to remove labels which may exceed graphical range
22. Let disable usage of `TAxis` custom labels via context menu
23. Let configure default draw options via context menu, preserved in the local storage
24. Let save canvas as JSON file from context menu, object as JSON from inspector
25. Upgrade three.js r162 -> r168, use r162 only in node.js because of "gl" module
26. Create unified svg2pdf/jspdf ES6 modules, integrate in jsroot builds
27. Let create multi-page PDF document - in `TWebCanvas` batch mode
28. Let add in latex external links via `#url[link]{label}` syntax - including jsPDF support
29. Support `TAttMarker` style with line width bigger than 1
30. Provide link to ROOT class documentation from context menus
31. Implement axis labels and title rotations on lego plots
32. Internals - upgrade to eslint 9
33. Internals - do not select pad (aka gPad) for objects drawing, always use assigned pad painter
34. Fix - properly save zoomed ranges in drawingJSON()
35. Fix - properly redraw `TMultiGraph`
36. Fix - show empty bin in `TProfile2D` if it has entries #316
37. Fix - unzooming on log scale was extending range forever
38. Fix - display empty hist bin if fSumw2 not zero
39. Fix - geometry display on android devices

JSROOT is now used as default display in `jupyter`.


## Tools

### hadd

* Fixed a bug where in some circumstances `hadd` would not correctly merge objects in nested folders of a ROOT file.


## Tutorials

* New tutorials [accessiblecolorschemes.C](https://root.cern/doc/master/accessiblecolorschemes_8C.html) and [hstackcolorscheme.C](https://root.cern/doc/master/thstackcolorscheme_8C.html).

## Class Reference Guide


## Build, Configuration and Testing Infrastructure

- Coverage of the CI was greatly improved, with Clang builds, Alma9 ARM64 and Alma9 x86 NVidia GPU builds were added to the CI

The following builtins have been updated:

- daviX 0.8.7
- XRootD 5.7.1

## Bugs and Issues fixed in this release

More than 200 items were addressed for this release. The full list is:

* [[#17040](https://github.com/root-project/root/issues/17040)] - Small difference between kp6Violet implementation and official value from Petroff paper
* [[#16976](https://github.com/root-project/root/issues/16976)] - Strange overflow bin bar when plotting TH1D with X1 option
* [[#16946](https://github.com/root-project/root/issues/16946)] - Crash in RDF constructor with empty file list
* [[#16942](https://github.com/root-project/root/issues/16942)] - another crash in finalization 
* [[#16834](https://github.com/root-project/root/issues/16834)] - `RFieldBase::Create` does not enforce valid field names
* [[#16826](https://github.com/root-project/root/issues/16826)] - RNTuple unexpected "field iteration over empty fields is unsupported"
* [[#16796](https://github.com/root-project/root/issues/16796)] - RooBinSamplingPdf does not forward expectedEventsFunc creation calls
* [[#16784](https://github.com/root-project/root/issues/16784)] - Remove default value of p from TH1::GetQuantiles() as is the case with TF1::GetQuantiles
* [[#16771](https://github.com/root-project/root/issues/16771)] - copying a default constructed `TH2Poly` fails.
* [[#16753](https://github.com/root-project/root/issues/16753)] - [ntuple] Free uncompressed page buffers in RPageSinkBuf with IMT
* [[#16752](https://github.com/root-project/root/issues/16752)] - [ntuple] Copy sealed page in RPageSinkBuf after compression
* [[#16736](https://github.com/root-project/root/issues/16736)] - Please improve documentation and/or argument names for TH1::GetQuantiles()
* [[#16715](https://github.com/root-project/root/issues/16715)] - TMVA fails to link to cudnn
* [[#16687](https://github.com/root-project/root/issues/16687)] - Loss of floating point precision when saving TCanvas as ROOT macro
* [[#16680](https://github.com/root-project/root/issues/16680)] - TMVA/Sofie tutorials used same name for generated files bur are run in parallel.
* [[#16647](https://github.com/root-project/root/issues/16647)] - ROOT_ADD_PYUNITTEST and ROOT_ADD_GTEST are naming test inconsitently.
* [[#16600](https://github.com/root-project/root/issues/16600)] - TMVA RReader not multithread safe
* [[#16588](https://github.com/root-project/root/issues/16588)] - Fix RFieldBase::GetNElements() for record/class fields
* [[#16562](https://github.com/root-project/root/issues/16562)] - TTreeViewer save session absolute path
* [[#16523](https://github.com/root-project/root/issues/16523)] - OpenGL doesn't work on macosx
* [[#16513](https://github.com/root-project/root/issues/16513)] - [ntuple] Clarifications about late schema extension
* [[#16479](https://github.com/root-project/root/issues/16479)] - Add THStack/TH1 constructor for TRatioPlot
* [[#16475](https://github.com/root-project/root/issues/16475)] - Unable to use EOS tokens with RDataFrame since 6.32
* [[#16474](https://github.com/root-project/root/issues/16474)] - Hadd does not add correctly histograms in nested folders
* [[#16469](https://github.com/root-project/root/issues/16469)] - cppyy no aggregate initialization constructor
* [[#16419](https://github.com/root-project/root/issues/16419)] - RooUnblindOffset crashes for root version 6.32
* [[#16402](https://github.com/root-project/root/issues/16402)] - Importing ROOT prevents Python garbage collection
* [[#16374](https://github.com/root-project/root/issues/16374)] - Configuring with builtin xrootd can fail because of seemingly not found OpenSSL library
* [[#16366](https://github.com/root-project/root/issues/16366)] - Compiler warning in Bytes.h: casts away qualifiers
* [[#16360](https://github.com/root-project/root/issues/16360)] - [rdf] gcc14 issue warning in `RDF/InterfaceUtils.hxx`
* [[#16326](https://github.com/root-project/root/issues/16326)] - [ntuple] Better control of cluster ordering for parallel writes
* [[#16324](https://github.com/root-project/root/issues/16324)] - [ntuple] Allow for creating bare model from on-disk info
* [[#16321](https://github.com/root-project/root/issues/16321)] - [ntuple] Split RNTupleView<T, bool> in two classes
* [[#16298](https://github.com/root-project/root/issues/16298)] - [PyROOT] Conversion from `std::string` to `std::string_view` broken in 6.32
* [[#16290](https://github.com/root-project/root/issues/16290)] - [ntuple] Provide tutorial for (envisioned) framework usage
* [[#16252](https://github.com/root-project/root/issues/16252)] - tutorial-rcanvas-df104-py
* [[#16249](https://github.com/root-project/root/issues/16249)] - Iterating with a range for does one extra iteration
* [[#16244](https://github.com/root-project/root/issues/16244)] - JSROOT not drawing bins with content=0 but entries > 0 in TProfile2D
* [[#16241](https://github.com/root-project/root/issues/16241)] - [ntuple] Method to prepare cluster commit / flush column write buffers
* [[#16236](https://github.com/root-project/root/issues/16236)] - [ntuple] Improve field token usage for parallel writing
* [[#16219](https://github.com/root-project/root/issues/16219)] - Module map on the new XCode version for macos15-beta
* [[#16190](https://github.com/root-project/root/issues/16190)] - TFileMerger behaviour when the directory structure contains repeated names
* [[#16184](https://github.com/root-project/root/issues/16184)] - Serialisation (and therefore I/O) issues with TF1 and TFitResultPtr 
* [[#16167](https://github.com/root-project/root/issues/16167)] - TGeomPainter Web not behaving the same way as TGeomPainter ROOT
* [[#16149](https://github.com/root-project/root/issues/16149)] - CMake and xrootd builtin
* [[#16135](https://github.com/root-project/root/issues/16135)] - [ntuple] Cannot create RFieldBase for signed char
* [[#16124](https://github.com/root-project/root/issues/16124)] - RNTupleInspector returns wrong compressed size for large N-tuples
* [[#16121](https://github.com/root-project/root/issues/16121)] - Potential memory leak in clang triggered by `findScope`
* [[#16051](https://github.com/root-project/root/issues/16051)] - TColor::GetFreeColorIndex()  returns index that is already used
* [[#16047](https://github.com/root-project/root/issues/16047)] - TMVA SOFIE shadow declaration
* [[#16031](https://github.com/root-project/root/issues/16031)] - VecOps binary functions not using the right types
* [[#16024](https://github.com/root-project/root/issues/16024)] - `thisroot.sh` tries to drop the wrong lib paths from the existing environment
* [[#15977](https://github.com/root-project/root/issues/15977)] - [gui] Event StatusBar does not work well when TMarker outside of zoom region
* [[#15962](https://github.com/root-project/root/issues/15962)] - outdated help links
* [[#15959](https://github.com/root-project/root/issues/15959)] - [RF] Make Offset(“bin”) usable for CLs method
* [[#15948](https://github.com/root-project/root/issues/15948)] - Tex Gyre fonts has a bad side effect ...
* [[#15924](https://github.com/root-project/root/issues/15924)] - python -c 'import ROOT' fails on macOS if ROOT is built with gnuinstall=ON
* [[#15919](https://github.com/root-project/root/issues/15919)] - Problem with TClass::GetListOfAllPublicMethods() in python
* [[#15912](https://github.com/root-project/root/issues/15912)] - Clad issues with `MacOSX15.0.sdk`
* [[#15887](https://github.com/root-project/root/issues/15887)] - Broken plot .C macros for default Name() argument in plotOn()
* [[#15883](https://github.com/root-project/root/issues/15883)] - Initialize TRatioPlot margins from Pad margins set in the current style
* [[#15851](https://github.com/root-project/root/issues/15851)] - Support for TProfile and TProfile2D projectionX and projectionXY options in JSROOT
* [[#15774](https://github.com/root-project/root/issues/15774)] - [ci] Add Python version to Windows precomplied release title or filename
* [[#15756](https://github.com/root-project/root/issues/15756)] - [RF][HS3] ATLAS ttbar workspaces roundtrip
* [[#15740](https://github.com/root-project/root/issues/15740)] - `THStack` does not automatically shows negative bins
* [[#15738](https://github.com/root-project/root/issues/15738)] - Segmentation violation during build on ix86 (32 bit intel)
* [[#15736](https://github.com/root-project/root/issues/15736)] - [df] ProgressBar reporting on number of files is now broken 
* [[#15727](https://github.com/root-project/root/issues/15727)] - Windows CMake project cannot find_library() after integrating with ROOT.
* [[#15703](https://github.com/root-project/root/issues/15703)] - Leaking memory though strings in PyROOT
* [[#15686](https://github.com/root-project/root/issues/15686)] - JITted code changes the execution order of computation graph nodes
* [[#15666](https://github.com/root-project/root/issues/15666)] - [ntuple][doc] document RNTuple Anchor format
* [[#15661](https://github.com/root-project/root/issues/15661)] - [ntuple] Cannot properly read late model extension (meta)data
* [[#15643](https://github.com/root-project/root/issues/15643)] - TGFileContainer crashes in pyroot 
* [[#15617](https://github.com/root-project/root/issues/15617)] - `RDF::Describe` returns an incorrect file count
* [[#15590](https://github.com/root-project/root/issues/15590)] - Infinite recursion in TFile::Open
* [[#15537](https://github.com/root-project/root/issues/15537)] - [cling] Crash when non-void function does not return a value
* [[#15534](https://github.com/root-project/root/issues/15534)] - RNTuple: fields with mixed STL types sometimes fail to be filled
* [[#15511](https://github.com/root-project/root/issues/15511)] - Possible memory corruption in cling
* [[#15503](https://github.com/root-project/root/issues/15503)] - Allow users to change default Snapshot behaviour of collections
* [[#15460](https://github.com/root-project/root/issues/15460)] - TEnum::GetEnum("B")->GetUnderlyingType() does not following typedefs
* [[#15447](https://github.com/root-project/root/issues/15447)] - `-Dminimal=ON` disables `runtime_cxxmodules`
* [[#15442](https://github.com/root-project/root/issues/15442)] - Distributed RDataFrame does not see all defined column names
* [[#15425](https://github.com/root-project/root/issues/15425)] - TTreeProcessorMP processes events multiple times when there are more threads than entries
* [[#15419](https://github.com/root-project/root/issues/15419)] - RNTuple: add max key length field to RNTuple anchor
* [[#15407](https://github.com/root-project/root/issues/15407)] - `cling::utils::Lookup::Named` does not look into using directive
* [[#15406](https://github.com/root-project/root/issues/15406)] - `TEnum::GetEnum` does not seem to see 'through' using statements.
* [[#15405](https://github.com/root-project/root/issues/15405)] - [RF] ExternalConstraints documentation incorrect for RooMCStudy
* [[#15384](https://github.com/root-project/root/issues/15384)] - GetCppName: Mangled version of the C++ symbol
* [[#15336](https://github.com/root-project/root/issues/15336)] - [MSVC] ROOT_x86 failed due to libCling.exp : error LNK2001: unresolved external symbol "char const * __cdecl __std_find_trivial<char const ,char>(char const *,char const *,char)
* [[#15321](https://github.com/root-project/root/issues/15321)] - [MSVC] Root is failed with error G694476FC: static_assert failed "Unexpected size"
* [[#15285](https://github.com/root-project/root/issues/15285)] - Fast element setter/getter for TMatrixT/TVectorT classes
* [[#15270](https://github.com/root-project/root/issues/15270)] - MakeClass and MakeSelector fails with special character in branchname.
* [[#15269](https://github.com/root-project/root/issues/15269)] - Iterators in pyROOT working differently in ROOT master compared to 6.30/02
* [[#15213](https://github.com/root-project/root/issues/15213)] - cmake warning while configuring
* [[#15178](https://github.com/root-project/root/issues/15178)] - ROOT generates CMake warnings when building from the tarball
* [[#15118](https://github.com/root-project/root/issues/15118)] - jsoninterface does not build if provided with RapidYAML
* [[#15107](https://github.com/root-project/root/issues/15107)] - [ci] clang-format fails when adding commits
* [[#15090](https://github.com/root-project/root/issues/15090)] - TClass::GetClassInfo() is not thread safe
* [[#15039](https://github.com/root-project/root/issues/15039)] - [RDataFrame] Expose more local df operations for distributed RDF
* [[#14966](https://github.com/root-project/root/issues/14966)] - Fix print check for object that return different types for begin() and end()
* [[#14871](https://github.com/root-project/root/issues/14871)] - [ntuple] add streamer info records to TFile
* [[#14809](https://github.com/root-project/root/issues/14809)] - [ntuple] Incorrect treatment of unsplittable classes
* [[#14808](https://github.com/root-project/root/issues/14808)] - [ntuple] TObject serialization faulty
* [[#14789](https://github.com/root-project/root/issues/14789)] - interpreter fails with assertion in debug builds on ARM when upgrading gcc
* [[#14767](https://github.com/root-project/root/issues/14767)] - rootn.exe instant crash on startup
* [[#14710](https://github.com/root-project/root/issues/14710)] - `std::set` not working in Windows PyROOT
* [[#14697](https://github.com/root-project/root/issues/14697)] - [FreeBSD] davix build failure
* [[#14592](https://github.com/root-project/root/issues/14592)] - Error value and context of call to FT_Set_Char_Size in TTF::SetTextSize should be in error message
* [[#14561](https://github.com/root-project/root/issues/14561)] - [ROOT-4936] TMatrixTSym is not actually symmetric
* [[#14544](https://github.com/root-project/root/issues/14544)] - [ROOT-8515] Make TEntryList class reference relevant
* [[#14541](https://github.com/root-project/root/issues/14541)] - [ROOT-6193] Editor for palette axis cannot set title properties
* [[#14487](https://github.com/root-project/root/issues/14487)] - Assert when trying to write RNTuple to full disk
* [[#14217](https://github.com/root-project/root/issues/14217)] - Module merge problems with GCC 13, C++20, Pythia8
* [[#14173](https://github.com/root-project/root/issues/14173)] - Adding a couple of useful methods in THnD
* [[#14132](https://github.com/root-project/root/issues/14132)] - Lazy multithread RDataFrame::Snapshot cause unnessary warning and break gDirectory
* [[#14055](https://github.com/root-project/root/issues/14055)] - Failing build with `-Dasan=ON` and memory leak in minimal build
* [[#13729](https://github.com/root-project/root/issues/13729)] - [math] Contour method has some problems with Minuit2
* [[#13677](https://github.com/root-project/root/issues/13677)] - [Cling] Potential unloading issue which breaks distributed execution
* [[#13511](https://github.com/root-project/root/issues/13511)] - TMapFile can't work 
* [[#13498](https://github.com/root-project/root/issues/13498)] - Assertion failure in TMVA `can't dereference value-initialized vector iterator`
* [[#13481](https://github.com/root-project/root/issues/13481)] - Update doc to express deprecation of genreflex and usage of rootcling as a replacement
* [[#13432](https://github.com/root-project/root/issues/13432)] - TCling::AutoLoad may not work if a pcm linked to the library is not preloaded
* [[#13055](https://github.com/root-project/root/issues/13055)] - -Dtmva-sofie=OFF does not switch off sofie.
* [[#13016](https://github.com/root-project/root/issues/13016)] - Extra vertical space on a canvas when CanvasPreferGL is set to true, reproducible via SSH
* [[#12935](https://github.com/root-project/root/issues/12935)] - [RF] Global correlation coefficients after SumW2Error
* [[#12842](https://github.com/root-project/root/issues/12842)] - [ntuple] Review the column representation of nullable fields
* [[#12509](https://github.com/root-project/root/issues/12509)] - TClass prefers `<Double32_t`> over  `<double>` specialization
* [[#12460](https://github.com/root-project/root/issues/12460)] - [ntuple] Set non-negative column flag for unsigned integer fields
* [[#12428](https://github.com/root-project/root/issues/12428)] - Test failure in RNTuple: RNTuple.TClassEBO fails
* [[#12426](https://github.com/root-project/root/issues/12426)] - RNTuple endian issues
* [[#12334](https://github.com/root-project/root/issues/12334)] - TTreeReader fails to read `T<Double_32>` as `T<double>`
* [[#12272](https://github.com/root-project/root/issues/12272)] - CI: releases
* [[#12251](https://github.com/root-project/root/issues/12251)] - Problems with `TH1::GetQuantiles`
* [[#12182](https://github.com/root-project/root/issues/12182)] - TPython::Eval does not work with string with python3.8+ for ROOT 6.24-6.26.8
* [[#12136](https://github.com/root-project/root/issues/12136)] - [ntuple] `RNTupleView`'s move ctor causes double delete
* [[#12108](https://github.com/root-project/root/issues/12108)] - `constexpr` function return incorrect value in Windows
* [[#11749](https://github.com/root-project/root/issues/11749)] - Remove empty files from the source distribution tarball
* [[#11707](https://github.com/root-project/root/issues/11707)] - Crash when macro is named main.cpp
* [[#11603](https://github.com/root-project/root/issues/11603)] - Disable automatic 'call home' in cmake when not needed
* [[#11353](https://github.com/root-project/root/issues/11353)] - Compiled program with libNew.so crash
* [[#10317](https://github.com/root-project/root/issues/10317)] - [Doxygen] tutorials appear as namespaces
* [[#10239](https://github.com/root-project/root/issues/10239)] - ? wildcard broken in TChain::Add()
* [[#10010](https://github.com/root-project/root/issues/10010)] - TLeaf::ReadBasket invalid write in TMVA test
* [[#9792](https://github.com/root-project/root/issues/9792)] - should fLogger be persistant ?
* [[#9646](https://github.com/root-project/root/issues/9646)] - Numerically stable computation of invariant mass
* [[#9637](https://github.com/root-project/root/issues/9637)] - `TGraph::Add(TF1 *f)` method like for `TH1`'s
* [[#9445](https://github.com/root-project/root/issues/9445)] - Hit errors when build ROOT with msvc on AddressSanitizer mode
* [[#9425](https://github.com/root-project/root/issues/9425)] - [RF] Figure out how to handle RooArgList with duplicates and hash-assisted find
* [[#9188](https://github.com/root-project/root/issues/9188)] - Unnecessary (?) warnings reading `unique_ptr`
* [[#9137](https://github.com/root-project/root/issues/9137)] - [tree] TTree/TChain silently return bogus data if friend is shorter than main tree
* [[#8833](https://github.com/root-project/root/issues/8833)] - Crash reading >= 3D array in TTree via MakeClass in Windows ROOT6 compilation
* [[#8828](https://github.com/root-project/root/issues/8828)] - Crash when defining something in the Detail namespace after a lookup of that namespace
* [[#8815](https://github.com/root-project/root/issues/8815)] - TBB not inheriting CXXFLAGS
* [[#8716](https://github.com/root-project/root/issues/8716)] - Minuit2: FCNGradientBase::CheckGradient() is ignored
* [[#8704](https://github.com/root-project/root/issues/8704)] - [DF] Add support for 'missing' columns
* [[#8367](https://github.com/root-project/root/issues/8367)] - *** Break *** segmentation violation in case of compilation errors in unnamed macros
* [[#8194](https://github.com/root-project/root/issues/8194)] - TClass::GetStreamerInfo crashes for several classes
* [[#8031](https://github.com/root-project/root/issues/8031)] - Reserve "build" directory name in ROOT sources for build files
* [[#7875](https://github.com/root-project/root/issues/7875)] - [ntuple] Improve normalization of platform-specific primitives and typedefs
* [[#7823](https://github.com/root-project/root/issues/7823)] - [RF] RooStatsUtils::MakeCleanWorkspace
* [[#7713](https://github.com/root-project/root/issues/7713)] - [Tree] Bogus data silently read when trying to access an indexed friend TTree with an invalid index
* [[#7160](https://github.com/root-project/root/issues/7160)] - MacOS: -Dcocoa=ON -Dopengl=OFF pass cmake but fail compilation
* [[#7103](https://github.com/root-project/root/issues/7103)] - [RF] HistFactory::FlexibleInterpVar Interpolation code2 and code3 are the same
* [[ROOT-10975](https://its.cern.ch/jira/browse/ROOT-10975)] - ACLiC should make rootcling warnings visible
* [[ROOT-10908](https://its.cern.ch/jira/browse/ROOT-10908)] - SMatrix<double> is written as a Double32_t
* [[ROOT-10902](https://its.cern.ch/jira/browse/ROOT-10902)] - SMatrix read from TTree contains all zeroes
* [[ROOT-10883](https://its.cern.ch/jira/browse/ROOT-10883)] - Warning in TBrowser when selecting "Add" method of a histogram
* [[ROOT-10865](https://its.cern.ch/jira/browse/ROOT-10865)] - [RVec] No Doxygen documentation about arithmetic operators
* [[ROOT-10698](https://its.cern.ch/jira/browse/ROOT-10698)] - Valgrind dies at assertion ‘!overlap’ failed
* [[ROOT-10539](https://its.cern.ch/jira/browse/ROOT-10539)] - Slow tutorials/dataframe/df027_SQliteDependencyOverVersion.C
* [[ROOT-10414](https://its.cern.ch/jira/browse/ROOT-10414)] - rootcling doesn't parse -isystem correctly
* [[ROOT-10342](https://its.cern.ch/jira/browse/ROOT-10342)] - valuePrint 'forgets' template argument in type when printing about an assignment statement.
* [[ROOT-10200](https://its.cern.ch/jira/browse/ROOT-10200)] - Automatic reloading doesn't work for std::cout on macOS
* [[ROOT-9961](https://its.cern.ch/jira/browse/ROOT-9961)] - TTree::Print("toponly") inserts extra newline between listed items
* [[ROOT-9953](https://its.cern.ch/jira/browse/ROOT-9953)] - TRint should not terminate on assert
* [[ROOT-9919](https://its.cern.ch/jira/browse/ROOT-9919)] - TFile construction silently drops XRootD protocol
* [[ROOT-9918](https://its.cern.ch/jira/browse/ROOT-9918)] - Crash TMVA by running (unused?) public function from TMVA::Factory
* [[ROOT-9705](https://its.cern.ch/jira/browse/ROOT-9705)] - flag to disable (root)test(s) that uses remote files
* [[ROOT-9673](https://its.cern.ch/jira/browse/ROOT-9673)] - Printout from TMinuit::mnrazz() can not be suppressed
* [[ROOT-9448](https://its.cern.ch/jira/browse/ROOT-9448)] - libNew returns nullptr instead of implementing operator new, has many warnings
* [[ROOT-9420](https://its.cern.ch/jira/browse/ROOT-9420)] - CTest: Fail on warnings in tutorials
* [[ROOT-9395](https://its.cern.ch/jira/browse/ROOT-9395)] - ROOTTEST_ADD_TEST does not complain if source file does not exist
* [[ROOT-9354](https://its.cern.ch/jira/browse/ROOT-9354)] - [TTreeReader] Crash when reading array from in-memory tree
* [[ROOT-9266](https://its.cern.ch/jira/browse/ROOT-9266)] - Cannot unload python code / shared library
* [[ROOT-8991](https://its.cern.ch/jira/browse/ROOT-8991)] - Cling exports buggy include paths to AcLIC
* [[ROOT-8775](https://its.cern.ch/jira/browse/ROOT-8775)] - TTree::MakeSelector can produce invalid C++ code
* [[ROOT-8745](https://its.cern.ch/jira/browse/ROOT-8745)] - Reloading of code that uses R__LOAD_LIBRARY fails
* [[ROOT-8519](https://its.cern.ch/jira/browse/ROOT-8519)] - Bug when use simple math functions in TTree::SetAlias()
* [[ROOT-8271](https://its.cern.ch/jira/browse/ROOT-8271)] - roofit asymmetry plots create corrupted pdf when not providing a custom binning
* [[ROOT-8256](https://its.cern.ch/jira/browse/ROOT-8256)] - Limit to complexity of TTreeFormula? - "Bad Numerical Expression"
* [[ROOT-8240](https://its.cern.ch/jira/browse/ROOT-8240)] - Must not unload or reload cling runtime universe
* [[ROOT-8078](https://its.cern.ch/jira/browse/ROOT-8078)] - Tab completion fails for lambda functions
* [[ROOT-7137](https://its.cern.ch/jira/browse/ROOT-7137)] - Unsafe object ownership issue with TClonesArray/TObjArray
* [[ROOT-6968](https://its.cern.ch/jira/browse/ROOT-6968)] - Interpretation of nparam argument to TMethodCall::SetParamPtrs changed in root 6
* [[ROOT-6931](https://its.cern.ch/jira/browse/ROOT-6931)] - Tab completion of file names in directories with '+'
* [[ROOT-6822](https://its.cern.ch/jira/browse/ROOT-6822)] - Dangerous behavior of TTreeFormula::EvalInstance64
* [[ROOT-6313](https://its.cern.ch/jira/browse/ROOT-6313)] - TClingClassInfo::ClassProperty() might give wrong results
* [[ROOT-5983](https://its.cern.ch/jira/browse/ROOT-5983)] - Add test for wrong data member in TBranchElement
* [[ROOT-5963](https://its.cern.ch/jira/browse/ROOT-5963)] - Re-implement tab completion for ROOT
* [[ROOT-5843](https://its.cern.ch/jira/browse/ROOT-5843)] - List of loaded libraries
* [[ROOT-5439](https://its.cern.ch/jira/browse/ROOT-5439)] - Dump-output of TH1 not showing pointerness of fArray
* [[ROOT-2345](https://its.cern.ch/jira/browse/ROOT-2345)] - Optimize TMatrixDSparse operation kAtA

## Release 6.34.02

Published on December 16, 2024

### Items addressed in this release

This release includes a few minor fixes in RDataFrame and RooFit, besides the item below. Moreover, built-in Davix was patched to build with GCC14 while waiting for the new Davix release.

* [[#17145](https://github.com/root-project/root/issues/17145)] - Distributed RDataFrame cannot deal with same column name in different branches
- [[#17190](https://github.com/root-project/root/issues/17190)] - Compiler error with GCC 14.2.0 related to Davix
* [[#17222](https://github.com/root-project/root/issues/17222)] - Regression in Python ownership for histograms within subdirectories with ROOT 6.34.00
- [[#17223](https://github.com/root-project/root/issues/17223)] - TFileMerger leaves files open resulting in corrupt metadata

## HEAD of the v6-34-00-patches branch
