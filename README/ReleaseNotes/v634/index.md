% ROOT Version 6.34 Release Notes
% 2024-11
<a name="TopOfPage"></a>

## Introduction

ROOT version 6.34.00 is scheduled for release at the end of November 2024.

For more information, see:

[http://root.cern](http://root.cern)

The following people have contributed to this new version:

 Guilherme Amadio, CERN/IT,\
 Bertrand Bellenot, CERN/EP-SFT,\
 Jakob Blomer, CERN/EP-SFT,\
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


The release 6.34 is a short term support cycle release: it will be superseded by ROOT 6.36 in May 2025 and patches will be provided until July 2025. It is not intended to be used for data taking.

## Removal and Deprecation

The following interfaces have been removed:

- The `RooAbsReal::plotSliceOn()` function that was deprecated since at least ROOT 6 was removed. Use `plotOn(frame,Slice(...))` instead.
- Multiple overloads of internal Minuit 2 constructors and functions have been removed. If your code fails to compile, you can easily change to another overload that takes a `MnUserParameterState`, which is a change backwards compatible with older ROOT versions.

The following interfaces are deprecated and will be removed in future releases:

- The `RooTemplateProxy` constructors that take a `proxyOwnsArg` parameter to manually pass ownership are deprecated and replaced by a new constructor that takes ownership via `std::unique_ptr<T>`. They will be removed in ROOT 6.36.
- Several RooFit legacy functions are deprecated and will be removed in ROOT 6.36 (see section "RooFit libraries")
- The `int ROOT::CompressionSettings(ROOT::ECompressionAlgorithm algorithm, int compressionLevel)` function is deprecated and will be removed in ROOT 6.36. Please use `int CompressionSettings(RCompressionSetting::EAlgorithm::EValues algorithm, int compressionLevel)` instead.
- The `void R__zip(int cxlevel, int *srcsize, char *src, int *tgtsize, char *tgt, int *irep)`  is deprecated and will be removed in ROOT 6.36. Please use `void R__zipMultipleAlgorithm(int cxlevel, int *srcsize, char *src, int *tgtsize, char *tgt, int *irep, ROOT::RCompressionSetting::EAlgorithm::EValues algorithm)` instead.
- The `Bool_t TGeoShape::AreOverlapping(const TGeoBBox *box1, const TGeoMatrix *mat1, const TGeoBBox *box2, const TGeoMatrix *mat2)` is deprecated and will be removed in ROOT 6.36.
- The `TPython::Eval()` function that is deprecated scheduled for removal in ROOT 6.36.


## Core Libraries

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
* Many additional bug fixes and improvements.

## TTree Libraries

## RDataFrame

* The `GetColumnNames` function to retrieve the number of available column names in the RDataFrame object is now also
  usable from a node of a distributed computation graph. This makes the generation of said computation graph slightly
  less lazy than before. Notably, it used to be the case that a distributed computation graph could be defined with
  code that was not yet available on the user's local application, but that would only become available in the
  distributed worker. Now a call such as `df.Define("mycol", "return run_my_fun();")` needs to be at least declarable
  to the interpreter also locally so that the column can be properly tracked.

## Histogram Libraries

### Upgrade TUnfold to version 17.9

The [TUnfold package](https://www.desy.de/~sschmitt/tunfold.html) inside ROOT is upgraded from version 17.6 to version 17.9.

## Math Libraries

### Minuit2 

* **Usage of `std::span<const double>`in the interface**: To avoid forcing the user to do manual memory allocations via `std::vector`, the interfaces of Minuit 2 function adapter classes like `ROOT::Minuit2::FCNBase` or `ROOT::Minuit2::FCNGradientBase` were changed to accept `std::span<const double>` arguments instead of `std::vector<double> const&`.
This should have minimal impact on users, since one should usual use Minuit 2 via the `ROOT::Math::Minimizer` interface, which is unchanged.

* **Initial error/covariance matrix values for Hessian matrix**: Initial error/covariance matrix values can be passed for initializating the Hessian matrix to be used in minimization algorithms by attaching the covariance matrix to the `ROOT::Minuit2::MnUserParameterState` instance used for seeding via the method `AddCovariance(const MnUserCovariance &);`.

## RooFit Libraries

### Miscellaneous

* Setting `useHashMapForFind(true)` is not supported for RooArgLists anymore, since hash-assisted finding by name hash can be ambiguous: a RooArgList is allowed to have different elements with the same name. If you want to do fast lookups by name, convert your RooArgList to a RooArgSet.

* The function `RooFit::bindFunction()` now supports arbitrary many input variables when binding a Python function.

* The `ExportOnly()` attribute of the `RooStats::HistFactory::Measurement` object is now switched on by default, and the associated getter and setter functions are deprecated. They will be removed in ROOT 6.36. If you want to fit the model as well instead of just exporting it to a RooWorkspace, please do so with your own code as demonstrated in the `hf001` tutorial.

* Initial error values can be used for initializating the Hessian matrix to be used in Minuit2 minimization algorithms by setting the `RooMinimizer::Config` option `setInitialCovariance` to `true`. These values correspond to the diagonal entries of the initial covariance matrix.

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

## TMVA
### SOFIE
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

## 2D Graphics Libraries


## 3D Graphics Libraries


## Geometry Libraries


## Database Libraries


## Networking Libraries


## GUI Libraries


## Montecarlo Libraries


## PROOF Libraries


## PyROOT

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

## Language Bindings


## JavaScript ROOT


## Tutorials


## Class Reference Guide


## Build, Configuration and Testing Infrastructure

- Coverage of the CI was greatly improved, with Clang builds, Alma9 ARM64 and Alma9 x86 NVidia GPU builds were added to the CI

The following builtins have been updated:

- daviX 0.8.7
- XRootD 5.7.1

## Bugs and Issues fixed in this release

More than 160 items were addressed for this release. The full list is:

* [[#16834](https://github.com/root-project/root/issues/16834)] - `RFieldBase::Create` does not enforce valid field names
* [[#16826](https://github.com/root-project/root/issues/16826)] - RNTuple unexpected "field iteration over empty fields is unsupported"
* [[#16796](https://github.com/root-project/root/issues/16796)] - RooBinSamplingPdf does not forward expectedEventsFunc creation calls
* [[#16784](https://github.com/root-project/root/issues/16784)] - Remove default value of p from TH1::GetQuantiles() as is the case with TF1::GetQuantiles
* [[#16771](https://github.com/root-project/root/issues/16771)] - copying a default constructed `TH2Poly` fails.
* [[#16753](https://github.com/root-project/root/issues/16753)] - [ntuple] Free uncompressed page buffers in RPageSinkBuf with IMT
* [[#16752](https://github.com/root-project/root/issues/16752)] - [ntuple] Copy sealed page in RPageSinkBuf after compression
* [[#16736](https://github.com/root-project/root/issues/16736)] - Please improve documentation and/or argument names for TH1::GetQuantiles()
* [[#16715](https://github.com/root-project/root/issues/16715)] - TMVA fails to link to cudnn
* [[#16680](https://github.com/root-project/root/issues/16680)] - TMVA/Sofie tutorials used same name for generated files bur are run in parallel.
* [[#16647](https://github.com/root-project/root/issues/16647)] - ROOT_ADD_PYUNITTEST and ROOT_ADD_GTEST are naming test inconsitently.
* [[#16600](https://github.com/root-project/root/issues/16600)] - TMVA RReader not multithread safe
* [[#16562](https://github.com/root-project/root/issues/16562)] - TTreeViewer save session absolute path
* [[#16523](https://github.com/root-project/root/issues/16523)] - OpenGL doesn't work on macosx
* [[#16513](https://github.com/root-project/root/issues/16513)] - [ntuple] Clarifications about late schema extension
* [[#16479](https://github.com/root-project/root/issues/16479)] - Add THStack/TH1 constructor for TRatioPlot
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
* [[#12426](https://github.com/root-project/root/issues/12426)] - RNTuple endian issues
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
* [[ROOT-10975](https://its.cern.ch/jira/browse/ROOT-10975)] - ACLiC should make rootcling warnings visible
* [[ROOT-10865](https://its.cern.ch/jira/browse/ROOT-10865)] - [RVec] No Doxygen documentation about arithmetic operators
* [[ROOT-10539](https://its.cern.ch/jira/browse/ROOT-10539)] - Slow tutorials/dataframe/df027_SQliteDependencyOverVersion.C
* [[ROOT-9918](https://its.cern.ch/jira/browse/ROOT-9918)] - Crash TMVA by running (unused?) public function from TMVA::Factory
* [[ROOT-9705](https://its.cern.ch/jira/browse/ROOT-9705)] - flag to disable (root)test(s) that uses remote files
* [[ROOT-9673](https://its.cern.ch/jira/browse/ROOT-9673)] - Printout from TMinuit::mnrazz() can not be suppressed
* [[ROOT-9420](https://its.cern.ch/jira/browse/ROOT-9420)] - CTest: Fail on warnings in tutorials
* [[ROOT-9354](https://its.cern.ch/jira/browse/ROOT-9354)] - [TTreeReader] Crash when reading array from in-memory tree
* [[ROOT-8526](https://its.cern.ch/jira/browse/ROOT-8526)] - Limit to complexity of TTreeFormula? - "Bad Numerical Expression"
* [[ROOT-8271](https://its.cern.ch/jira/browse/ROOT-8271)] - roofit asymmetry plots create corrupted pdf when not providing a custom binning
* [[ROOT-5843](https://its.cern.ch/jira/browse/ROOT-5843)] - List of loaded libraries
* [[ROOT-6968](https://its.cern.ch/jira/browse/ROOT-6968)] - Interpretation of nparam argument to TMethodCall::SetParamPtrs changed in root 6
* [[ROOT-6822](https://its.cern.ch/jira/browse/ROOT-6822)] - Dangerous behavior of TTreeFormula::EvalInstance64
* [[ROOT-2811](https://its.cern.ch/jira/browse/ROOT-2811)] - proper support for ndim numerical arrays




