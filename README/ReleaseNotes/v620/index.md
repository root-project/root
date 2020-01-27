% ROOT Version 6.20 Release Notes
% 2020-01-10
<a name="TopOfPage"></a>

## Introduction

ROOT version 6.20/00 was released on February 26, 2020.

For more information, see:

[http://root.cern](http://root.cern)

The following people have contributed to this new version:

 Kim Albertsson, CERN/ATLAS,\
 Guilherme Amadio, CERN/SFT,\
 Bertrand Bellenot, CERN/SFT,\
 Iliana Betsou, CERN/SFT,\
 Jakob Blomer, CERN/SFT,\
 Brian Bockelman, Nebraska,\
 Rene Brun, CERN/SFT,\
 Philippe Canal, FNAL,\
 Javier Cervantes Villanueva, CERN/SFT,\
 Olivier Couet, CERN/SFT,\
 Alexandra Dobrescu, CERN/SFT,\
 Giulio Eulisse, CERN/ALICE,\
 Massimiliano Galli, CERN/SFT and Unibo,\
 Gerri Ganis, CERN/SFT,\
 Andrei Gheata, CERN/SFT,\
 Hadrien Grasland, CNRS,\
 Enrico Guiraud, CERN/SFT,\
 Stephan Hageboeck, CERN/SFT,\
 Desislava Kalaydjieva, CERN/SFT,\
 Jan Knedlik, GSI,\
 Philip Leindecker, CERN/SFT,\
 Sergey Linev, GSI,\
 Alfonso Luis Castano Marin, UMU,\
 Pere Mato, CERN/SFT,\
 Emmanouil Michalainas, AUTh,\
 Lorenzo Moneta, CERN/SFT,\
 Alja Mrak-Tadel, UCSD/CMS,\
 Axel Naumann, CERN/SFT,\
 Joana Niermann, CERN-SFT, \
 Vincenzo Eduardo Padulano, Bicocca/SFT,\
 Danilo Piparo, CERN/SFT,\
 Fons Rademakers, CERN/SFT,\
 Otto Schaile, Uni-Muenchen,\
 Henry Schreiner, Princeton,\
 Oksana Shadura, Nebraska,\
 Simon Spies, GSI,\
 Matevz Tadel, UCSD/CMS,\
 Yuka Takahashi, Princeton and CERN/SFT,\
 Enric Tejedor Saavedra, CERN/SFT,\
 Vassil Vassilev, Princeton/CMS,\
 Wouter Verkerke, NIKHEF/ATLAS,\
 Stefan Wunsch, CERN/SFT,\
 Luca Zampieri, CERN-SFT, \
 Zhe Zhang, Nebraska

## General

### Splash screen

The venerable splash screen is now disabled by default to make ROOT's startup
faster. Many users already use `root -l` to start ROOT, but this also hides the
useful text banner with version information along with the splash screen. With
this new default, starting up ROOT as just `root` will show only the text banner
instead of the splash screen. The splash screen can still be seen with `root -a`
or in `TBrowser` by opening `Browser Help → About ROOT`.

## Deprecation and Removal

 * rootcling flags `-cint`, `-gccxml`, `-p`, `-r` and `-c` have no effect
   and will be removed in a future release. Please remove them from the rootcling invocations.
 * rootcling legacy cint flags `+P`, `+V` and `+STUB` have no effect and will be
   removed in a future release. Please remove them from the rootcling invocations.
 * genreflex flag `--deep` has no effect and will be removed in a future release. Please remove it
   from the genreflex invocation.
 * rootcling warns if it sees and unrecognized flag (usually coming from the
   CXXFLAGS of the build system). Please remove them from the invocation because
   the warning will become a hard error in the next releases.
 * The empty headers `Gtypes.h` and `Htypes.h` are deprecated. Please include
   `Rtypes.h`
 * TInterpreter::EnableAutoLoading currently does nothing and is deprecated.

## Core Libraries

* Speed-up startup, in particular in case of no or poor network accesibility, by avoiding
  a network access that was used as input to generate a globally unique ID for the current
  process.
* This network access is replaced by a passive scan of the network interface. This
  reduces somewhat the uniqueness of the unique ID as the IP address is no longer
  guaranteed by the DNS server to be unique.   Note that this was already the case when
  the network access (used to look up the hostname and its IP address) failed.


## I/O Libraries

 * TFile: A new bit `TFile::kReproducible` was introduced. It can be enabled by
  specifying the `"reproducible"` url option when creating the file:

       TFile *f = TFile::Open("name.root?reproducible","RECREATE","File title");
  Unlike regular `TFile`s, the content of such file has reproducible binary
  content when writing exactly same data. This achieved by writing pre-defined
  values for creation and modification date of TKey/TDirectory objects and null
  value for TUUID objects inside TFile. As drawback, TRef objects stored in such
  file cannot be read correctly.

* Significantly improved the scaling of hadd tear-down/cleanup-phase in the presence
  of large number histograms and in the presence of large number of directories.
* TMemFile: Apply customization of minimal block size also to the first block.
* Add renaming rule for instances of the math classes from `genvector` and `smatrix` to
  instance for one floating point type (`float`, `double`, `Double32_t`, `Float16_t`) to
  instances for any other floating point type.
* Corrected the application of  `I/O customization rules` when the target classes contained
  typedefs (in particular `Double32_t`)
* Prevent splitting of objects when a `Streamer Function` was explicitly attached to their
 `TClass`.
* In hadd fix verbose level arg parsing
* Allow user to change the type of the content of a TClonesArray.
* Avoid deleted memory access in `MakeProject` and in handling of
 `I/O customization rules`.

### Compression algorithms

* Added a new compression algorithm ZSTD [https://github.com/facebook/zstd.git], a dictionary-type algorithm (LZ77) with large search window and fast implementations of entropy coding stage, using either very fast Finite State Entropy (tANS) or Huffman coding. ZSTD offers a *better compression ratio* and *faster decompression speed* comparing to ZLIB. Its decompression speed is 2x faster then ZLIB and at least 6x faster comparing to LZMA. ZSTD provides a wide range of compression levels, and after evaluation we recommend to use for your purposes compression level 5 (or 6). 
  To use ZSTD please use next settings:
  ```
   root [1] _file0->SetCompressionAlgorithm(ROOT::RCompressionSetting::EAlgorithm::kZSTD)
   root [2] _file0->SetCompressionLevel(ROOT::RCompressionSetting::ELevel::kDefaultZSTD)
  ```
  or
  ```
  root [3] _file0->SetCompressionSettings(505)
  ```
  or
  ```
  root [3] _file0->SetCompressionSettings(ROOT::RCompressionSetting::EDefaults::kUseGeneralPurpose)
  ```

## TTree Libraries

* Prevent a situation in `TTreeFormula` when stale cached information was re-used.
* Prevent a noticeable memory leak when reading uncompressed TTree.

### RDataFrame

- Improved CSV data source
  * Support for 1.0e+10 syntax for type inference of double columns
  * Empty lines are now skipped


## Histogram Libraries

* Allow reading v5 TF1 that were stored memberwise in a TClonesArray.
* Make RHist bin iteration order consistent with that of THx.
* Add RCoordArray constructor from std::array.
* Remove thread-unsafe accessors of RHistConcurrentFill.
* Fix Flush logic of buffered RHist wrappers.
* New class TGraphMultiErrors: A TGraph with asymmetric error bars and multiple y error
  dimensions (author: Simon Spies).

## Math Libraries
* [ROOT::Math::KahanSum](https://root.cern/doc/master/classROOT_1_1Math_1_1KahanSum.html) can use SIMD instructions for faster accumulation

## RooFit Libraries
* **Documentation** Many improvements to the doxygen documentation of RooFit classes.

* **Automatic legends for RooPlot** RooPlot now supports [BuildLegend](https://root.cern.ch/doc/master/classRooPlot.html#acd16cf22aca843f08ef405c93753c512) as a good starting point for a legend.

* **Short prefits** In unbinned fits that take long, good starting values for parameters can be found
by running a short prefit before the final fit. Passing `PrefitDataFraction(0.1)` to [`fitTo()`](https://root.cern.ch/doc/master/classRooAbsPdf.html#a8f802a3a93467d5b7b089e3ccaec0fa8)
will *e.g.* run a prefit on 1% of the data.

* **Iterating over categories** Category classes deriving from [RooAbsCategory](https://root.cern.ch/doc/master/classRooAbsCategory.html), *e.g.* [RooCategory](https://root.cern.ch/doc/master/classRooCategory.html)
now support "natural" iterating using range-based for loops in C++ or Python loops:

      import ROOT
      cat = ROOT.RooCategory("cat", "cat")
      cat.defineType("1Lep", 1)
      cat.defineType("2Lep", 2)
      for state in cat:
        print(state.getVal(), state.GetName())

      (1, '1Lep')
      (2, '2Lep')
* **Asymptotically correct parameter uncertainties**
Added computation of asymptotically correct parameter uncertainties in likelihood fits with event weights. See [arXiv 1911.01303](https://arxiv.org/abs/1911.01303) and [rf611_weightedfits.C](https://root.cern/doc/master/rf611__weightedfits_8C.html).

* **Barlow-Beeston tutorial**
The tutorial [rf709_BarlowBeeston.C](https://root.cern/doc/master/rf709__BarlowBeeston_8C.html) has
been added to RooFit to demonstrate how to incorporate Monte Carlo statistics as systematic
uncertainties into a fit. It demonstrates both the "full" and the "light" method.

* **RooFitMore for GSL** All parts of RooFit that use the GSL (PDFs and integrators) have been moved into the library
`RooFitMore`. It gets enabled automatically with the `MathMore` library (`-Dmathmore=ON`, default).
Note that `-lRooFitMore` might now be required when linking against RooFit.

### Fast function evaluation and vectorisation
A `BatchMode` for faster unbinned fits has been added. By loading data more efficiently,
**unbinned likelihood computations can run about 3x faster** if the PDFs support it.
To enable it, use
```
pdf.fitTo(data, RooFit::BatchMode());
```
Most unbinned PDFs that are shipped with RooFit have been updated to support this mode.

In addition, if ROOT is compiled for a specific architecture, SIMD instructions can be used in PDF
computations. This requires ROOT to be compiled with `-march=native` or *e.g.* `-mavx2` if the hardware
supports it. For maximal performance, ROOT should also be configured with `-Dvdt=ON`.
[VDT](https://github.com/dpiparo/vdt) is a library of fast math functions, which will automatically
be used in RooFit when available.
Depending on the compiler, on the instruction set supported by the CPU and on what kind of PDFs are used,
**PDF evaluations will speed up 5x to 16x**.
For details see [CHEP 2019](https://indico.cern.ch/event/773049/contributions/3476060/).

### New RooFormulaVar / RooGenericPdf
RooFormula has been updated to use ROOT's [TFormula](https://root.cern.ch/doc/master/classTFormula.html).
This means that expressions passed to RooFormulaVar / RooGenericPdf are compiled with `-O2` optimisation
level before they are evaluated. For complicated functions, this might improve the speed of the
computation. Further, functions that are known to the interpreter can be used in the expression passed
to a PDF. The following macro *e.g.* prints the expected result 5.4:
```
  double func(double x, double a) {
    return a*x*x + 1.;
  }

  void testRooFormulaWithClingFunction() {
    RooRealVar x("x", "x", 2, -10, 10);
    RooRealVar a("a", "a", 1.1, -10, 10);

    RooGenericPdf pdf("pdfWithExternalFunc", "func(x, a)", {a, x});
    std::cout << pdf.getVal() << std::endl;
  }
```

#### New PDFs added to RooFit
* [RooHypatia2](https://root.cern.ch/doc/master/classRooHypatia2.html), featuring a hyperbolic, Crystal-ball-like core and two adjustable tails.
* [RooWrapperPdf](https://root.cern.ch/doc/master/classRooWrapperPdf.html), a class that can convert any function (= not normalised) into a PDF by integrating and normalising it.


### RooStats / HistFactory
* To facilitate plotting of pre-fit model uncertainties, gamma parameters now have reasonable pre-fit uncertainties.
* RooStats global config switches are now all accessible in one place,
[`RooStats::GetGlobalRooStatsConfig()`](https://root.cern.ch/doc/master/namespaceRooStats.html#a827f04b74fab219d613f178fa24d0bc9).
* Tools like ToyMCSampler and HypoTestInverter have been stabilised and better validate their inputs to
prevent infinite loops or crashes.
* SPlot now accepts additional arguments for the internal fitting step to *e.g.* add fit ranges or constraint flags.


## 2D Graphics Libraries

 * Provide support of NDC coordinates for `TArrow`.
 * Fix interactive movement of `TLine/TArrow` objects when NDC coordinates are used.
 * Provide `TGraph::MovePoints()` method.
 * New options `RX`and `RY` for TMultiGraph in order to draw reverse axis along X and Y.
 * Combined with the option "Z" the option "CJUST" allows to draw the color palette
   with axis labels justified on the color boundaries (implemented by Otto Schaile).
 * The `TCanvas` Event Status Bar now displays the date and time when the mouse cursor
   is moved over a time axis (implemented by Otto Schaile).
 * Negative values were not painted with option "TEXT" for `TH2Poly`.
 * The Z title was not properly set in `TEfficiency`.
 * Implement `TImage:ls()`
 * `TGraph2D`: X Y and Z titles were not saved by SavePrimitive.
 * In some cases vertical hatches drawn by `TPad::PaintHatches` disappeared.
 * Save the X and Y axis limits in `TMultiGraph::SavePrimitive`.
 * Some markers did not match between PDF and PNG.
 * The MaxDigits attribute was not imported from TAxis to TGaxis.


## 3D Graphics Libraries
### Technology preview release of EVE-7: a server-client implementation of TEve for ROOT-7.
  * Visualization of simple physics objects
     * points, tracks, jets, line-sets, vertex ellipsoid, and generic geometrical shapes
   * Support for handling of physics object collections -- a container of objects of a given class 
     * physics items can be represented in all views:
         * graphical views of any type (3D, RhoZ and RhoPhi)
         * table views with possibility to edit content dynamically
     * coherent selection of collection items across all instantiated views
 * To try it out:
     * build with `-DCMAKE_CXX_STANDARD="17" -Dhttp="ON" -Droot7="ON"`
     * see examples in `tutorials/eve7`

## Language Bindings

### Jupyter Notebook Integration
- When starting Jupyter server with `root --notebook arg1 arg2 ...`, extra arguments can be provided.
  All these arguments delivered as is to jupyter executable and can be used for configuration.
  Like server binding to specific host `root --notebook --ip=hostname`
- Remove `c.NotebookApp.ip = '*'` from default jupyter config. One has to provide ip address for server
  binding using `root --notebook --ip=<hostaddr>` arguments
- Now Jupyter Notebooks will use JSROOT provided with ROOT installation. This allows to use notebooks
  without internet connection (offline).


## JavaScript ROOT
- Provide monitoring capabilities for TGeoManager object. Now geomtry with some tracks can be displayed and
  updated in web browser, using THttpServer monitoring capability like histogram objects.
- JSROOT graphics are now supported in the JupyterLab interface. They are activated in the same way as in
  the classic Jupyter, i.e. by typing at the beginning of a notebook cell:
~~~ {.python}
%jsroot on
~~~

## Tutorials
- Add the "Legacy" category collecting the old tutorials which do not represent any more best practices


## Class Reference Guide
- Images in tutorials can now be displayed à JavaScript thanks to the (js) option
  added next to the directive `\macro_image`
- As the tutorial `palettes.C` is often hit when searching the keyword `palette`
  in the reference guide, a direct link from this example to the full list of
  predefined palettes given in `TColor` has been added.
- Revisited the TSpectrum2 documentation. All the static images have been replaced
  by macros generating images at reference guide build time. These macros have
  been added in the tutorial section of the reference guide.
- The Reference Guide can now be accessed directly from the ROOT prompt thanks to
  a great extension (implemented by Desislava Kalaydjieva) of the `.help` command.
  For example to access the Reference Guide for `TTree` it is enough to type:

      root[0] .help TTree

  To open the reference guide for a function/member:

      root[0] .help TTree::Draw

- Change the layout of the ROOT reference.

## Build, Configuration and Testing Infrastructure

- Make MLP optional via the `-Dmlp={OFF,ON}` switch for CMake
- Make Spectrum optional via the `-Dspectrum={OFF,ON}` switch for CMake
- ROOT now fails to configure when any package is missing
  when `-Dfail-on-missing=ON` is passed to CMake
- The `-Dall=ON` now switches the default value of all optional packages to `ON`
- The options `astiff`, `cling`, `pch`, `thread`, and `explicitlink` have been
  removed and are now ignored. They either had no effect (their value was not
  being used in the build system), or could not be disabled (like `cling` and
  `explicitlink`).
- ROOT library targets now export which C++ standard they were built with via
  the target compile features `cxx_std_11`, `cxx_std_14`, and `cxx_std_17`.
- The file `RootNewMacros.cmake` has been renamed to `RootMacros.cmake`.
  Including the old file by name is deprecated and will generate a warning.
  Including `RootMacros.cmake` is not necessary, as now it is already included
  when calling `find_package(ROOT)`. If you still need to inherit ROOT's compile
  options, however, you may use `include(${ROOT_USE_FILE})` as before.
- ROOT's internal CMake modules (e.g. CheckCompiler.cmake, SetUpLinux.cmake, etc)
  are no longer installed with `make install`. Only the necessary files by
  dependent projects are installed by default now, and they are installed
  directly into the cmake/ directory, not cmake/modules/ as before.
- The macro `ROOT_GENERATE_DICTIONARY()` can now attach the generated source
  file directly to a library target by using the option `MODULE <library>`, where
  `<library>` is an existing library target. This allows the dictionary to inherit
  target properties such as compile options and include directories from the library
  target, even when they are added after the call to `ROOT_GENERATE_DICTIONARY()`.
- The macros `REFLEX_GENERATE_DICTIONARY()` and `ROOT_GENERATE_DICTIONARY()` can
  now have custom extra dependencies added with the options `DEPENDS` and
  `EXTRA_DEPENDENCIES`, respectively.
- CMake build types `DEBUGFULL`, `OPTIMIZED`, and `PROFILE` have been removed.
  They were not in use and did not work in all platforms.
- The default build type is now empty to ensure that `CXXFLAGS` set by the user
  are always respected. If no flags are set, then the build type defaults to
  release.
- The version of Python that ROOT has been built with is now exported in the
  CMake variable `${ROOT_PYTHON_VERSION}`, available after calling `find_package(ROOT)`.

The following builtins have been updated:

- CFITSIO 3.450
- FFTW3 3.3.8
- GSL 2.5
- Intel TBB 2019 U9
- PCRE 8.43
- OpenSSL 1.0.2s
- Vdt 0.4.3
- VecCore 0.6.0
- XRootD 4.10.0
- Zstd 1.4.0
- LZ4 1.9.2

## PyROOT

### Current PyROOT

- Several changes for forward compatibility with experimental PyROOT and new Cppyy have been added:
  * Template instantiation can be done with square brackets. The parenthesis syntax is deprecated.
~~~ {.python}
my_templated_function['int','double']()  # new syntax
my_templated_function('int','double')()  # old sytax, throws a deprecation warning
~~~
  * When converting `None` to a null pointer in C++, a deprecation warning is issued.
  * When using `buffer.SetSize` a deprecation warning is issued, the forward compatible alternative is `buffer.reshape`.
  * When using `ROOT.Long` or `ROOT.Double`, a deprecation warning is issued in favour of their equivalent `ctypes` types
(`c_long`, `c_int`, `c_double`)
  * Added the forward compatible names `as_cobject` and `bind_object` for `AsCObject` and `BindObject`, respectively.
  * nullptr is also accessible as `cppyy.nullptr`, not only as `cppyy.gbl.nullptr`.
  * Pythonization functions (e.g. `add_pythonization`) are accessible via `cppyy.py`.
  * Some attributes of Python proxies have been added with the name they have in the new Cppyy
(`__creates__`, `__mempolicy__`, `__release_gil__` for function proxies, `__smartptr__` for object proxies).
- The support for enums (both scoped and non-scoped) was improved. Now, when creating an enum from Python,
its underlying type is obtained by PyROOT.
- Added support for non-ASCII Python strings (e.g. UTF-8) to `std::string` and `C string`.
- Added converters from `bytes` to `std::string` and to C string.
- Added compatibility of STL iterators with GCC9.
- Added support for templated methods with reference parameters.
- Introduced two teardown modes: soft and hard. The soft one only clears the proxied objects, while the hard one also
shuts down the interpreter.

### Experimental PyROOT

- MultiPython: build and install PyROOT with multiple Python versions.
  - Build:
  ~~~ {.bash}
  cmake -DPYTHON_EXECUTABLE=/path/to/first/Python/installation /path/to/ROOT/source
  cmake --build .
  cmake -DPYTHON_EXECUTABLE=/path/to/second/Python/installation /path/to/ROOT/source
  cmake --build .
  ( ... )
  ~~~
  - Source a specific built version (last one picked as default):
  ~~~ {.bash}
  ROOT_PYTHON_VERSION=X.Y source /path/to/bin/thisroot.sh
  ~~~
  - PyROOT installation directory can be customized:
  ~~~ {.bash}
  cmake -DCMAKE_INSTALL_PYROOTDIR=/path/to/PyROOT/install/dir /path/to/ROOT/source
  cmake --build .
  make install
  ~~~
- Updated cppyy packages to the following versions:
  * cppyy: cppyy-1.5.3
  * cppyy_backend: clingwrapper-1.10.3
  * CPyCppyy: CPyCppyy-1.9.3
- Introduced two teardown modes: soft and hard. The soft one only clears the proxied objects, while the hard one also
shuts down the interpreter.
- A few changes for backward compatibility with current PyROOT have been added:
  * Added `MakeNullPointer(klass)` as `bind_object(0,klass)`
  * Provided `BindObject` and `AsCObject`
- `ROOT.Long` and `ROOT.Double` are no longer in the API of PyROOT, since new Cppyy requires to use their equivalent
`ctypes` types (`c_long`, `c_int`, `c_double`).
- Added TPython.
- Added support for `from ROOT import *` in Python2 (only).

## TMVA
 - Introduce `RTensor` class, a container for multi-dimensional arrays similar to NumPy arrays
 - Add `AsTensor` to convert data from a `RDataFrame` to a `RTensor`
 - Introduce a fast tree inference engine able to generate optimized inference code using just-in-time compilation of model parameters
 - New experimental TMVA reader interface `RReader` following a sklearn-like API in C++ and Python
 - New experimental interface for preprocessing methods (`RStandardScaler`)
 - New GPU implmentation of convolutional layer using the cuDNN library. It is used as default when the cuDNN library is installed in the system 

### Bugs and Issues fixed in this release

* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9817'>ROOT-9817</a>] Batch evaluations in RooxxxPDFs
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9818'>ROOT-9818</a>] Generic `batchEvaluate` in `RooAbsPdf`
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9819'>ROOT-9819</a>] `batchEvaluate` functions for heavily used PDFs
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9882'>ROOT-9882</a>] Use VDT Math functions
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10048'>ROOT-10048</a>] [DF] Improve error handling in Profile1D and Profile2D
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10128'>ROOT-10128</a>] Fix GCC 9 warnings in core libraries
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10129'>ROOT-10129</a>] Fix GCC 9 warnings in cling/metacling
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10134'>ROOT-10134</a>] Fix GCC 9 warnings in I/O libraries
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10137'>ROOT-10137</a>] Fix GCC 9 warnings in histogram libraries
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10307'>ROOT-10307</a>] Implementation of Hypatia PDF
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-4756'>ROOT-4756</a>] RooFit: Fitting in multiple disjoint ranges broken
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-5976'>ROOT-5976</a>] tutorials/roofit/rf313_paramranges.C division by zero at RooIntegrator1D.cxx:285
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-6008'>ROOT-6008</a>] `RooAddPdf::_coefCache` does not check bounds
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-6260'>ROOT-6260</a>] data looses their name when copying a RooWorkspace
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-6988'>ROOT-6988</a>] Make enum's underlying type available
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-7139'>ROOT-7139</a>] Seg Fault in `TH2Poly` -> `TH1::Divide`
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-7240'>ROOT-7240</a>] namespace of class enum from dictionaries
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-7520'>ROOT-7520</a>] Segmentation fault when loading a `RooLinearVar` from a `RooWorkspace`
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-7836'>ROOT-7836</a>] running roottest via ctest popules ~/.root_hist
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-8291'>ROOT-8291</a>] RooFormulaVar in RooWorkspace after saving to ROOT file
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-8412'>ROOT-8412</a>] RMS calculation with negative bin content
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-8522'>ROOT-8522</a>] `RooDataHist`'s assignment operator seems to be broken
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-8932'>ROOT-8932</a>] Impossible to plot multiple filled PDFs using `RooFit::AddTo()`
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-8936'>ROOT-8936</a>] ROOT's dictionary can get "confused" when asking for it with the "wrong name"
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9112'>ROOT-9112</a>] rootcing wildcarding should ignore private/protected inner classes
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9385'>ROOT-9385</a>] CMake unexpected behavior with `-Dfail-on-missing=ON`
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9449'>ROOT-9449</a>] cmake `-Dexplicitlink=OFF` configuration does not compile
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9647'>ROOT-9647</a>] Cannot configure with `-Droottest=ON` when using a release tarball
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9661'>ROOT-9661</a>] Cannot configure ROOT with `builtin_clang=OFF`
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9678'>ROOT-9678</a>] Builtin TBB fails to compile on Mac OS when CUDA toolkit is installed
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9708'>ROOT-9708</a>] Automatic first checkout of roottest fails if not on master
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9741'>ROOT-9741</a>] ROOT fails to read objects derived from `TH2Poly`
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9755'>ROOT-9755</a>] Linker error - Missing symbols from libcurl in fitsio package
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9758'>ROOT-9758</a>] Installed ROOT cannot find its libraries
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9828'>ROOT-9828</a>] ROOT installs internal CMake modules
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9842'>ROOT-9842</a>] Build Error: fatal error: cyclic dependency in module `ROOTGraphicsPrimitives`
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9965'>ROOT-9965</a>] Segfault with TEve on ROOT 6.16.00 from LCG 95
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10031'>ROOT-10031</a>] [Exp PyROOT] Cppyy error when accessing typedef in a struct
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10050'>ROOT-10050</a>] [Exp PyROOT] `TPython::LoadMacro` is broken in Cppyy 
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10094'>ROOT-10094</a>] ROOT Cross-hatch for `TGraph` does not fill completely
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10096'>ROOT-10096</a>] [Exp PyROOT] Lack of memory management for `TFile`-owned objects
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10144'>ROOT-10144</a>] `TFormula` v5 reads/writes after array
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10188'>ROOT-10188</a>] `RooDataSet`s with large weight array cannot be serialised
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10193'>ROOT-10193</a>] Using an undeclared identifier makes ROOT crash
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10194'>ROOT-10194</a>] using ofstream in the prompt makes ROOT crash
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10197'>ROOT-10197</a>] CUDA Builds broken after adding dependency in TMVA to Vecops
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10211'>ROOT-10211</a>] GCC 9: `TStyle` has user defined copy constructor, but no operator=
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10216'>ROOT-10216</a>] `TFile::MakeProject` severely broken
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10221'>ROOT-10221</a>] `cling::Value` broken if placement-new constructor fails
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10222'>ROOT-10222</a>] PyROOT converts Python 3 str → `const char*` and not bytes → `const char*`
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10224'>ROOT-10224</a>] Crash in opening TFile 
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10227'>ROOT-10227</a>] [TreeProcMT] Last cluster in processed `TTree` is dropped under specific circumstances
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10228'>ROOT-10228</a>] `TCanvas.SaveAs` saves png files only in batch mode
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10241'>ROOT-10241</a>] Header `TBulkBranchRead.hxx` recursively includes `TBranch.h`
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10242'>ROOT-10242</a>] `RooMCStudy` crashes when requesting pulls for parameters that don't have pulls
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10244'>ROOT-10244</a>] Nested `std::pair` written with 6.14/GCC 8.2 not readable with 6.18/GCC 8.3
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10245'>ROOT-10245</a>] Template proxy error when invoking reset on `std::shared_ptr`
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10250'>ROOT-10250</a>] roottest picks up installed ROOT instead of the ROOT being built/tested
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10254'>ROOT-10254</a>] FoldFileOutput breaks output log
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10255'>ROOT-10255</a>] `TCutG` copy constructor is wrong
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10257'>ROOT-10257</a>] ROOT does not respect users' CXXFLAGS
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10259'>ROOT-10259</a>] `RooDataSet::add` has side effects on event weight error
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10280'>ROOT-10280</a>] `TH2Poly::Add` wrongly adds bin content
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10290'>ROOT-10290</a>] MacOS: `TRint` doesn't catch exceptions
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10292'>ROOT-10292</a>] PYROOT: "TypeError: can not resolve method template call"
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10293'>ROOT-10293</a>] Missing CMake information about Python version
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10295'>ROOT-10295</a>] [PyROOT experimental] Segfault at destruction time of a `TFile`/`TTree` combination
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10297'>ROOT-10297</a>] Crash in `RooDataHist::sum`
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10305'>ROOT-10305</a>] Leading underscore in branch name causes a crash
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10306'>ROOT-10306</a>] Failure with `std::unique_ptr` in STL container
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10313'>ROOT-10313</a>] [RCsvDS] Double column deduced as `std::string`
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10323'>ROOT-10323</a>] Segfault in PyROOT at exit
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10331'>ROOT-10331</a>] ROOT behavior changes when build with Debug and Release
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10333'>ROOT-10333</a>] seg fault with `std::shared_ptr`
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10334'>ROOT-10334</a>] `TFile::GetBestBuffer` casting to `Int`_t Overflows
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10337'>ROOT-10337</a>] `TClassEdit::ResolveTypedef` leaves a spurious "std"
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10339'>ROOT-10339</a>] First of the tree tutorials crashes when trying to histogram a leaf
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10340'>ROOT-10340</a>] Segmentation violation associated with `TSocket::Recv`
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10343'>ROOT-10343</a>] illegal memory overwrite in `TMemberInspector::TParentBuf::Append`
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10344'>ROOT-10344</a>] stack overflow in rootcling on anonymous union
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10358'>ROOT-10358</a>] RooRealVar's assignment operator is broken
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10360'>ROOT-10360</a>] `TFitResult->Scan()` crashes when multi-threading is enabled
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10367'>ROOT-10367</a>] `RooFormulaVar` with categories crash when plotted
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10369'>ROOT-10369</a>] Inverted logic in assert in `TEveVectorT` on Windows
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10370'>ROOT-10370</a>] Builtin CFITSIO has several security vulnerabilities
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10375'>ROOT-10375</a>] V523 The `then` statement is equivalent to the `else` statement. MethodMLP.cxx 423
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10391'>ROOT-10391</a>] Marker 29 looks different in png and pdf output
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10411'>ROOT-10411</a>] `RooFormula` doesn't compile when arguments are renamed
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10413'>ROOT-10413</a>] `RooDataSet` Import Tree Branch with long name 
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10426'>ROOT-10426</a>] Crash in recursive compilation / new autoloading
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10449'>ROOT-10449</a>] `Measurement::PrintXML()` crashes on 6.18+
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10452'>ROOT-10452</a>] ROOT configured with `-Dgnuinstall=ON` doesn't work when installed with make install
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10469'>ROOT-10469</a>] CMS: pyroot teardown issue
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10473'>ROOT-10473</a>] `RooAbsData::convertToTreeStore` forgets event weights
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10474'>ROOT-10474</a>] Cannot iterate on `std::map` in Python using gcc9 builds
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10475'>ROOT-10475</a>] Ownership of `TTree` is shared between RooFit and file after writing a `RooDataSet`
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10481'>ROOT-10481</a>] `TBinomialEfficiencyFitter::Fit` ignores the parameter limits
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10491'>ROOT-10491</a>] `AsNumpy` fails with Boolean columns
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10495'>ROOT-10495</a>] `PyArrayObject` redefined in TMVA/PyMethodBase.h with different type
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10504'>ROOT-10504</a>] Mixed dictionary / autoparse fails Cling/Clang assertion
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10511'>ROOT-10511</a>] clang assertion in debug builds with Python
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10512'>ROOT-10512</a>] Implicit conversion from `string_view` to `TString` breaks existing code
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10517'>ROOT-10517</a>] `RooFit::CutRange` is buggy with multiple range
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10518'>ROOT-10518</a>] `chi2FitTo` and `fitTo` are not working properly with multiple ranges
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10526'>ROOT-10526</a>] Segfault in [some] `std::vector<T, custom_alloc<T>>` I/O
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10528'>ROOT-10528</a>] The `TClass` for `HepMC::GenVertex` is being unloaded when in state 3
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10552'>ROOT-10552</a>] `TStreamerInfo` removed from the file when new object is written
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10557'>ROOT-10557</a>] Wrong results when using `TTree::Draw`/`Scan` on specific entries of vectors of vectors
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10574'>ROOT-10574</a>] `TClassEdit::GetNormalizedName` is wrong on OSX
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10167'>ROOT-10167</a>] `root-config --python-version` returns an empty string
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-8777'>ROOT-8777</a>] Making it possible to turn off the `stdout` message in `RooMCStudy`
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-9332'>ROOT-9332</a>] `TFileMerger`/`hadd` using restricted sources list on recursive call ?
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-8646'>ROOT-8646</a>] Roo(Stats,Fit) - use new `TFormula`
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10127'>ROOT-10127</a>] Fix GCC 9 Warnings
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-7549'>ROOT-7549</a>] Can't make CMake dictionary generation functional with Ninja
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-8575'>ROOT-8575</a>] `ROOT_GENERATE_DICTIONARY` interface for CMake Usage Requirements/Generator Expressions
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10022'>ROOT-10022</a>] [DF] Add support for `TBranchObjects` (e.g. branches containing `TH2F`)
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10103'>ROOT-10103</a>] tutorial nbviewer links point to root.cern.ch, not root.cern
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10164'>ROOT-10164</a>] Replace v5 `TFormula` in RooFit by JITted `TFormula`
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10173'>ROOT-10173</a>] Re-enable `RooJohnson` unit test
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10230'>ROOT-10230</a>] `TPythia8`: Add option to disable banner in constructor
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10268'>ROOT-10268</a>] [PyROOT] Allow to pickle the numpy array wrapper used in `AsNumpy`
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10299'>ROOT-10299</a>] `Info` symbols found twice in Pythia and ROOT
* [<a href='https://sft.its.cern.ch/jira/browse/ROOT-10310'>ROOT-10310</a>] Reverse axis drawing options RX and RY should be also allowed in `TMultiGraph`



## HEAD of the v6-20-00-patches branch

These changes will be part of a future 6.20/02.

* None so far.
