% ROOT Version 6.30 Release Notes
% 2023-11-06
<a name="TopOfPage"></a>

## Introduction

ROOT version 6.30/00 was release on November 6, 2023.

For more information, see:

[http://root.cern](http://root.cern)

The following people have contributed to this new version:

 Guilherme Amadio, CERN/IT,\
 Bertrand Bellenot, CERN/EP-SFT,\
 Jakob Blomer, CERN/EP-SFT,\
 Rene Brun,\
 Carsten Burgard, TU Dortmund,\
 Will Buttinger, Rutherford Appleton Lab,\
 Philippe Canal, FNAL,\
 Olivier Couet, CERN/EP-SFT,\
 Marta Czurylo, CERN/EP-SFT,\
 Mattias Ellert, Uppsala Uni,\
 Edward Finkelstein, JGU Mainz,\
 Gerri Ganis, CERN/EP-SFT,\
 Florine de Geus, CERN/ATLAS,\
 Andrei Gheata, CERN/EP-SFT,\
 Enrico Guiraud, CERN/EP-SFT and Princeton,\
 Jonas Hahnfeld, CERN/EP-SFT,\
 Fernando Hueso González, CSIC/UV,\
 Baidyanath Kundu, CERN/EP-SFT and Princeton,\
 Giovanna Lazzari Miotto, CERN/EP-SFT,\
 Sergey Linev, GSI,\
 Jerry Ling, Harvard Uni,\
 Javier Lopez-Gomez, CERN/EP-SFT,\
 Pere Mato, CERN/EP-SFT,\
 Lorenzo Moneta, CERN/EP-SFT,\
 Ole Morud, CERN/EP-SFT,\
 Alja Mrak Tadel, UCSD/CMS,\
 Axel Naumann, CERN/EP-SFT,\
 Vincenzo Eduardo Padulano, CERN/EP-SFT,\
 Danilo Piparo, CERN/EP-SFT,\
 [QuillPusher](https://github.com/QuillPusher), [Compiler Research Group](https://compiler-research.org/team/),\
 Fons Rademakers, CERN/IT,\
 Jonas Rembser, CERN/EP-SFT,\
 Sanjiban Sengupta, CERN/EP-SFT,\
 Garima Singh, CERN/EP-SFT and Princeton,\
 Enric Tejedor Saavedra, CERN/EP-SFT,\
 Matevz Tadel, UCSD/CMS,\
 Vassil Vassilev, Princeton/CMS,\
 Wouter Verkerke, NIKHEF/Atlas,

## Platform support

- GCC 12 is now supported.
- macOS 14 is now supported.

## Deprecation and Removal

- The minimum C++ standard supported by ROOT is now C++17.
- Support for Python 2 is now deprecated and it will be removed in next release 6.32.

### Deprecated and removed ROOT modules
The following previously deprecated build options have been removed:

- alien
- gfal
- gsl_shared
- jemalloc
- monalisa
- pyroot_legacy
- tcmalloc
- xproofd

The following build options have now been deprecated and will be removed in the future v6.32:

- cxxmodules
- exceptions
- oracle
- pythia6
- pythia6_nolink
- pyroot-python2

Please let us know at [rootdev@cern.ch](mailto:rootdev@cern.ch) if their planned removal would cause problems for you!


### Deprecated and removed interfaces

- `ROOT::RDF::RResultHandle::GetResultPtr` has been deprecated. Please use `RResultPtr` directly instead and only cast to `RResultHandle` in order to call `ROOT::RDF::RunGraphs`.
- The RDataFrame factory functions `MakeCsvDataFrame`, `MakeArrowDataFrame`, `MakeNTupleDataFrame` and `MakeSqliteDataFrame` that were deprecated in v6.28 have been removed. Use `FromCSV`, `FromArrow`, `FromRNTuple` or `FromSqlite` instead.
- The TStorage reallocation routine without a size (`TStorage::ReAlloc(void *ovp, size_t size`) and heap related routines (`TStorage::AddToHeap`, `TStorage::IsOnHeap`, `TStorage::GetHeapBegin`, `TStorage::GetHeapEnd`) that were deprecated in v6.02/00 have been removed.
- The deprecated `Format(const char* option, int sigDigits)` option for `RooAbsPdf::paramOn()` was removed. Please use the `Format(const char* option, ...)` overload that takes command arguments.
- The deprecated `RooAbsPdf::paramOn()` overload that directly takes a formatting string was removed. Please take the overload that uses command arguments.
- The `RooCatType` class was deprecated in ROOT 6.22 and its original `RooCatType.h` header is now removed. If you still need access to this class, please include `RooFitLegacy/RooCatTypeLegacy.h`.
- The `RooAbsString` that was only an alias for `RooStringVar` got removed.
- The `RooDataWeightedAverage` is now deprecated and will be removed in 6.32. It was only supposed to be an implementation detail of RooFits plotting that is now not necessary anymore.
- The `RooSpan` class was removed and its place in the implementation details of RooFit is now taken by `std::span`.
- The `RooAbsArg::isCloneOf()` and `RooAbsArg::getCloningAncestors()` member functions were removed because they didn't work (always returned `false` and an empty list respectively)
- `ROOT::Math::KelvinFunctions` had an incompatible license and needed to be removed without deprecation.
- The use of `ROOT_GIT_BRANCH` and `ROOT_GIT_COMMIT` have been deprecated in favor of parsing `etc/gitinfo.txt`. This later file is now generated as part of the build of ROOT; `RGitCommit.h` (defining `ROOT_GIT_BRANCH` and `ROOT_GIT_COMMIT`) is not updated anymore. This simplifies ROOT's build and release procedure.

## Core Libraries

- Increase thread-safety in parts of core libraries (TCling, TClingMethodInfo, TClingTypeInfo, TFunction) to allow for parallel workflows using RDataFrame in multiple C++ `std::thread`s.
- The maximum number of threads used by ROOT's thread pools can now be limited by the environment variable `ROOT_MAX_THREADS`.

## I/O Libraries

- Improve parsing of input argument to `TChain::Add`. Now it supports the case of globbing files while also using the `?#` token to specify the tree name.

## RDataFrame
- Instead of returning nothing, `ROOT::RDF::RunGraphs` now returns the number of separate computation graphs that have been run.

- Introduce [`ProgressBar`](https://root.cern/doc/master/classROOT_1_1RDataFrame.html#progressbar) feature that can be added to any RDataFrame program.

- The `RDatasetSpec` class and its users now employ the concept of 'sample' rather than the original naming 'group' for groups of files with associated metadata.

- `df106_HiggsToFourLeptons` tutorials (both python and C++) now showcase the `ProgressBar`. They now use `FromSpec` to define multiple samples and `Vary` for systematic variations.

### Distributed RDataFrame

- Vastly improve runtime performance when using an RDataFrame with simulated dataset, i.e. `RDataFrame(nentries)`, by removing usage of `Range` operation to define the per-task entry range.

- Explicitly error out when trying to process a TTree with a TTreeIndex in distributed mode. The feature is currently not supported.

- JITting the RDataFrame computation graph now only happens once per worker process, not once per task. This greatly reduces memory usage and runtime overhead at of each task.

## TTree Libraries

Many bug fixes, improvements for multi-threaded usage, and optimizations.

## RNTuple
ROOT's experimental successor of TTree has seen a large number of updates during the last few months. Specifically, v6.30 includes the following changes:

- Support for custom ROOT I/O rules that target transient members of a user-defined class (see PR [#11944](https://github.com/root-project/root/pull/11944)).  If a rule only targets transient members and it was working in TTree, it should work unmodified in RNTuple.

- Improved support for user-defined classes that behave as a collection.  Specifically, RNTuple now relies on the iterator interface defined in `TVirtualCollectionProxy` (see PR [#12380](https://github.com/root-project/root/pull/12380) for details).
Note that associative collections are not yet supported.

- Support for new field types: `std::bitset<N>`, `std::unique_ptr<T>`, `std::set<T>`, `Double32_t`, scoped and unscoped enums with dictionary.

- Full support for late model extension, which allows the RNTuple model to be extended after a `RNTupleWriter` has been created from the initial model (see PR [#12376](https://github.com/root-project/root/pull/12376)).
New top-level fields can be created at any time during the writing process.
On read-back, zero-initialized values are read for entries before the field was first seen.
The example below illustrates the use of this feature.
```c++
auto model = RNTupleModel::Create();
auto fieldPt = model->MakeField<float>("pt", 42.0);
auto ntuple = RNTupleWriter::Recreate(std::move(model), "myNTuple", "out.ntuple");
ntuple->Fill();

auto modelUpdater = ntuple->CreateModelUpdater();
modelUpdater->BeginUpdate();
std::array<double, 2> fieldArray;
modelUpdater->AddField<std::array<double, 2>>("array", &fieldArray);
modelUpdater->CommitUpdate();

// After this point, entries will have a new field of type `std::array<double, 2>`
ntuple->Fill();
```

- Support for alternative column representations (Split / Zigzag encoding).  These encodings allow for better compression and are used by default if compression is enabled.
Alternatively, users can pick a different column representation for a field by calling `RFieldBase::SetColumnRepresentative()`.

- RNTuple now defaults to 64bit offset columns, which allow for representing large collections.
RNTuple can still use 32bit offset columns, e.g.
```c++
RNTupleWriteOptions options;
options.SetHasSmallClusters(true);
auto writer = RNTupleWriter::Recreate(std::move(model), "myNTuple", "out.ntuple");
```

- Support for projected fields, i.e. exposing other fields' data as a different (compatible) C++ type.
Users should provide a mapping function that maps each projected subfield in the tree to the underlying real field, e.g.
```c++
auto model = RNTupleModel::Create();
auto fvec = model->MakeField<std::vector<float>>("vec");

auto aliasVec = RFieldBase::Create("aliasVec", "std::vector<float>").Unwrap();
model->AddProjectedField(std::move(aliasVec), [](const std::string &fieldName) {
   if (fieldName == "aliasVec") return "vec";
   else                         return "vec._0";
});
```
Projected fields are stored as part of the metadata.

- Improvements on the internal `RField` value API.  The `RFieldValue` class has been deprecated in favor of `RField::Value` and the related interfaces have changed accordingly (see [#13219](https://github.com/root-project/root/pull/13219) and [#13264](https://github.com/root-project/root/pull/13264)).
If you were not using `RField::(Read|Append)` directly, this change should not impact you.

- The new `RNTupleImporter` class provides automatic conversion of TTree to RNTuple.
Note that not all of the C++ types supported in TTree are currently supported in RNTuple.

- Many bug fixes and performance improvements

Please, report any issues regarding the above mentioned features should you encounter them.
RNTuple is still experimental and is scheduled to become production grade by end of 2024.
Thus, we appreciate feedback and suggestions for improvement.

## Histogram Libraries

2D Scatter plots are a very popular way to represent scientific data. Many scientific plotting
packages have this functionality. For many years ROOT itself as offered this kind of
visualization dedicated drawing options for TGraph or TTree. But there was no simple way
to produced 4D scatter plots from data stored in simple vectors. To fulfil these requirements
the new class, TScatter, has been implemented. It is able to draw a four variables scatter
plot on a single plot. A [detailed description](https://root.cern/blog/new-class-tscatter/)
was given on the website as a blog-post.

## Math Libraries

### Minuit2 is now the default minimizer

Many ROOT-based frameworks and users employ Minuit2 as the minimizer of choice for a long time already.
Therefore, Minuit2 is now the default minimizer used by ROOT.
This affects also **RooFit**, which inherits the default minimizer from ROOT Math.

The default can be changed back to the old Minuit implementation as follows:
```c++
ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit");
```

Alternatively, you can add this line to your `~/.rootrc` file:
```
Root.Fitter: Minuit
```

### Behavior change of `TMath::AreEqualAbs()`

The `TMath::AreEqualAbs()` compares two numbers for equality within a certain absolute range.
So far, it would tell you that `inf != inf` if you define `inf` as `std::numeric_limits<double>::infinity()`, which is inconsistent with the regular `==` operator.

This is unexpected, because one would expect that if two numbers are considered exactly equal, they would also be considered equal within any range.
Therefore, the behavior of `TMath::AreEqualAbs()` was changed to return always `true` if the `==` comparison would return `true`.

## RooFit Libraries

### Changes in RooFormulaVar and RooGenericPdf

The TFormula-based RooFit classes `RooFormulaVar` and `RooGenericPdf` change a bit their behavior to be more consistent:

1. No matter which variables you pass to the constructor, only the variables that the formula depends on are registered as value servers.
2. Similarly, the `dependents()` method of RooFormulaVar and RooGenericPdf will only return the list of actual value servers.

### Removal of the RooGenFunction and RooMultiGenFunction classes

The `RooGenFunction` was only a lightweight adaptor that exports a RooAbsReal as a `ROOT::Math::IGenFunction`.
The same can be easily achieved with the generic `ROOT::Math::Functor1D`, so in the spirit of not duplicating interfaces, the `RooGenFunction` is removed in this release.

Here is an example that shows how to replace it in the unlikely case you were using it:

```C++
RooArgSet normSet{x}; // normalization set

// Old way 1: create a RooGenFunction:
RooGenFunction func1{pdf, x, {}, normSet};

// Old way 2: use `RooAbsReal::iGenFunction()`:
std::unique_ptr<ROOT::Math::IGenFunction> func2{
    pdf.iGenFunction(x, normSet)
};

// How to do it now:
RooFunctor functor{pdf, x, {}, normSet};
ROOT::Math::Functor1D func3{functor};
// Functor1D takes by reference, so the RooFunctor also needs to stay alive.
```

For the same reason, the `RooMultiGenFunction` class that implements a multidimensional `ROOT::Math::IMultiGenFunction` is removed too.
It can easily be replaced by a `ROOT::Math::Functor`:

```C++
RooFunctor functor{pdf, observables, {}, normSet};
ROOT::Math::Functor func4{functor, static_cast<unsigned int>(functor.nObs())};
// Functor takes by reference, so the RooFunctor also needs to stay alive.
```

### Define infinity as `std::numeric_limits<double>::infinity()`

RooFit has its internal representation of infinity in `RooNumber::infinity()`, which was `1e30` before.

Now, it is defined as `std::numeric_limits<double>::infinity()`, to be consistent with the C++ standard library and other code.

This change also affects the `RooNumber::isInfinite()` function.

### Remove `add(row, weight, weightError)` from RooAbsData interface

It was not good to have this signature in RooAbsData, because the
implementations in the two derived classes RooDataHist and RooDataSet were
inconsistent.

The RooDataSet indeed took the weight error as the third argument, but
the RooDataHist version instead took the sum of weights squared, which
is equivalent to the squared weight error.

Therefore, the virtual `RooAbsData::add(row, weight, weightError)` function was removed.

### Removal of `RooMomentMorphND` class

The `RooMomentMorphND` and `RooMomentMorphFuncND` were almost exactly the same,
only that one inherited from `RooAbsPdf` and the other from `RooAbsReal`.

Thanks to the `RooWrapperPdf`, this code duplication in the RooFit implementation can now be avoided.
Instead of using the removed `RooMomentMorphND` (which is the pdf), you now need to use the `RooMomentMorphFuncND`,
change its behavior to exactly match the former `RooMomentMorphND`, and then wrap it into a pdf object:

```C++
RooMomentMorphFuncND func{<constructor args you previously passed to RooMomentMorphFunc>};

func.setPdfMode(); // change behavior to be exactly like the former RooMomentMorphND

// Pass the selfNormalized=true` flag to the wrapper because the
RooMomentMorphFuncND already normalizes itself in pdf mode.
RooWrapperPdf pdf{"pdf_name", "pdf_name", func, /*selfNormalized=*/true};
```

### Removal of several internal classes from the public RooFit interface

Several RooFit classes of which the headers are publicly exposed in the interface were only meant as implementation details of other RooFit classes.
Some of these classes are now removed from the public interface:

1. `RooGenProdProj`, which was an implementation detail of the `RooProdPdf`
2. `RooScaledFunc`, which was an implementation detail of the plotting in RooFit
   In the supposedly very rare case where you used this class in your own
   implementations, just multiply the underlying RooAbsReal function with the
   scale factor and create a RooRealBinding, e.g.:
   ```c++
   RooProduct scaledFunc{"scaled_func", "", func, scaleFactor};
   RooRealBinding scaleBind(scaledFunc, x) ;
   ```
   instead of:
   ```c++
   RooRealBinding binding(func, x) ;
   RooScaledFunc scaledBinding(binding, scaleFactor);
   ```
3. The `RooAbsRootFinder`, which was the base class of `RooBrentRootFinder`.
   The `RooAbsRootFinder` was only used as the base class of
   `RooBrentRootFinder`, which is an implementation detail of several
   RooFit/RooStats functions. However, polymorphism never not relevant for root
   finding, so the `RooAbsRootFinder` is removed. In the rare case where you
   might have used it, please ROOT's other functionalities: RooFit is not for
   root finding.
4. The `RooFormula` class, which was not meant as a user-facing class, but as a
   shared implementation detail of `RooFormulaVar` and `RooGenericPdf`.
5. The `RooIntegratorBinding`, which was an implementation detail of the
   `RooIntegrator2D` and `RooSegmentedIntegrator2D` classes.
6. The `RooRealAnalytic`, which was an implementation detail of the
   `RooRealIntegral` class.

### Consistent default for `Extended()` command in RooAbsPdf::fitTo() and RooAbsPdf::chi2FitTo()

If no `RooFit::Extended()` command argument is passed, `RooAbsPdf::chi2FitTo()`
method now does an extended fit by default if the pdf is extendible. This makes
the behavior consistent with `RooAbsPdf::fitTo()`. Same applies to
`RooAbsPdf::createChi2()`.

## 2D Graphics Libraries

- Introduce `TAxis::ChangeLabelByValue` to set custom label defined by axis value. It works also
  when axis zooming changes and position and index of correspondent axis label changes as well.
  `TAxis::ChangeLabel` method to change axis label by index works as before.

- Introduce `TCanvas::SaveAll` method. Allows to store several pads at once into different image file formats.
  File name can include printf qualifier to code pad number. Also allows to store all pads in single PDF
  or single ROOT file. Significantly improves performance when creating many image files using web graphics.

- Introduce `TCanvas::UpdateAsync` method. In case of web-based canvas triggers update of the canvas on the client side,
  but does not wait that real update is completed. Avoids blocking of caller thread.
  Have to be used if called from other web-based widget to avoid logical dead-locks.
  In case of normal canvas just canvas->Update() is performed.


## Language Bindings

- Add support for Python 3.12.
- Speedup inclusion of ROOT module.
- Improve numba support for PyROOT, see `tutorials/pyroot/pyroot004_NumbaDeclare.py`.

## JavaScript ROOT

- Many improvements to ROOT's' JavaScript plotting / graphics facilities.

## Tutorials

- `dataframe/df106_HiggsToFourLeptons.C` now demonstrates `Vary()`, `RDatasetSpec` (`FromSpec()`), and `AddProgressBar()`.
- New tutorials for `RBatchGenerator`: `tutorials/tmva/RBatchGenerator_PyTorch.py`, `tutorials/tmva/RBatchGenerator_NumPy.py`, `tutorials/tmva/RBatchGenerator_TensorFlow.py`, `tutorials/tmva/RBatchGenerator_filters_vectors.py`.
- Showcase new `TScatter` class in `tutorials/graphs/scatter.C`.
- Demonstrate the use of `SOFIE` for fast inference in `tutorials/tmva/TMVA_SOFIE_RSofieReader.C`.


## Class Reference Guide

- Doxygen 1.9.8 is now supported, including its support for dark mode.

## Build, Configuration and Testing Infrastructure

If `-Droottest=ON` is specified, the ROOT build system used to clone a matching branch of the `roottest` repository.
This logic has been improved and is now as follows:

 - If the current head is a well-known branch, e.g. `master` or `v6-28-00-patches`, use the matching branch upstream;
 - otherwise, try a branch that matches the name of the current head in the forked repository, if it exists; else try using the closest upstream head/tag below `HEAD`'s parent commit;
 - as a last resort, if there is no preferred candidate, checkout the remote's default head.


## Bugs and Issues fixed in this release

* [[#11987](https://github.com/root-project/root/issues/11987)] - New importer tutorial fails to run a second time
* [[#11947](https://github.com/root-project/root/issues/11947)] - [hist] `TH1::GetCumulative` doesn't set the bin error for the generated histogram, but change that of the original hist instead
* [[#12031](https://github.com/root-project/root/issues/12031)] - [ntuple] `RNTupleReader::Show()` produces empty output if user does not impose a model
* [[#11907](https://github.com/root-project/root/issues/11907)] - thread local `gDirectory` not properly updated when another delete the file its point to.
* [[#12020](https://github.com/root-project/root/issues/12020)] - [RF] Cannot generate nested RooSimultaneous from prototype category data
* [[#11930](https://github.com/root-project/root/issues/11930)] - Failure in `TClass::GetMethodWithPrototype`
* [[#12130](https://github.com/root-project/root/issues/12130)] - ROOT Head fails to compile on macOS(11, 12) since Jan 27, 2023
* [[#12184](https://github.com/root-project/root/issues/12184)] - CI: group build logs
* [[#11487](https://github.com/root-project/root/issues/11487)] - [RF] Weights are lost when importing two weighted `RooDataSets` into one
* [[#12286](https://github.com/root-project/root/issues/12286)] - [RF] Generation of `RooKeysPdf` fails if using protodata
* [[#11483](https://github.com/root-project/root/issues/11483)] - Cannot use globs with `?#` syntax in `TChain::Add`
* [[#12148](https://github.com/root-project/root/issues/12148)] - Failing tests with numpy 1.24
* [[#12270](https://github.com/root-project/root/issues/12270)] - CI: run roottest as part of ROOT CI
* [[#11938](https://github.com/root-project/root/issues/11938)] - Implement RX option in case of Log Scale.
* [[#10062](https://github.com/root-project/root/issues/10062)] - `RootMacros.cmake` on macOS 12.2.1, root version v6-26
* [[#11271](https://github.com/root-project/root/issues/11271)] - ROOT fails to compile with current VecGeom
* [[#12293](https://github.com/root-project/root/issues/12293)] - ROOT 6.28.00 fails on s390x: Added modules have incompatible data layouts
* [[#12338](https://github.com/root-project/root/issues/12338)] - [math] `ROOT::Math::Minimizer` ignores error level if set after the function in the case of analytical gradient
* [[#12260](https://github.com/root-project/root/issues/12260)] - [DF] Bogus data read from indexed friend trees in multi-thread runs
* [[#12294](https://github.com/root-project/root/issues/12294)] - ROOT 6.28.00 fails on aarch64: cling JIT session error: Failed to materialize symbols
* [[#11671](https://github.com/root-project/root/issues/11671)] - [ntuple] Pending optimizations for storage of classes with an associated collection proxy
* [[#12391](https://github.com/root-project/root/issues/12391)] - [PyROOT] TypeError: no python-side overrides supported (failed to compile the dispatcher code)
* [[#12398](https://github.com/root-project/root/issues/12398)] - [VecOps] Masking `RVec<T>` is broken for non-trivially-constructible `T`s
* [[#12231](https://github.com/root-project/root/issues/12231)] - Hang with XRootD from eospublic on Debian Unstable
* [[#12430](https://github.com/root-project/root/issues/12430)] - [RF] `RooProdPdf` test failure on s390x
* [[#12438](https://github.com/root-project/root/issues/12438)] - Unstable result of lz4 compression
* [[#12186](https://github.com/root-project/root/issues/12186)] - CI: test summary
* [[#12302](https://github.com/root-project/root/issues/12302)] - [ci] Make it possible to force a CI build from scratch
* [[#12297](https://github.com/root-project/root/issues/12297)] - CI: config specification
* [[#12498](https://github.com/root-project/root/issues/12498)] - `TTime` class is not splitable
* [[#6911](https://github.com/root-project/root/issues/6911)] - Test depending on histfactory are run even if histfactory is not build
* [[#12185](https://github.com/root-project/root/issues/12185)] - CI: log parsing
* [[#12520](https://github.com/root-project/root/issues/12520)] - `RDF::FromCSV` gives wrong output with colTypes specified
* [[#10660](https://github.com/root-project/root/issues/10660)] - `TH2::SetShowProjectionXY`
* [[#12267](https://github.com/root-project/root/issues/12267)] - CI: macOS
* [[#12453](https://github.com/root-project/root/issues/12453)] - [RF] Weight errors are wrong when splitting `RooDataSet` and `RooDataHist` with weight errors
* [[#12579](https://github.com/root-project/root/issues/12579)] - [backport] support `__cast_cpp__` to implicitly convert python objects to proxies
* [[#12597](https://github.com/root-project/root/issues/12597)] - [DF] Display of `RVec<bool>` shows wrong values
* [[#12427](https://github.com/root-project/root/issues/12427)] - Test failures in `RNTuple`: 32 bit specific
* [[#12263](https://github.com/root-project/root/issues/12263)] - `root_generate_dictionary` does not properly handle "all" include directory properties from its dependencies.
* [[#12358](https://github.com/root-project/root/issues/12358)] - Self-reference through a map of tuples breaks the interpreter (2023.02.20.)
* [[#12623](https://github.com/root-project/root/issues/12623)] - `gDirectory` macro should include global-namespace qualification
* [[#12329](https://github.com/root-project/root/issues/12329)] - [RF] Writing copy of `RooWorkspace` messes up proxies
* [[#11876](https://github.com/root-project/root/issues/11876)] - [RF] Problem writing `RooMomentMorphND` to workspace in RooFit
* [[#12378](https://github.com/root-project/root/issues/12378)] - `GetClass(<typedef>)` works only at the second try
* [[#12661](https://github.com/root-project/root/issues/12661)] - `TTreeReader` does not support enum type as parameter
* [[#12652](https://github.com/root-project/root/issues/12652)] - [RF] Plots of nested `RooSimultaneous` ignore `ProjWData`
* [[#12570](https://github.com/root-project/root/issues/12570)] - Remove GIF output from stressGraphics
* [[#11562](https://github.com/root-project/root/issues/11562)] - Interpreter warns when calling `[[nodiscard]]` functions
* [[#12552](https://github.com/root-project/root/issues/12552)] - Race condition when loading dictionary shared libraries
* [[#12704](https://github.com/root-project/root/issues/12704)] - [RF] Off-by-one error in `Roofit/Histfactory/HistoToWorkspaceFactoryFast`
* [[#12646](https://github.com/root-project/root/issues/12646)] - [RF] `RooLinearVar` not used in plot projection integrals
* [[#12679](https://github.com/root-project/root/issues/12679)] - Draw two `TF1`
* [[#12742](https://github.com/root-project/root/issues/12742)] - [ntuple] Assertion `fWritePage[fWritePageIdx].IsEmpty()` violated at line 87 of `RColumn.hxx`
* [[#10895](https://github.com/root-project/root/issues/10895)] - [cling] void macro should not return value
* [[#12766](https://github.com/root-project/root/issues/12766)] - OpenSSL 3.1 not supported in build of module net/http
* [[#12307](https://github.com/root-project/root/issues/12307)] - [CI] Build is green even if tests are failing
* [[#12533](https://github.com/root-project/root/issues/12533)] - Backport missing commits to our fork of the LLVM monorepo
* [[#11977](https://github.com/root-project/root/issues/11977)] - Nonzero exit status of `root --help`
* [[#12457](https://github.com/root-project/root/issues/12457)] - Failing Cling test for unloading Lambda in template instantiation
* [[#12783](https://github.com/root-project/root/issues/12783)] - [IO] Writing `HistFactory` model file twice gives strange results since ROOT 6.26.02
* [[#12649](https://github.com/root-project/root/issues/12649)] - `TTreeCache` scale (very) poorly with number of baskets/clusters.
* [[#12567](https://github.com/root-project/root/issues/12567)] - Wrong y zoom with empty hbar histogram
* [[#12715](https://github.com/root-project/root/issues/12715)] - Issue with `TClass` object managed in case of multi-threaded 'rapid' set of dlopen/dlclose on same library.
* [[#12496](https://github.com/root-project/root/issues/12496)] - Graphics crash when using `EnableImplicitMT()` with a python for loop
* [[#12455](https://github.com/root-project/root/issues/12455)] - Failing Cling tests with multiple interpreters
* [[#12870](https://github.com/root-project/root/issues/12870)] - RDF `Graph()` title: x vs y
* [[#12686](https://github.com/root-project/root/issues/12686)] - Some warnings in `argparse2help` with Python 3.12
* [[#12967](https://github.com/root-project/root/issues/12967)] - [RF] `HistFactory` workspaces incompatible between versions 6.26 and 6.28
* [[#12922](https://github.com/root-project/root/issues/12922)] - Add flag option "--web=off" for rootbrowse macro
* [[#10020](https://github.com/root-project/root/issues/10020)] - [ntuple] Support backfilling
* [[#13005](https://github.com/root-project/root/issues/13005)] - Ambiguous default parameter in `TF2::Integral` documentation
* [[#11294](https://github.com/root-project/root/issues/11294)] - [PyROOT] Multiple issues with `Numba.Declare`
* [[#12535](https://github.com/root-project/root/issues/12535)] - `TGraph2D`'s `Interpolate()` returns 0
* [[#12003](https://github.com/root-project/root/issues/12003)] - "different definitions in different modules" with C++17 build on macOS
* [[#10608](https://github.com/root-project/root/issues/10608)] - Scatter plot combining Color Size and Alpha
* [[#10291](https://github.com/root-project/root/issues/10291)] - Problems with std::map in a Debug build using recent versions of GCC
* [[#12787](https://github.com/root-project/root/issues/12787)] - compilation fails on FreeBSD 13.2 RELEASE
* [[#13080](https://github.com/root-project/root/issues/13080)] - [FreeBSD] macro compilation fail (evolution test)
* [[#13068](https://github.com/root-project/root/issues/13068)] - [FreeBSD] `TThread` related tests fail
* [[#9805](https://github.com/root-project/root/issues/9805)] - Add env variable to limit number of threads used by ROOT
* [[#13128](https://github.com/root-project/root/issues/13128)] - [FreeBSD] merger tests fail with floating point exception
* [[#13157](https://github.com/root-project/root/issues/13157)] - documentation gives misleading examples
* [[#13168](https://github.com/root-project/root/issues/13168)] - `TTree::fChainOffset` is never set
* [[#10990](https://github.com/root-project/root/issues/10990)] - libAfterImage: `TEve` broken with giflib 5
* [[#13215](https://github.com/root-project/root/issues/13215)] - Properly support macOS in `rootssh`
* [[#12868](https://github.com/root-project/root/issues/12868)] - When unloading a library the altername class name are not unloaded.
* [[#13017](https://github.com/root-project/root/issues/13017)] - TMVARegression crashes for nevents < 100
* [[#13037](https://github.com/root-project/root/issues/13037)] - [DF] `FromSpec`: parsing of the JSON file is done according to the alphabetical ordering of the main keys
* [[#13177](https://github.com/root-project/root/issues/13177)] - Typo in `tmva/tmva/src/DataLoader.cxx`
* [[#12958](https://github.com/root-project/root/issues/12958)] - [ntuple] `UnsealPage()` should be a no-op for the page zero
* [[#12832](https://github.com/root-project/root/issues/12832)] - [RF] Change precision of `RooAbsL` EventSections tests
* [[#13174](https://github.com/root-project/root/issues/13174)] - [FreeBSD] hardcoded `/bin/bash` does not work on all platforms
* [[#13182](https://github.com/root-project/root/issues/13182)] - Infinite loop in `TDirectoryFile::ls`
* [[#8650](https://github.com/root-project/root/issues/8650)] - `TFile::ls` hangs in infinite loop when slash in key
* [[#9449](https://github.com/root-project/root/issues/9449)] - [cling] `error: function definition is not allowed here` when trying to overload `operator<=`
* [[#13233](https://github.com/root-project/root/issues/13233)] - Classic `TRootBrowser: Histogramming` leafs which are functions fails
* [[#13276](https://github.com/root-project/root/issues/13276)] - Primer wiki misspelling
* [[#13088](https://github.com/root-project/root/issues/13088)] - Retain tree name through `TChain` with `RNTupleImporter`
* [[#13325](https://github.com/root-project/root/issues/13325)] - Debug assertion failure "cannot seek vector iterator after end" in RTensor.hxx
* [[#13323](https://github.com/root-project/root/issues/13323)] - Script to update PDG table
* [[#13058](https://github.com/root-project/root/issues/13058)] - Test failures with C++17, but without `runtime_cxxmodules`
* [[#13249](https://github.com/root-project/root/issues/13249)] - Builds of the new CI do not actually run latest changes in (some) tests
* [[#11920](https://github.com/root-project/root/issues/11920)] - [CMake] Attempt to write `ClingConfig.cmake.tmp` to the external LLVM library directory, causing configuration failure
* [[#12152](https://github.com/root-project/root/issues/12152)] - Building fails with `builtin_llvm=OFF` due to unintend libbsd linking
* [[#10326](https://github.com/root-project/root/issues/10326)] - [ntuple] Add I/O support for `std::set`
* [[#12156](https://github.com/root-project/root/issues/12156)] - Building fails with `builtin_llvm=OFF`:  CommandLine Error: Option 'W' registered more than once!
* [[#13136](https://github.com/root-project/root/issues/13136)] - Excessive `Form` usage in TMVA
* [[#13156](https://github.com/root-project/root/issues/13156)] - ROOT header `core/clib/inc/strlcpy.h` incompatible with latest glibc
* [[#12960](https://github.com/root-project/root/issues/12960)] - [ntuple] `RPageSourceDaos` should be able to populate zero pages from `kTypePageZero` locators
* [[#13462](https://github.com/root-project/root/issues/13462)] - Segmentation violation for a trivial `std::unique_ptr`
* [[#13404](https://github.com/root-project/root/issues/13404)] - [RF] Documentation issue for `RooAbsReal::getPropagatedError`
* [[#12828](https://github.com/root-project/root/issues/12828)] - `ROOT::TestSupport` library not working as intended
* [[#13449](https://github.com/root-project/root/issues/13449)] - Slow closing of `TFile` with very large number of directories.
* [[#13450](https://github.com/root-project/root/issues/13450)] - `Error while building module 'std' imported from input_line_1:1:...`
* [[#13516](https://github.com/root-project/root/issues/13516)] - Compilation of `dataframe/test/datasource_arrow.cxx`  fails on standard linux distribution
* [[#7978](https://github.com/root-project/root/issues/7978)] - Dependencies using `${GENERATE_REFLEX_TEST}` variable do not work in roottest
* [[#13551](https://github.com/root-project/root/issues/13551)] - [df] Creation of Snapshot actions writes to uninitialized memory
* [[#13445](https://github.com/root-project/root/issues/13445)] - hadd help and manual describe inconsistently the "-f" option
* [[#13529](https://github.com/root-project/root/issues/13529)] - Compilation error on arch using gcc 13.2.1: strlcopy has a different exception specifier
* [[#13503](https://github.com/root-project/root/issues/13503)] - Double call to `TNetXNGFile::Close()` leads to a segfault
* [[#13431](https://github.com/root-project/root/issues/13431)] - Add more information to `TInterpreter::ReadRootmapFile` warnings
* [[#13560](https://github.com/root-project/root/issues/13560)] - Printing pad to pdf is missing endobj in some cases
* [[#9058](https://github.com/root-project/root/issues/9058)] - [TPython] Fix `TPython::ExecScript` for Python3
* [[#13569](https://github.com/root-project/root/issues/13569)] - [RHist] variable `bins_per_hyperplane` is being used without being initialized.
* [[#8051](https://github.com/root-project/root/issues/8051)] - Impossible to understand what to `#include` for `ROOT::Math::PtEtaPhiMVector` etc
* [[#11580](https://github.com/root-project/root/issues/11580)] - `TDatabasePDG::GetParticle(...)` not being thread safe
* [[#13424](https://github.com/root-project/root/issues/13424)] - Artefact when drawing `TGraph2D` with "tri1" draw option
* [[#13597](https://github.com/root-project/root/issues/13597)] - CMake fails with `LLVM_ENABLE_SPHINX=ON`
* [[#13600](https://github.com/root-project/root/issues/13600)] - `TGraph(Asymm,Bent)Errors` getters segfault in PyROOT
* [[#13605](https://github.com/root-project/root/issues/13605)] - [RF] `RooDataSet.from_numpy` gives wrong result when the input arrays are not c-contiguous
* [[#10484](https://github.com/root-project/root/issues/10484)] - `Graph` and `GraphAsymmErrors` Action Helpers not deallocating memory when event loop is interrupted
* [[#13513](https://github.com/root-project/root/issues/13513)] - A typo in source code
* [[#13631](https://github.com/root-project/root/issues/13631)] - `LiveVisualize` does not update canvas on Jupyter notebook
* [[#11965](https://github.com/root-project/root/issues/11965)] - [RF] Completely implement `Offset("bin")` feature
* [[#13698](https://github.com/root-project/root/issues/13698)] - [Meta] ROOT's llvm has issues with ranges coming from the Macos 14.3 sdk (llvm 15)
* [[#13693](https://github.com/root-project/root/issues/13693)] - Different PDF output between `RCanvas` and `TCanvas`
* [[#13632](https://github.com/root-project/root/issues/13632)] - Segmentation fault in `TGraph::Sort` with large number of entries
* [[#13672](https://github.com/root-project/root/issues/13672)] - Pyroot's `TProfile3D` cannot display the model under `'% jsroot on'`.
* [[#13678](https://github.com/root-project/root/issues/13678)] - Different results using `SetFillStyle` using `TCanvas` and WebCanvas
* [[#9847](https://github.com/root-project/root/issues/9847)] - Changing type `std::map<const std::string>` to `std::map<std::string>` can cause unexpected cling-related crashes in Python on macOS
* [[#6775](https://github.com/root-project/root/issues/6775)] - [bug] [minuit] FPE/crash in Minuit2
* [[#13719](https://github.com/root-project/root/issues/13719)] - Different behaviour of new and old graphics
* [[#13707](https://github.com/root-project/root/issues/13707)] - Different PDF output `TCanvas` / `TWebCanvas`
* [[#13730](https://github.com/root-project/root/issues/13730)] - [math] `TComplex` arithmetic operators don't work with integral types
* [[#7063](https://github.com/root-project/root/issues/7063)] - TBB moved to CMake-only in version 2021 and builtin TBB needs to be adapted for the future
* [[#11179](https://github.com/root-project/root/issues/11179)] - TBB will be broken with newer versions of Clang
* [[#10984](https://github.com/root-project/root/issues/10984)] - Compilation fails with VecGeom
* [[#13690](https://github.com/root-project/root/issues/13690)] - [DF] `RDataFrame`'s `FromNumpy` silently loads wrong values if input arrays have a stride
* [[#13691](https://github.com/root-project/root/issues/13691)] - `TDirectoryFile` destructor segfault in compiled C++ program
* [[#13429](https://github.com/root-project/root/issues/13429)] - `TROOT::EndOfProcessCleanups` fails when using `TCMalloc` on different destructors
* [[#12043](https://github.com/root-project/root/issues/12043)] - Wrong interaction of `DefinePerSample` with multiple executions
* [[#11797](https://github.com/root-project/root/issues/11797)] - Incorrect Dependency on VDT
* [[#13798](https://github.com/root-project/root/issues/13798)] - TMVA cannot be initialized properly in distributed Python environments
* [[#13079](https://github.com/root-project/root/issues/13079)] - Builtin TBB library sometimes not found (or more exactly the one install in `/usr/lib` sometimes take priority)
* [[#13410](https://github.com/root-project/root/issues/13410)] - cppyy crash involving template overload of `operator()`
* [[#13734](https://github.com/root-project/root/issues/13734)] - [PyROOT] Can't trivially `gSystem.Load()` libraries compiled with ACLiC on macOS 14
* [[#13851](https://github.com/root-project/root/issues/13851)] - Test crash with GCC 13 and C++20
* [[#12817](https://github.com/root-project/root/issues/12817)] - Performance regression with repr for pyroot objects in root 6.28 nightlies.
* [[#7187](https://github.com/root-project/root/issues/7187)] - TInterpreter::ToString hogs memory
* [[#13906](https://github.com/root-project/root/issues/13906)] - Tutorial `fit/fit2dHist.C` crashed when running with global combined fit
* [[#9114](https://github.com/root-project/root/issues/9114)] - Deprecate then remove exceptions config flag
* [[#13848](https://github.com/root-project/root/issues/13848)] - Incorrect initialization of `TMatrixTSparse`
* [[#13543](https://github.com/root-project/root/issues/13543)] - `rootcling --genreflex` ignores `<field... transient="true"/>` in selection XML
* [[#13964](https://github.com/root-project/root/issues/13964)] - Interactive PNG output is incorrect with MacOSX 14.1 and Clang 15
* [[#13927](https://github.com/root-project/root/issues/13927)] - `[TF1,TF2,TF3]::Save` works incorrectly when called with default arguments (all 0)
* [[ROOT-2869](https://sft.its.cern.ch/jira/browse/ROOT-2869)] - `TTree:Draw`/`fEstimate` is erratic and poorly documented
* [[ROOT-4188](href='https://sft.its.cern.ch/jira/browse/ROOT-4188)] - `RooGamma` random number sampling
* [[ROOT-7922](href='https://sft.its.cern.ch/jira/browse/ROOT-7922)] - `RooStats::ModelConfig::Set*(const char*)`

## HEAD of the v6-30-00-patches branch

These changes will be part of a future 6.30/02.

- None so far.

