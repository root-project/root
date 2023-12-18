% ROOT Version 6.30 Release Notes
% 2022-12-21
<a name="TopOfPage"></a>

## Introduction

ROOT version 6.30/00 is scheduled for release in October, 2023.

For more information, see:

[http://root.cern](http://root.cern)

The following people have contributed to this new version:

 Daniel Álvarez Conde, CERN/EP-SFT,\
 Guilherme Amadio, CERN/IT,\
 Bertrand Bellenot, CERN/EP-SFT,\
 Jakob Blomer, CERN/EP-SFT,\
 Patrick Bos, Netherlands eScience Center,\
 Rene Brun,\
 Carsten Burgard, TU Dortmund,\
 Will Buttinger, Rutherford Appleton Lab,\
 Philippe Canal, FNAL,\
 Olivier Couet, CERN/EP-SFT,\
 Marta Czurylo, CERN/EP-SFT,\
 Mattias Ellert, Uppsala Uni,\
 Edward Finkelstein, JGU Mainz,\
 Gerri Ganis, CERN/EP-SFT,\
 Paul Gessinger, CERN/EP-SFT,\
 Florine de Geus, CERN/ATLAS,\
 Andrei Gheata, CERN/EP-SFT,\
 Enrico Guiraud, CERN/EP-SFT and Princeton,\
 Ahmat Hamdan, CERN/EP-SFT,\
 Stephan Hageboeck, CERN/IT,\
 Jonas Hahnfeld, CERN/EP-SFT,\
 Fernando Hueso González, CSIC/UV,\
 Attila Krasznahorkay, CERN/ATLAS,\
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
 Dante Niewenhuis, CERN/EP-SFT,\
 Vincenzo Eduardo Padulano, CERN/EP-SFT,\
 Ioanna Maria Panagou, CERN/EP-SFT,\
 Danilo Piparo, CERN/EP-SFT,\
 Fons Rademakers, CERN/IT,\
 Jonas Rembser, CERN/EP-SFT,\
 Jakob Schneekloth, CERN/EP-SFT,\
 Sanjiban Sengupta, CERN/EP-SFT,\
 Neel Shah, GSoC,\
 Garima Singh, CERN/EP-SFT and Princeton,\
 Yash Solanki, GSoC,\
 Uri Stern, CERN/EP-SFT,\
 Enric Tejedor Saavedra, CERN/IT,\
 Matevz Tadel, UCSD/CMS,\
 [QuillPusher](https://github.com/QuillPusher), [Compiler Research Group](https://compiler-research.org/team/),\
 Vassil Vassilev, Princeton/CMS,\
 Wouter Verkerke, NIKHEF/ATLAS,\
 Daniel Werner, CERN/EP-SFT,\
 Zef Wolffs, NIKHEF/ATLAS

## Deprecation and Removal
- The minimum C++ standard supported by ROOT is now C++17.
- Support for Python 2 is now deprecated and it will be removed in next release 6.32.
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

## I/O Libraries

- Improve parsing of input argument to `TChain::Add`. Now it supports the case of globbing files while also using the `?#` token to specify the tree name.

## RDataFrame
- instead of returning nothing, `ROOT::RDF::RunGraphs` now returns the number of separate computation graphs that have been run.

- Introduce [`ProgressBar`](https://root.cern/doc/master/classROOT_1_1RDataFrame.html#progressbar) feature that can be added to any RDataFrame program.

- The `RDatasetSpec` class and its users now employ the concept of 'sample' rather than the original naming 'group' for groups of files with associated metadata.

- `df106_HiggsToFourLeptons` tutorials (both python and C++) now showcase the `ProgressBar`. They now use `FromSpec` to define multiple samples and `Vary` for systematic variations.

### Distributed RDataFrame

- Vastly improve runtime performance when using an RDataFrame with simulated dataset, i.e. `RDataFrame(nentries)`, by removing usage of `Range` operation to define the per-task entry range.

- Explicitly error out when trying to process a TTree with a TTreeIndex in distributed mode. The feature is currently not supported.

- JITting the RDataFrame computation graph now only happens once per worker process, not once per task. This greatly reduces memory usage and runtime overhead at of each task.

## TTree Libraries

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
```
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
```
RNTupleWriteOptions options;
options.SetHasSmallClusters(true);
auto writer = RNTupleWriter::Recreate(std::move(model), "myNTuple", "out.ntuple");
```

- Support for projected fields, i.e. exposing other fields' data as a different (compatible) C++ type.
Users should provide a mapping function that maps each projected subfield in the tree to theunderlying real field, e.g.
```
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

Please, report any issues regarding the abovementioned features should you encounter them.
RNTuple is still experimental and is scheduled to become production grade by end of 2024.
Thus, we appreciate feedback and suggestions for improvement.

## Histogram Libraries


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
change its behavior to exactly match the formter `RooMomentMorphND`, and then wrap it into a pdf object:

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

## TMVA
### SOFIE : Code generation for fast inference of Deep Learning models
TMVA SOFIE now supports parsing and further inference of Graph Neural Networks based on DeepMind's [graph_nets](https://github.com/google-deepmind/graph_nets). The list of all operators supported in the `RModel` class is the one provided below for the ONNX parser.

#### SOFIE-GNN
1. The SOFIE-GNN implementation brought a major change in SOFIE's architecture. Instead of having only the RModel class to store model information, now SOFIE has RModel, RModel_GNN and RModel_GraphIndependent classes which are inherited from RModel_Base.
2. **RModel_GNN** is used to store a GNN model having nodes, edges, and globals with functions for their update and aggregate(for inter-relationships).
3. **RModel_GraphIndependent** is used to store an independent Graph model with nodes, edges and globals with their individual update functions.
4. **RFunctions** are used to declare update/aggregate operations over graph components. Currently supported RFunctions include:
    - **Update Functions**
        - RFunction_MLP
    - **Aggregate Functions**
        - RFunction_Mean
        - RFunction_Sum
5. Pythonized functions for parsing a Graphnets' model can be used to generate inference code
```
   import graph_nets as gn
   from graph_nets import utils_tf

   GraphModule = gn.modules.GraphNetwork(
      edge_model_fn=lambda: snt.nets.MLP([2,2], activate_final=True),
      node_model_fn=lambda: snt.nets.MLP([2,2], activate_final=True),
      global_model_fn=lambda: snt.nets.MLP([2,2], activate_final=True))

   GraphData = get_graph_data_dict(2,1,2,2,2)

   model = ROOT.TMVA.Experimental.SOFIE.RModel_GNN.ParseFromMemory(GraphModule, GraphData)
   model.Generate()
   model.OutputGenerated()

```
A complete tutorial for the SOFIE-GNN implementation can be found [here](https://github.com/root-project/root/blob/master/tutorials/tmva/TMVA_SOFIE_GNN.py)

#### SOFIE ONNX Parser

The ONNX parser supports now several new ONNX operators. The list of the current supported ONNX operator is the following:
- Gemm
- Conv (in 1D,2D and 3D)
- RNN, GRU, LSTM
- Relu, Selu, Sigmoid, Softmax, Tanh, LeakyRelu
- BatchNormalization
- MaxPool, AveragePool, GlobalAverage
- ConvTranspose
- Gather
- Expand, Reduce
- Neg, Exp, Sqrt, Reciprocal
- Add, Sum, Mul, Div
- Reshape, Flatten, Transpose
- Squeeze, Unsqueeze, Slice
- Concat, Reduce
- Identity
- Shape
- Custom
- Error
- Log

#### SOFIE Keras Parser
- The Swish Activation function is now supported in the SOFIE Keras parser.

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

- The Delaunay triangles (used by TGraph2D) were computed by the external package `triangle.c`
  included in the ROOT distribution. This package had several issues:
     - It was not maintained anymore.
     - Its license was not compatible with LGPL
  This code is now replaced by the [CDT package](https://github.com/artem-ogre/CDT) which is
  properly maintained and has a license (MLP) compatible with LGPL. It will appear in 6.03.02.


## Machine Learning integration

- ROOT now offers functionality to extract batches of events out of a dataset for use in common ML training workflows. For example, one can generate PyTorch tensors from a TTree. The functionality is available through the `RBatchGenerator` class and can be seamlessly integrated in user code, for example:
   ```python
   # Returns two generators that return training and validation batches as PyTorch tensors.
   gen_train, gen_validation = ROOT.TMVA.Experimental.CreatePyTorchGenerators(
      tree_name, file_name, batch_size, chunk_size, target=target, validation_split=0.3)
   ```
   The functionality is also available for TensorFlow datasets and Python generators of numpy arrays. See more in the `RBatchGenerator*` tutorials under the TMVA folder.

## 3D Graphics Libraries


## Geometry Libraries


## Database Libraries


## Networking Libraries


## GUI Libraries


## Montecarlo Libraries


## PROOF Libraries


## Language Bindings


## JavaScript ROOT

- The default `TCanvas` and `TBrowser` is switched back to the traditional look. (`--web=off` is no longer needed)

## Tutorials


## Class Reference Guide


## Build, Configuration and Testing Infrastructure

- The traditional versioning convention of ROOT (e.g. 6.28/10) has been changed to standard semantic versioning (6.28.10), i.e. the slash is changed by a point. Please update any user script that relied on parsing the slash.

- `mathmore` (and thus other features depending on it) is no longer enabled by default as it's not LGPL-compliant.
  
- System-wide `afterimage` and `nlohmann_json` packages are preferred over the `builtin` options in the binary releases.

- If `-Droottest=ON` is specified, the ROOT build system used to clone a matching branch of the `roottest` repository.
This logic has been improved and is now as follows:
_(i)_ If the current head is a well-known branch, e.g. `master` or `v6-28-00-patches`, use the matching branch upstream;
_(ii)_ otherwise, try a branch that matches the name of the current head in the forked repository, if it exists; else try using the closest upstream head/tag below `HEAD`'s parent commit;
_(iii)_ as a last resort, if there is no preferred candidate, checkout the remote's default head.
