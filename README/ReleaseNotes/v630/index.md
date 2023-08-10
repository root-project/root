% ROOT Version 6.30 Release Notes
% 2022-12-21
<a name="TopOfPage"></a>

## Introduction

ROOT version 6.30/00 is scheduled for release in May, 2023.

For more information, see:

[http://root.cern](http://root.cern)

The following people have contributed to this new version:

 Bertrand Bellenot, CERN/SFT,\
 Jakob Blomer, CERN/SFT,\
 Rene Brun, CERN/SFT,\
 Philippe Canal, FNAL,\
 Olivier Couet, CERN/SFT,\
 Gerri Ganis, CERN/SFT,\
 Andrei Gheata, CERN/SFT,\
 Enrico Guiraud, CERN/SFT,\
 Sergey Linev, GSI,\
 Javier Lopez-Gomez, CERN/SFT,\
 Pere Mato, CERN/SFT,\
 Lorenzo Moneta, CERN/SFT,\
 Axel Naumann, CERN/SFT,\
 Danilo Piparo, CERN/SFT,\
 Fons Rademakers, CERN/SFT,\
 Jonas Rembser, CERN/SFT,\
 Enric Tejedor Saavedra, CERN/SFT,\
 Matevz Tadel, UCSD/CMS,\
 Vassil Vassilev, Princeton/CMS,\
 Wouter Verkerke, NIKHEF/Atlas,

## Deprecation and Removal
- `ROOT::RDF::RResultHandle::GetResultPtr` has been deprecated. Please use `RResultPtr` directly instead and only cast to `RResultHandle` in order to call `ROOT::RDF::RunGraphs`.
- The RDataFrame factory functions `MakeCsvDataFrame`, `MakeArrowDataFrame`, `MakeNTupleDataFrame` and `MakeSqliteDataFrame` that were deprecated in v6.28 have been removed. Use `FromCSV`, `FromArrow`, `FromRNTuple` or `FromSqlite` instead.
- The TStorage reallocation routine without a size (`TStorage::ReAlloc(void *ovp, size_t size`) and heap related routines (`TStorage::AddToHeap`, `TStorage::IsOnHeap`, `TStorage::GetHeapBegin`, `TStorage::GetHeapEnd`) that were deprecated in v6.02/00 have been removed.
- The deprecated `Format(const char* option, int sigDigits)` option for `RooAbsPdf::paramOn()` was removed. Please use the `Format(const char* option, ...)` overload that takes command arguments.
- The deprecated `RooAbsPdf::paramOn()` overload that directly takes a formatting string was removed. Please take the overload that uses command arguments.
- The `RooCatType` class was deprecated in ROOT 6.22 and its original `RooCatType.h` header is now removed. If you still need access to this class, please include `RooFitLegacy/RooCatTypeLegacy.h`.
- The `RooAbsString` that was only an alias for `RooStringVar` got removed.
- The `RooDataWeightedAverage` is now deprecated and will be removed in 6.32. It was only supposed to be an implementation detail of RooFits plotting that is now not necessary anymore.

## Core Libraries


## I/O Libraries


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

### Behavior change of `TMath::AreEqualAbs()`

The `TMath::AreEqualAbs()` compares two numbers for equality within a certain absolute range.
So far, it would tell you that `inf != inf` if you define `inf` as `std::numeric_limits<double>::infinity()`, which is inconsistent with the regular `==` operator.

This is unexpected, because one would expect that if two numbers are considered exactly equal, they would also be considered equal within any range.
Therefore, the behavior of `TMath::AreEqualAbs()` was changed to return always `true` if the `==` comparision would return `true`.

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

### Removal of serveral internal classes from the public RooFit interface

Several RooFit classes of which the headers are publically exposed in the interface were only meant as implementation details of other RooFit classes.
Some of these classes are now removed from the public interface:

1. `RooGenProdProj`, which was an implementation detail of the `RooProdPdf`
2. `RooScaledFunc`, which was an implementaiton detail of the plotting in RooFit
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

## 2D Graphics Libraries


## 3D Graphics Libraries


## Geometry Libraries


## Database Libraries


## Networking Libraries


## GUI Libraries


## Montecarlo Libraries


## PROOF Libraries


## Language Bindings


## JavaScript ROOT


## Tutorials


## Class Reference Guide


## Build, Configuration and Testing Infrastructure

- If `-Droottest=ON` is specified, the ROOT build system used to clone a matching branch of the `roottest` repository.
This logic has been improved and is now as follows:
_(i)_ If the current head is a well-known branch, e.g. `master` or `v6-28-00-patches`, use the matching branch upstream;
_(ii)_ otherwise, try a branch that matches the name of the current head in the forked repository, if it exists; else try using the closest upstream head/tag below `HEAD`'s parent commit;
_(iii)_ as a last resort, if there is no preferred candidate, checkout the remote's default head.

