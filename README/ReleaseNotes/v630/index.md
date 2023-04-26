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
- The RDataFrame factory functions `MakeCsvDataFrame`, `MakeArrowDataFrame`, `MakeNTupleDataFrame` and `MakeSqliteDataFrame` that were deprecated in v6.28 have been removed. Use `FromCSV`, `FromArrow`, `FromRNTuple` or `FromSqlite` instead.
- The TStorage reallocation routine without a size (`TStorage::ReAlloc(void *ovp, size_t size`) and heap
related routines (`TStorage::AddToHeap`, `TStorage::IsOnHeap`, `TStorage::GetHeapBegin`, `TStorage::GetHeapEnd`)
that were deprecated in v6.02/00.
- The deprecated `Format(const char* option, int sigDigits)` option for `RooAbsPdf::paramOn()` was removed. Please use the `Format(const char* option, ...)` overload that takes command arguments.
- The deprecated `RooAbsPdf::paramOn()` overload that directly takes a formatting string was removed. Please take the overload that uses command arguments.

## Core Libraries


## I/O Libraries


## TTree Libraries


## Histogram Libraries


## Math Libraries

### Behavior change of `TMath::AreEqualAbs()`

The `TMath::AreEqualAbs()` compares two numbers for equality within a certain absolute range.
So far, it would tell you that `inf != inf` if you define `inf` as `std::numeric_limits<double>::infinity()`, which is inconsistent with the regular `==` operator.

This is unexpected, because one would expect that if two numbers are considered exactly equal, they would also be considered equal within any range.
Therefore, the behavior of `TMath::AreEqualAbs()` was changed to return always `true` if the `==` comparision would return `true`.

## RooFit Libraries

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
RooMomentMorphFuncND func{<c'tor args you previously passed to RooMomentMorphFunc>};

func.setPdfMode(); // change behavior to be exactly like the former RooMomentMorphND

// Pass the selfNormalized=true` flag to the wrapper because the
RooMomentMorphFuncND already normalizes itself in pdf mode.
RooWrapperPdf pdf{"pdf_name", "pdf_name", func, /*selfNormalized=*/true};
```

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


