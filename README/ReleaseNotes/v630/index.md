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

## Core Libraries


## I/O Libraries


## TTree Libraries


## Histogram Libraries


## Math Libraries


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

* `root-config` can also print the path to the Python executable used when building ROOT. Options `--python-executable`,
  `--python2-executable`, `--python3-executable` are now supported by the utility.