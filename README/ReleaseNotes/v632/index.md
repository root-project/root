% ROOT Version 6.32 Release Notes
% 2023-10-10
<a name="TopOfPage"></a>

## Introduction

ROOT version 6.32/00 is scheduled for release around May 2024.

For more information, see:

[http://root.cern](http://root.cern)

The following people have contributed to this new version:

 Guilherme Amadio, CERN/IT,\
 Bertrand Bellenot, CERN/EP-SFT,\
 Jakob Blomer, CERN/EP-SFT,\
 Rene Brun,\
 Philippe Canal, FNAL,\
 Jolly Chen, CERN/EP-SFT,\
 Olivier Couet, CERN/EP-SFT,\
 Gerri Ganis, CERN/EP-SFT,\
 Florine de Geus, CERN/ATLAS,\
 Andrei Gheata, CERN/EP-SFT,\
 Enrico Guiraud, CERN/EP-SFT and Princeton,\
 Jonas Hahnfeld, CERN/EP-SFT,\
 Akeem Hart, Queen Mary University of London/DUNE and MINERvA,\
 Sergey Linev, GSI,\
 Pere Mato, CERN/EP-SFT,\
 Lorenzo Moneta, CERN/EP-SFT,\
 Alja Mrak Tadel, UCSD/CMS,\
 Axel Naumann, CERN/EP-SFT,\
 Vincenzo Padulano, CERN/EP-SFT,\
 Danilo Piparo, CERN/EP-SFT,\
 Fons Rademakers, CERN/IT,\
 Jonas Rembser, CERN/EP-SFT,\
 Matevz Tadel, UCSD/CMS,\
 Vassil Vassilev, Princeton/CMS,\
 Wouter Verkerke, NIKHEF/Atlas,

## Deprecation and Removal
- The RooFit legacy iterators are deprecated and will be removed in ROOT 6.34 (see section "RooFit libraries")
- Some memory-unsafe RooFit interfaces were removed

## Core Libraries


## I/O Libraries


## TTree Libraries
### Add files from subdirectories with `TChain::Add` globbing
It is now possible to add files from multiple subdirectories with `TChain::Add` globbing. For example,
```
TChain::Add("/path/to/tree/*/*.root")
```
grabs all the root files with the path `/path/to/tree/somedir/file.root` (but not `/path/to/tree/file.root` and `/path/to/tree/somedir/anotherdir/file.root`).

Another example:
```
TChain::Add("/path/to/tree/subdir[0-9]/*.root")
```
This grabs all the root files in subdirectories that have a name starting with `subdir` and ending with some digit.

## Histogram Libraries


## Math Libraries


## RooFit Libraries

### Compile your code with memory safe interfaces

If you define the `ROOFIT_MEMORY_SAFE_INTERFACES` preprocessor macro, the
RooFit interface changes in a way such that memory leaks are avoided.

The most prominent effect of this change is that many functions that used to
return an owning pointer (e.g., a pointer to an object that you need to
manually `delete`) are then returning a `std::unique_pt` for automatic memory
management.

For example this code would not compile anymore, because there is the rist that
the caller forgets to `delete params`:
```c++
RooArgSet * params = pdf.getParameters(nullptr);
```
If you wrap such return values in a `std::unique_ptr`, then your code will
compile both with and without memory safe interfaces:
```c++
std::unique_ptr<RooArgSet> params{pdf.getParameters(nullptr)};
```

Also some `virtual` RooFit functions like [RooAbsReal::createIntegral()](https://root.cern.ch/doc/master/classRooAbsReal.html#aff4be07dd6a131721daeeccf6359aea9)
are returning a different type conditional on `ROOFIT_MEMORY_SAFE_INTERFACES`.
If you are overriding such a function, you need to use the `RooFit::OwningPtr`
return type, which is an alias for `std::unique_ptr` in memory-safe mode or an
alias for a raw pointer otherwise.
```c++
RooFit::OwningPtr<RooAbsReal> RooAbsReal::createIntegral(...) const override
{
   std::unique_ptr<RooAbsReal> integral;
   // Prepare a std::unique_ptr as the return value
   ...
   // Use the RooFit::makeOwningPtr<T>() helper to translate the
   // std::unique_ptr to the actual return type (either std::unique_ptr<T> or T*).
   return RooFit::makeOwningPtr<RooAbsReal>(std::move(integral));
}
```

The biggest application of the memory-safe interfaces is to spot memory leaks
in RooFit-based frameworks. If you make sure that your framework compiles both
with and without `ROOFIT_MEMORY_SAFE_INTERFACES`, you can get rid of all memory
leaks related to RooFit user error! After making the necessary changes, you can
remove the marco definition again to keep backwards compatibility.

Note that the memory-safe interfaces might become the default at some point, so
doing this **backwards-compatible migration early** is strongly encouraged and
appreciated.

### Removal of some memory-unsafe interfaces

* The final `bool takeOwnership` parameter of the **RooAddition** and
  **RooStats::HistFactory::PiecewiseInterpolation** constructors was removed.
  This is to avoid situations where ownership is not clear to the compiler.
  Now, ownership of the input RooAbsArgs is never passed in the constructor. If
  you want the pass input ownership to the created object, please use
  `addOwnedComponents`. If you want to be extra safe, make sure the inputs are
  in an owning collection and then `std::move` the collection, so that the
  ownership is always clear.

  Example:
  ```c++
  RooArgList sumSet;
  sumSet.add(*(new RooRealVar("var1", "var1", 1.0)));
  sumSet.add(*(new RooRealVar("var2", "var2", 3.0)));
  RooAddition addition{"addition", "addition", sumSet, /*takeOwnership=*/true};
  ```
  should become:
  ```c++
  RooArgList sumSet;
  sumSet.addOwned(std::make_unique<RooRealVar>("var1", "var1", 1.0));
  sumSet.addOwned(std::make_unique<RooRealVar>("var2", "var2", 3.0));
  RooAddition addition{"addition", "addition", sumSet};
  addition.addOwnedComponents(std::move(sumSet));
  ```

### Deprecation of legacy iterators

The following methods related to the RooFit legacy iterators are deprecated and will be removed in ROOT 6.34.
They should be replaced with the suitable STL-compatible interfaces, or you can just use range-based loops:

- `RooAbsArg::clientIterator()`: use `clients()` and `begin()`, `end()` or range-based loops instead
- `RooAbsArg::valueClientIterator()`: use `valueClients()`
- `RooAbsArg::shapeClientIterator()`: use `shapeClients()`
- `RooAbsArg::serverIterator()`: use `servers()`
- `RooAbsArg::valueClientMIterator()`: use `valueClients()`
- `RooAbsArg::shapeClientMIterator()`: use `shapeClients()`
- `RooAbsArg::serverMIterator()`: use `servers()`

- `RooAbsCollection::createIterator()`: use `begin()`, `end()` and range-based for loops
- `RooAbsCollection::iterator()`: same
- `RooAbsCollection::fwdIterator()`: same

- `RooWorkspace::componentIterator()`: use `RooWorkspace::components()` with range-based loop

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


