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
 Dennis Klein, GSI,\
 Sergey Linev, GSI,\
 Pere Mato, CERN/EP-SFT,\
 Lorenzo Moneta, CERN/EP-SFT,\
 Alja Mrak Tadel, UCSD/CMS,\
 Axel Naumann, CERN/EP-SFT,\
 Vincenzo Eduardo Padulano, CERN/EP-SFT,\
 Danilo Piparo, CERN/EP-SFT,\
 Fons Rademakers, CERN/IT,\
 Jonas Rembser, CERN/EP-SFT,\
 Matevz Tadel, UCSD/CMS,\
 Vassil Vassilev, Princeton/CMS,\
 Wouter Verkerke, NIKHEF/Atlas,

## Deprecation and Removal
- The RooFit legacy iterators are deprecated and will be removed in ROOT 6.34 (see section "RooFit libraries")
- Some memory-unsafe RooFit interfaces were removed
- Some redundant **RooDataSet** constructors are deprecated and will be removed in ROOT 6.34.
  Please use the RooDataSet constructors that take RooFit command arguments instead
- ROOT does not longer support Python 2. The minimum required Python version to build ROOT is 3.8.
- Support for wildcard imports like `from ROOT import *` is dropped from PyROOT
- Support for external (ie. non-builtin) libAfterImage is now deprecated and it will be removed in next release 6.34.
- The `TList::TList(TObject*)` constructor is deprecated and will be removed in ROOT 6.34
- The deprecated `TProofOutputList::TProofOutputList(TObject *o)` constructor was removed

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

## Parallelism
  - The ROOT::Experimental::TFuture template has been removed.

## RooFit Libraries

### New CPU likelihood evaluation backend by default

The new vectorizing CPU evaluation backend is not the default for RooFit likelihoods.
Likelihood minimization is now up to 10x faster on a single CPU core.

If you experience unexpected problems related to the likelihood evaluation, you
can revert back to the old backend by passing `RooFit::EvalBackend("legacy")`
to `RooAbsPdf::fitTo()` or `RooAbsPdf::createNLL()`.

In case you observe any slowdowns with the new likelihood evaluation, please
open a GitHub issue about this, as such a performance regression is considered
a bug.

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

### Deprecation of legacy test statistics classes in public interface

Instantiating the following classes and even including their header files is deprecated, and the headers will be removed in ROOT 6.34:

* RooAbsTestStatistic
* RooAbsOptTestStatistic
* RooNLLVar
* RooChi2Var
* RooXYChi2Var

Please use the higher-level functions `RooAbsPdf::createNLL()` and `RooAbsPdf::createChi2()` if you want to create objects that represent test statistics.

### Change of RooParamHistFunc

The `RooParamHistFunc` didn't take any observable `RooRealVar` as constructor
argument. It assumes as observable the internal variables in the passed
RooDataHist. This means it was in most contexts unusable, because the input
can't be changed, other than loading a different bin in the dataset.

Furthermore, there was actually a constructor that took a `RooAbsArg x`, but it
was simply ignored.

To fix all these problems, the existing constructors were replaced by a new one
that takes the observable explicitly.

Since the old constructors resulted in wrong computation graphs that caused
trouble with the new CPU evaluation backend, they had to be removed without
deprecation. Please adapt your code if necessary.

### Renaming of some RooFit classes

The `RooPower` was renamed to `RooPowerSum`, and `RooExpPoly` was renamed to `RooLegacyExpPoly`.

This was a necessary change, because the names of these classes introduced in ROOT 6.28 collided with some classes in CMS combine, which were around already long before. Therefore, the classes had to be renamed to not cause any problems for CMS.

In the unlikeliy case where you should have used these new classes for analysis already, please adapt your code to the new names and re-create your workspaces.

## RDataFrame

* The RDataFrame constructors that take in input one or more file names (or globs thereof) will now infer the format of the dataset, either TTree or RNTuple, that is stored in the first input file. When multiple files are specified, it is assumed that all other files contain a coherent dataset of the same format and with the same schema, exactly as it used to happen with TChain. This automatic inference further contributes towards a zero-code-change experience when moving from processing a TTree to processing an RNTuple dataset while using an RDataFrame. It also introduces a backwards-incompatible behaviour, i.e. now the constructor needs to open one file in order to infer the dataset type. This means that if the file does not exist, the constructor will throw an exception. Previously, an exception would be thrown only at a JIT-ting time, before the start of the computations.

## 2D Graphics Libraries


## 3D Graphics Libraries


## Geometry Libraries


## Database Libraries


## Networking Libraries


## GUI Libraries


## Montecarlo Libraries


## PROOF Libraries


## PyROOT

PyROOT was rebased on the latest version of the [cppyy library](https://cppyy.readthedocs.io/en/latest/).
This means PyROOT benefits from many upstream improvements and fixes, for example related to the conversion of NumPy arrays to vectors, implicit conversion from nested Python tuples to nested initializer lists, and improved overload resolution.

Related to this cppyy upgrade, there are some changes in PyROOT behavior.

### Different representation of `std::string`

Calling `repr()` on a `cppyy.gbl.std.string` object now comes with a "b" prefix, i.e. a bytes object is returned instead of a Python string.
This is an intentional change for better unicode support.

See: https://github.com/root-project/root/issues/15153#issuecomment-2040504962

### No more implicit conversion of static size `char` buffer to Python strings

A static size character buffer of type `char[n]` is not converted to a Python string anymore. 
The reason for this: since it was previously assumed the string was
null-terminated, there was no way to get the bytes after a `null`, even if you
wanted to.

```python
import ROOT

ROOT.gInterpreter.Declare("""
struct Struct { char char_buffer[5] {}; }; // struct with char[n]
void fill_char_buffer(Struct & st) {
    std::string foo{"foo"};
    std::memcpy(st.char_buffer, foo.data(), foo.size());
}
""")

struct = ROOT.Struct()
ROOT.fill_char_buffer(struct)
char_buffer = struct.char_buffer

# With thew new cppyy, you get access to the lower level buffer instead of a
# Python string:
print("struct.char_buffer            : ", char_buffer)

# However, you can turn the buffer into a string very easily with as_string():
print("struct.char_buffer.as_string(): ", char_buffer.as_string())
```
The output of this script with ROOT 6.32:
```
struct.char_buffer            :  <cppyy.LowLevelView object at 0x74c7a2682fb0>
struct.char_buffer.as_string():  foo
```

### Deprecate the attribute pythonization of `TDirectory` in favor of item-getting syntax

The new recommended way to get objects from a `TFile` or any `TDirectory` in general is now via `__getitem__`:

```python
tree = my_file["my_tree"] # instead of my_file.my_tree
```

This is more consistent with other Python collections (like dictionaries), makes sure that member functions can't be confused with branch names, and easily allows you to use string variables as keys.

With the new dictionary-like syntax, you can also get objects with names that don't qualify as a Python variable. Here is a short demo:
```python
import ROOT

with ROOT.TFile.Open("my_file.root", "RECREATE") as my_file:

    # Populate the TFile with simple objects.
    my_file.WriteObject(ROOT.std.string("hello world"), "my_string")
    my_file.WriteObject(ROOT.vector["int"]([1, 2, 3]), "my vector")

    print(my_file["my_string"])  # new syntax
    print(my_file.my_string)  # old deprecated syntax

    # With the dictionary syntax, you can also use names that don't qualify as
    # a Python variable:
    print(my_file["my vector"])
    # print(my_file.my vector) # the old syntax would not work here!
```

The old pythonization with the `__getattr__` syntax still works, but emits a deprecation warning and will be removed from ROOT 6.34.

## Language Bindings


## JavaScript ROOT


## Tutorials


## Class Reference Guide


## Build, Configuration and Testing Infrastructure


