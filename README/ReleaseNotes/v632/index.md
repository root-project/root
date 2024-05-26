% ROOT Version 6.32 Release Notes
% 2024-05-26
<a name="TopOfPage"></a>

## Introduction

ROOT version 6.32.00 was released on 28 May 2024.
This release is a long term support one, ideal for inclusion in production or
data taking software stacks of experiments.

For more information, see:

[http://root.cern](http://root.cern)

The following people have contributed to this new version:

 Anton Alkin, Sungkyunkwan University\
 Guilherme Amadio, CERN/IT,\
 Abhigyan Acherjee, University of Cincinnati,\
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
 Mattias Ellert, Uppsala University,\
 Gerri Ganis, CERN/EP-SFT,\
 Florine de Geus, CERN/University of Twente,\
 Andrei Gheata, CERN/EP-SFT,\
 Bernhard Manfred Gruber,\
 Enrico Guiraud,
 Jonas Hahnfeld, CERN/EP-SFT,\
 Fernando Hueso Gonzalez, University of Valencia\
 Attila Krasznahorkay, CERN/EP-ADP-OS,\
 Wim Lavrijsen, LBL,\
 Dennis Klein, GSI,\
 Christoph Langenbruch, Heidelberg University/LHCb,\
 Sergey Linev, GSI,\
 Javier Lopez-Gomez,\
 Pere Mato, CERN/EP-SFT,\
 Alaettin Serhan Mete, Argonne,\
 Thomas Madlener, DESY,\
 Lorenzo Moneta, CERN/EP-SFT,\
 Alja Mrak Tadel, UCSD/CMS,\
 Axel Naumann, CERN/EP-SFT,\
 Dante Niewenhuis, VU Amsterdam\
 Luis Antonio Obis Aparicio, University of Zaragoza,
 Ianna Osborne, Princeton University,\
 Vincenzo Eduardo Padulano, CERN/EP-SFT,\
 Danilo Piparo, CERN/EP-SFT,\
 Fons Rademakers, CERN/IT,\
 Jonas Rembser, CERN/EP-SFT,\
 Andrea Rizzi, University of Pisa,\
 Andre Sailer, CERN/EP-SFT,\
 Garima Singh, ETH,\
 Juraj Smiesko, CERN/RCS-PRJ-FC,
 Pavlo Svirin, National Technical University of Ukraine,\
 Maciej Szymanski, Argonne,\
 Christian Tacke, Darmstadt University,\
 Matevz Tadel, UCSD/CMS,\
 Alvaro Tolosa Delgado, CERN/RCS-PRJ-FC,\
 Devajith Valaparambil Sreeramaswamy, CERN/EP-SFT,\
 Peter Van Gemmeren, Argonne,\
 Vassil Vassilev, Princeton/CMS,\
 Wouter Verkerke, NIKHEF/ATLAS,
 Stefan Wunsch

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

The Cling interpreter now relies on LLVM version 16.

## I/O Libraries

### hadd respects compression settings

Fixed a bug that was previously changing the compression settings to a single digit number instead of the full value
(by default 101).

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

### Improved efficiency of TTree friends with indices

`TTreeIndex` and `TChainIndex` classes now implement the `Clone` method such that it does not use the ROOT I/O to clone the
index but just does a copy in memory. Notably, this improves processing efficiency for RDataFrame in multithreaded
execution since the same index must be copied over to all the threads and attached to the current tree for proper
event matching.

## RNTuple
ROOT's experimental successor of TTree has seen a number of updates since the last release. Specifically, 6.32 includes the following changes:

- A major refactoring of the interface, improving consistency across different parts and improving overall robustness. **Note that this is a breaking change with regard to 6.30!**
- The on-disk format has been updated to release candidate 2. **It will not be possible to read RNTuples written in the previous format anymore.**
- Support has been added for several new field types: `std::unordered_set<T>`, `std::map<K,V>`, `std::unordered_map<K,V>`
- Support has been added for on-disk half-precision (IEEE 754-2008 16-bit) float fields. This can be enabled through `RField<float>::SetHalfPrecision()`. On reading, values of such fields are represented as regular, 32-bit floats.
- A new `RNTupleInspector` utility class has been added, to provide information about the on-disk metadata of an RNTuple.
- A new `RNTupleParallelWriter` class has been added, providing (initial) support for parallel writing of RNTuples.
- A new static method `RFieldBase::Check()` has been added, which produces a support status report of a type with regards to RNTuple I/O.
- A new internal `RNTupleMerger` class has been added, enabling the merging of different page sources into one page sink. This also means that RNTuples can be merged through `hadd`.
- Zero-copy bulk reading has been added, with extra optimizations for `ROOT::RVec` fields.
- It is now possible to use the `RNTupleView` with an external address with type erasure, e.g.:
  ```cpp
  std::shared_ptr<void> data{new float()};
  auto view = reader->GetView("pt", data);
  ```
  This enables use cases such as reading one specific entry of one specific field into a previously allocated memory location.
- Further integration with [RDataFrame](#rdataframe): it is now possible to create RDataFrame for chains of RNTuples. This addition also comes with improvements to the multi-threaded work scheduling.
- Many additional bug fixes and improvements.

Please, report any issues regarding the above mentioned features should you encounter them. RNTuple is still in pre-production. The on-disk format is scheduled to be finalized by the end of 2024. Thus, we appreciate feedback and suggestions for improvement.

## Histogram Libraries

- Implement the FLT_MAX mechanism for `THStack::GetMaximum()` and `THStack::GetMiniumum()`.
- Print a warning when the range given to `TAxis::SetRange` is invalid.
- Fix projection name in `TH3` as requested [here](https://root-forum.cern.ch/t/project3d-letter-d-in-name-option/57612).

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

### Asymptotically correct uncertainties for extended unbinned likelihood fits

Added correct treatment of extended term in asymptotically correct method for uncertainty determination in the presence of weights.
This improvement will allow for extended unbinned maximum likelihood fits to use the asymptotically correct method when using the `RooFit::AsymptoticError()` command argument in [RooAbsPdf::fitTo()](https://root.cern.ch/doc/master/classRooAbsPdf.html#ab0721374836c343a710f5ff92a326ff5).
See also this [writeup on extended weighted fits](https://root.cern/files/extended_weighted_fits.pdf) that is also linked from the reference guide.
The [pull request](https://github.com/root-project/root/pull/14751) that introduced this feature might also be a good reference.

### Compile your code with memory safe interfaces

If you define the `ROOFIT_MEMORY_SAFE_INTERFACES` preprocessor macro, the
RooFit interface changes in a way such that memory leaks are avoided.

The most prominent effect of this change is that many functions that used to
return an owning pointer (e.g., a pointer to an object that you need to
manually `delete`) are then returning a `std::unique_pt` for automatic memory
management.

For example this code would not compile anymore, because there is the risk that
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

* The `RDataFrame` constructors that take in input one or more file names (or globs thereof) will now infer the format of the dataset, either `TTree` or `RNTuple`, that is stored in the first input file. When multiple files are specified, it is assumed that all other files contain a coherent dataset of the same format and with the same schema, exactly as it used to happen with `TChain`. This automatic inference further contributes towards a zero-code-change experience when moving from processing a `TTree` to processing an `RNTuple` dataset while using an `RDataFrame`. It also introduces a backwards-incompatible behaviour, i.e. now the constructor needs to open one file in order to infer the dataset type. This means that if the file does not exist, the constructor will throw an exception. Previously, an exception would be thrown only at a JIT-ting time, before the start of the computations.
* Distributed `RDataFrame` now supports processing an `RNTuple` dataset.
* In distributed `RDataFrame`, the `initialize` function useful to run initialization code at the beginning of every task
on a worker will now run only in the worker processes. Previously, it was also run eagerly at the point of calling, that
is in the main user process. This is done to better separate the user driver environment and the worker environments. If
necessary, the function passed to `initialize` can be called directly by the user in the main application to reproduce
the same effect as before.
* Some internal details of the `RDataFrame` implementation were reworked to decrease memory usage and runtime of programs
with very deep computation graphs (more than O(10K) nodes in the same branch). Preliminary tests indicate between 30%
and a factor 2.5 in memory decrease. This improvement is transparent for `RDataFrame` users.

## 2D Graphics Libraries

- TMultiGraph: Add the objects from the list of functions in legend produce by TLegend.
- Implement the IsInside method for TEllipse, TCrown and TDiamond. Also, a new graphics example `inside.C` has been added.
- Two new methods in TColor: `ListColors()` and `GetColorByname()`.
- Make sure the option `L` draws closed polygon for `TH2Poly`.
- Use Tex Gyre fonts for sans serif (similar to Helvetica) .
- The new method `TPad::ModifiedUpdate` is short cut to call `Modified()` and `Update()` in a single call. On Mac with Cocoa, it performs an additional ProcessEvents().
- Improve `SetTextSize` error: show code and values.
- Very long text string generated a wrong SVG file.
- Fix the option `SAME` works for `TGraph2D`.
- Implement the title for the palette of a `TH3`.
- Fix typo in `TLegend::PaintPrimitives()` and improve the exclusion graphs legend.
- `SetParameters(…)` or `SetParameter(…)` on a TF1 reset the properties of the axis that have been previously defined.
  This was due to the `Update()` that was done after the parameters definition.
- Update fonts' documentation (CMS request).
- Delaunay triangles were computed by the package `triangle.c` included in the ROOT code.
  This package had several problems:
      - It was not maintained anymore.
      - Its license was not compatible with LGPL.
  It is now replaced  by the CDT package which is properly maintained and has a license (MLP) compatible with LGPL

## 3D Graphics Libraries

### REve
* Introduce lightweight visualization of instanced shapes on the level of 100.000 instances. This is integrated in digit visualization of the type REveBoxSet. List of typed instances are boxes, hexagons, and cones. The digit sets support different types of transformation: positioning, rotation, and scaling in different combinations. With the digit set a palette GUI interface has also been added to enable setting digits threshold and value to color mapping.

<figure>
 <img src="reve-boxset-cones.png" >
 <figcaption>REveBoxSet screenshot with cone shape type. The set is using value to color map with overflow and underflow mark. The single REveBoxet object has a secondary selection enabled, where one can set a custom tooltip on mouse hover of an individual instance.</figcaption>
</figure>

* Update version of RenderCore to fix tone mapping of transparent objects.

## PROOF Libraries

By default, PROOF is not configured and built any more. It will be deprecated in the future given that its functionality is now provided by the superior RDataFrame and its distributed version, [DistRDF](https://root.cern/doc/master/classROOT_1_1RDataFrame.html#distrdf).

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

### Removal of Python 2 support

ROOT does no longer support Python 2. The minimum Python version necessary to use ROOT in a Python application is 3.8.
As a consequence, any reference to Python 2 in ROOT code was removed and certain configuration options are no longer
usable, e.g.

* `root-config --python2-version`
* cmake -Dpyroot-python2

The cmake build system now looks for the standard `Python3` package and previously custom Python-related cmake variables
are now just the ones automatically produced by cmake (see https://cmake.org/cmake/help/latest/module/FindPython.html).

### More usage of the public cppyy API

Many implementation details of the ROOT pythonizations were moved from C++ functions to pure Python bindings using the
public cppyy API. This helps in the integration with the tool but also improves code efficiency and memory usage.

## JavaScript ROOT

## Class Reference Guide

- Define missing doxygen groups.
- Fix a few typos in the `THStack` documentation.
- Small fixes in the `THistPainter` documentation.
- Improve the `TColor` documentation: use modern C++ in the examples.
- Make sure the python examples do not generate wrong namespaces in the documentation.
- The dataframe tutorials json spec files were not displayed properly. Moreover there was
  no direct correspondence between the code source and the json file. Those files do not
  have any doc in them. With a direct link to the GitHub source file the dependency between source
  code and json is now more obvious.
- Document how to remove shadow of `TPave`, as it was not evident (only explanations were hidden here and there in the forum).
- Improve the `SetFillColorAlpha` documentation.
- Simplify some graphics examples: arrow.C, crown.C, diamond.C and ellipse.C.
- Fix a typo in the documentation of `TGraph::SetHighlight` in `TGraph.cxx`.
- Change the marker style in the tutorial `df014_CSVDataSource`.
- Remove useless settings in the tutorial `scatter.C`.
- Fix the tutorial `h1analysisTreeReader.C`.
- Fix doxygen formatting in `TGNumberEntry.cxx`.
- Avoid the CDT documentation to appear in the reference guide.
- Remove last references to the old ROOT `drupal` website.

## Build, Configuration and Testing Infrastructure

Release v6.32.00 is the first one integrated and tested entirely through the new GitHub based build system.


