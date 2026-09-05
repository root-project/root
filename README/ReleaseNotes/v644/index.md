% ROOT Version 6.44 Release Notes
% 2027-05-15
<a name="TopOfPage"></a>

## Introduction

ROOT version 6.44.00 is scheduled for release in May 2027.

For more information, see:

[http://root.cern](http://root.cern)

The following people have contributed to this new version:

## Deprecation and Removal

* `TClassEdit::ESTLType`, `TDictionary::ESTLType`, `TStreamerElement::ESTLType` were redundant enums and have been removed. Please use `ROOT::ESTLType` instead.
* The `ROOT.SetHeuristicMemoryPolicy` method has been removed. This was introduced in 6.40 as a temporary debugging tool to aid users in addressing the changes in memory ownership heuristics.
* A `RooArgSet` cannot be constructed with an implicit conversion from Python `set` anymore. The problem is that Python sets are unordered while RooArgSets are ordered, and this mismatch can lead to subtle problems later on. Prefer conversion from Python lists or tuples, which are ordered too.
* The `RooFit::BatchMode` command argument has been removed, use `RooFit::EvalBackend` instead.
* The legacy evaluation backend of RooFit was deprecated in ROOT 6.42 and is now removed. See the RooFit section below for details.
* The `RooTrace` class was untested and is now removed. For memory debugging, please use general-purpose tools such as AddressSanitizer or Valgrind instead.
* Flattening nested **RooSimultaneous** in the constructor, deprecated in ROOT 6.42, is removed. RooSimultaneous components are now kept as they are, which also allows adding them with `RooSimultaneous::addPdf()`. They are only supported if their index category is a parameter ("switch mode"), which now also works for toy generation and for plotting slices of the outer RooSimultaneous. If the index category of a nested RooSimultaneous is in the dataset, creating the likelihood fails with a clear error for all evaluation backends. To combine channels, use one RooSimultaneous over a **RooCategory** with one state per channel, or over a **RooSuperCategory** if the channels span the full product of several categories. RooSimultaneous objects that were flattened by previous ROOT versions and are read from files are not affected. The internal `RooSimultaneous::InitializationOutput` struct and `RooSimultaneous::initialize()` function are removed as well.
* The PyMVA library and the associated `tmva-pymva` build option that were deprecated in ROOT 6.42 have been removed
* The **TMPIFile** class and the associated `mpi` build option (not to be confused with `minuit2_mpi`, which is unaffected) have been removed.


## Build system

### CMake version

As announced in the 6.42 release notes, the minimum CMake version required for building ROOT 6.44 rose to 3.26. It is the version shipped with RHEL 8.10, which is the oldest Linux distribution supported by this release.
After ROOT 6.44, the minimum may rise to 3.31 if an upgrade to LLVM 24 or later is performed, see the [minimum version of CMake required by LLVM](https://discourse.llvm.org/t/rfc-raising-minimum-required-cmake-version-to-3-31/91086), a warning will be emitted at configure time if your CMake version is older than this.

## Python Interface

## I/O

## Core

## Histograms

## Math

## RooFit

### Removal of the legacy evaluation backend

The `legacy` evaluation backend for likelihood and chi-square fits is removed.
It was superseded by the vectorized `cpu` backend, which is the default since
ROOT 6.32. After the removal of the legacy constant term optimization in ROOT 6.42, the
legacy backend also had no performance-relevant feature left that would justify
its continued maintenance.

Concretely, this means:

  * `RooFit::EvalBackend::Legacy()` and the corresponding enum value are
    removed. Passing `RooFit::EvalBackend("legacy")` to `fitTo()`,
    `createNLL()`, `chi2FitTo()` or `createChi2()` now throws an exception.
  * The implementation classes of the legacy test statistics are removed:
    **RooNLLVar**, **RooChi2Var**, **RooAbsOptTestStatistic** and
    **RooAbsTestStatistic**. Their headers were not part of the public
    interface anymore since ROOT 6.32, but they were still installed for
    backwards compatibility.
  * The old multiprocessing mechanism of the legacy backend is removed as well,
    consisting of the **RooRealMPFE** class and the underlying
    **BidirMMapPipe**. The `RooFit::NumCPU()` command argument no longer forks
    off one `RooRealMPFE` process per CPU: it now selects the number of worker
    threads for the multi-threaded batch evaluation in the RooBatchCompute
    library, and its interleaving strategy argument is ignored. For fits with a
    parallelized gradient, there is also the `RooFit::Parallelize()` argument,
    based on the `RooFit::MultiProcess` framework.
  * The `nll::name[pdf,data]` and `chi2::name[pdf,data]` expressions in the
    `RooWorkspace::factory()` language are removed, since they instantiated the
    removed classes directly. Use `RooAbsPdf::createNLL()` or
    `RooAbsReal::createChi2()` instead.
  * The `RooFit::TestStatistics::RooUnbinnedL` class now always evaluates with
    the `RooFit::Evaluator` and its `evalBackend` constructor parameter
    defaults to the `cpu` backend, like `RooFit::TestStatistics::NLLFactory`.

## Graphics and GUI

## Geometry

## Documentation and Examples

## Build, Configuration and Testing


