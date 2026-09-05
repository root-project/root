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


