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
* The `RooParamHistFunc` and `RooHistConstraint` classes are deprecated and will be removed in ROOT 6.46. Their functionality is too close to the HistFactory classes, which has repeatedly caused confusion about which classes to use and how they differ. Please use the HistFactory class `ParamHistFunc` instead, multiplied with a `RooHistFunc` and constrained by one `RooPoisson` per bin, like the models built by HistFactory. The [rf709_BarlowBeeston](https://root.cern/doc/v644/rf709__BarlowBeeston_8C.html) tutorial was rewritten to show how this works. We are aware that these classes might have been used in RooWorkspaces that were written to disk. If you face problems because of that, please reach out to the RooFit developers by opening a [GitHub issue](https://github.com/root-project/root/issues) or posting on the [ROOT forum](https://root-forum.cern.ch/).
* The **TMPIFile** class and the associated `mpi` build option (not to be confused with `minuit2_mpi`, which is unaffected) have been removed.


## Build system

### CMake version

As announced in the 6.42 release notes, the minimum CMake version required for building ROOT 6.44 rose to 3.26. It is the version shipped with RHEL 8.10, which is the oldest Linux distribution supported by this release.
After ROOT 6.44, the minimum may rise to 3.31 if an upgrade to LLVM 24 or later is performed, see the [minimum version of CMake required by LLVM](https://discourse.llvm.org/t/rfc-raising-minimum-required-cmake-version-to-3-31/91086), a warning will be emitted at configure time if your CMake version is older than this.

### RenderCore

RenderCore, the WebGL engine of the REve client, is fetched at build time as the tarball of its release, from [SPI](https://spi.web.cern.ch)'s [website](https://lcgpackages.web.cern.ch/) first and from its GitHub release second, instead of being bundled in `builtins/rendercore/`. A build without a network takes the tarball from `ROOT_DOWNLOAD_DIR`, as for every builtin. The build option `builtin_rendercore`, ON by default, controls the download. REve has no web location of RenderCore to fall back to, so the option can be OFF only when REve is not built, that is with `root7` or `webgui` OFF; otherwise the configuration stops with an error.

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
    removed, and `RooFit::EvalBackend("legacy")` now throws an exception.
  * The implementation classes of the legacy test statistics are removed:
    **RooNLLVar**, **RooChi2Var**, **RooAbsOptTestStatistic** and
    **RooAbsTestStatistic**. Their headers were not installed anymore
    since ROOT 6.34.
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
    the `RooFit::Evaluator`. The default evaluation backend of `RooUnbinnedL`
    and of `RooFit::TestStatistics::NLLFactory`, which is used by
    `buildLikelihood()`, changes from `legacy` to `cpu`.
  * A `RooFormulaVar` that wraps a likelihood now also reports the error level
    of the likelihood for the `cpu` backend, like `RooAddition`. Before, it only
    did this for the legacy backend, so minimizing such a `RooFormulaVar` with
    the `cpu` backend overestimated the parameter uncertainties by a factor
    of sqrt(2).

## Graphics and GUI

## Geometry

## Documentation and Examples

## Build, Configuration and Testing


