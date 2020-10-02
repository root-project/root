% ROOT Version 6.24 Release Notes
% 2020-05-19
<a name="TopOfPage"></a>

## Introduction

ROOT version 6.24/00 is scheduled for release in November 2020.

For more information, see:

[http://root.cern](http://root.cern)

The following people have contributed to this new version:

 Guilherme Amadio, CERN/SFT,\
 Bertrand Bellenot, CERN/SFT,\
 Jakob Blomer, CERN/SFT,\
 Rene Brun, CERN/SFT,\
 Philippe Canal, FNAL,\
 Olivier Couet, CERN/SFT,\
 Massimiliano Galli, CERN/SFT,\
 Andrei Gheata, CERN/SFT,\
 Hadrien Grasland, IJCLab/LAL,\
 Enrico Guiraud, CERN/SFT,\
 Claire Guyot, CERN/SFT,\
 Stephan Hageboeck, CERN/SFT,\
 Sergey Linev, GSI,\
 Pere Mato, CERN/SFT,\
 Lorenzo Moneta, CERN/SFT,\
 Alja Mrak-Tadel, UCSD/CMS,\
 Axel Naumann, CERN/SFT,\
 Vincenzo Eduardo Padulano, Bicocca/SFT,\
 Danilo Piparo, CERN/SFT,\
 Fons Rademakers, CERN/SFT,\
 Oksana Shadura, UNL/CMS,\
 Enric Tejedor Saavedra, CERN/SFT,\
 Matevz Tadel, UCSD/CMS,\
 Vassil Vassilev, Princeton/CMS,\
 Wouter Verkerke, NIKHEF/Atlas,\
 Stefan Wunsch, CERN/SFT

## Deprecation and Removal


## Core Libraries

Due to internal changes required to comply with the deprecation of Intel TBB's `task_scheduler_init` and related
interfaces in recent TBB versions, as of v6.24 ROOT will not honor a maximum concurrency level set with
`tbb::task_scheduler_init` but will require instead the usage of `tbb::global_control`:

```cpp
  //tbb::task_scheduler_init init(2); // does not affect the number of threads ROOT will use anymore

  tbb::global_control c(tbb::global_control::max_allowed_parallelism, 2);
  ROOT::TThreadExecutor p1;  // will use 2 threads
  ROOT::TThreadExecutor p2(/*nThreads=*/8); // will still use 2 threads
```

Note that the preferred way to steer ROOT's concurrency level is still through `[ROOT::EnableImplicitMT](https://root.cern/doc/master/namespaceROOT.html#a06f2b8b216b615e5abbc872c9feff40f)` or by passing the appropriate parameter to executors' constructors, as in `[TThreadExecutor::TThreadExecutor](https://root.cern/doc/master/classROOT_1_1TThreadExecutor.html#ac7783d52c56cc7875d3954cf212247bb)`.

See the discussion at [ROOT-11014](https://sft.its.cern.ch/jira/browse/ROOT-11014) for more context.

## I/O Libraries


## TTree Libraries


## RDataFrame

- With [ROOT-10023](https://sft.its.cern.ch/jira/browse/ROOT-10023) fixed, RDataFrame can now read and write certain branches containing unsplit objects, i.e. TBranchObjects. More information is available at [ROOT-10022](https://sft.its.cern.ch/jira/browse/ROOT-10022).
- Snapshot now respects the basket size and split level of the original branch when copying branches to a new TTree.
- For some `TTrees`, RDataFrame's `GetColumnNames` method returns multiple valid spellings for a given column. For example, leaf `"l"` under branch `"b"` might now be mentioned as `"l"` as well as `"b.l"`, while only one of the two spellings might have been recognized before.
- Introduce `ROOT::RDF::RunGraphs`, which allows to compute the results of multiple RDataFrames concurrently while sharing the same thread pool. The computation may be more efficient than running the RDataFrames sequentially if an analysis consists of many RDataFrames, which don't have enough data to fully utilize the available resources.
- CSV files can now be opened and processed from HTTP(S) locations
- Certain RDF-related types in the ROOT::Detail and ROOT::Internal namespaces have been renamed, most notably `RCustomColumn` is now `RDefine`. This does not impact code that only makes use of entities in the public ROOT namespace, and should not impact downstream code unless it was patching or reusing internal RDataFrame types.
- Just-in-time compilation of string expressions passed to `Filter` and `Define` now generates functions that take fundamental types by const value (rather than by non-const reference as before). This will break code that was assigning to column values in string expressions: this is an intended side effect as we want to prevent non-expert users from performing assignments (=) rather than comparisons (==). Expert users can resort to compiled callables if they absolutely have to assign to column values (not recommended). See [ROOT-11009](https://sft.its.cern.ch/jira/browse/ROOT-11009) for further discussion.

## Histogram Libraries


## Math Libraries

- Update the definitions of the physical constants using the recommended 2018 values from NIST.
 - Use also the new SI definition of base units from 2019, where the Planck constant, the Boltzman constant , the elementary electric charge and the Avogadro constant are exact numerical values. See
 <https://en.wikipedia.org/wiki/2019_redefinition_of_the_SI_base_units>. Note that with this new definition the functions `TMath::HUncertainty()`, `TMath::KUncertainty()`,
 `TMath::QeUncertainty()` and `TMath::NaUncertainty()` all return a  `0.0` value.



## RooFit Libraries


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

- a new cmake variable, `CMAKE_INSTALL_PYTHONDIR`, has been added: it allows customization of the installation directory of ROOT's python modules

## PyROOT

- Deprecate `TTree.AsMatrix` in this release and mark for removal in v6.26. Please use instead `RDataFrame.AsNumpy`.
