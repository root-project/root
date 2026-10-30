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
* The `RooTrace` class was untested and is now removed. For memory debugging, please use general-purpose tools such as AddressSanitizer or Valgrind instead.
* The PyMVA library and the associated `tmva-pymva` build option that were deprecated in ROOT 6.42 have been removed

## Python Interface

## I/O

## Core

## Histograms

## Math

## RooFit

* `RooKeysPdf` and `RooNDKeysPdf` now support datasets with negative event weights, such as sWeights or simulated samples with negative weights ([#12639](https://github.com/root-project/root/issues/12639)). Before, a single negative weight made `RooKeysPdf` return NaN everywhere, and `RooNDKeysPdf` could return NaN as well. The kernel widths are now computed from the magnitudes of the weights, and the events with negative weights subtract their kernels from the estimate. A dataset with a non-positive sum of weights now raises a `std::invalid_argument` exception, as it does not describe a distribution. Results for datasets without negative weights are unchanged.

## Graphics and GUI

## Geometry

## Documentation and Examples

## Build, Configuration and Testing


