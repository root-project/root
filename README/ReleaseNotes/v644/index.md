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
* Flattening nested **RooSimultaneous** in the constructor, deprecated in ROOT 6.42, is removed. RooSimultaneous components are now kept as they are, which also allows adding them with `RooSimultaneous::addPdf()`. They are only supported if their index category is a parameter ("switch mode"), which now also works for toy generation and for plotting slices of the outer RooSimultaneous. If the index category of a nested RooSimultaneous is in the dataset, creating the likelihood fails with a clear error for all evaluation backends. To combine channels, use one RooSimultaneous over a **RooCategory** with one state per channel, or over a **RooSuperCategory** if the channels span the full product of several categories. RooSimultaneous objects that were flattened by previous ROOT versions and are read from files are not affected. The internal `RooSimultaneous::InitializationOutput` struct and `RooSimultaneous::initialize()` function are removed as well.
* The PyMVA library and the associated `tmva-pymva` build option that were deprecated in ROOT 6.42 have been removed

## Python Interface

## I/O

## Core

## Histograms

## Math

## RooFit

## Graphics and GUI

## Geometry

## Documentation and Examples

## Build, Configuration and Testing


