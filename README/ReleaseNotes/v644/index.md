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
* A `RooArgSet` cannot be constructed with an implicit conversion from Python `set` anymore.

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

## Graphics and GUI

## Geometry

## Documentation and Examples

## Build, Configuration and Testing


