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


