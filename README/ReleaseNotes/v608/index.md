% ROOT Version 6.08 Release Notes
% 2015-11-12
<a name="TopOfPage"></a>

## Introduction

ROOT version 6.08/00 is scheduled for release in May, 2016.

For more information, see:

[http://root.cern.ch](http://root.cern.ch)

The following people have contributed to this new version:

 David Abdurachmanov, CERN, CMS,\
 Bertrand Bellenot, CERN/SFT,\
 Rene Brun, CERN/SFT,\
 Philippe Canal, Fermilab,\
 Cristina Cristescu, CERN/SFT,\
 Olivier Couet, CERN/SFT,\
 Gerri Ganis, CERN/SFT,\
 Andrei Gheata, CERN/SFT,\
 Christopher Jones, Fermilab, CMS,\
 Wim Lavrijsen, LBNL, PyRoot,\
 Sergey Linev, GSI, http,\
 Pere Mato, CERN/SFT,\
 Lorenzo Moneta, CERN/SFT,\
 Axel Naumann, CERN/SFT,\
 Danilo Piparo, CERN/SFT,\
 Timur Pocheptsov, CERN/SFT,\
 Fons Rademakers, CERN/IT,\
 Enric Tejedor Saavedra, CERN/SFT,\
 Liza Sakellari, CERN/SFT,\
 Manuel Tobias Schiller,\
 David Smith, CERN/IT,\
 Matevz Tadel, UCSD/CMS, Eve,\
 Vassil Vassilev, Fermilab/CMS,\
 Wouter Verkerke, NIKHEF/Atlas, RooFit

<a name="core-libs"></a>

## Core Libraries

ROOT prepares for [cxx modules](http://clang.llvm.org/docs/Modules.html). One of
the first requirements is its header files to be self-contained (section "Missing
Includes"). ROOT header files were cleaned up from extra includes and the missing
includes were added.

This could be considered as backward incompatibility (for good however). User
code may need to add extra includes, which were previously resolved indirectly
by including a ROOT header. For example:

  * TBuffer.h - TObject.h doesn't include TBuffer.h anymore. Third party code,
    replying on the definition of TBufer will need to include TBuffer.h, along
    with TObject.h.
  * TSystem.h - for some uses of gSystem.
  * GeneticMinimizer.h
  * ...

Other improvements, which may cause compilation errors in third party code:
  * If you get std::type_info from Rtypeinfo.h, type_info should be spelled
    std::type_info.

### Containers
A pseudo-container (generator) was created, ROOT::TSeq<T>. This template is
inspired by the xrange built-in function of Python. See the example
[here](https://root.cern.ch/doc/master/cnt001__basictseq_8C.html).

### Dictionaries

Fix ROOT-7760: fully allow the usage of the dylib extension on OSx.

### Interpreter Library

Exceptions are now caught in the interactive ROOT session, instead of terminating ROOT.

## Parallelisation
Three methods have been added to manage implicit multi-threading in ROOT: ROOT::EnableImplicitMT(numthreads), ROOT::DisableImplicitMT and ROOT::IsImplicitMTEnabled. They can be used to enable, disable and check the status of the global implicit multi-threading in ROOT, respectively.

## I/O Libraries
Custom streamers need to #include TBuffer.h explicitly (see
[section Core Libraries](#core-libs))


## TTree Libraries

* Repair setting the branch address of a leaflist style branch taking directly the address of the struct.  (Note that leaflist is nonetheless still deprecated and declaring the struct to the interpreter and passing the object directly to create the branch is much better).
* Provide an implicitly parallel implementation of TTree::GetEntry. The approach is based on creating a task per top-level branch in order to do the reading, unzipping and deserialisation in parallel. In addition, a getter and a setter methods are provided to check the status and enable/disable implicit multi-threading for that tree (see Parallelisation section for more information about implicit multi-threading).

## Histogram Libraries

* TH2Poly has a functional Merge method.

## Math Libraries


## RooFit Libraries


## 2D Graphics Libraries


## 3D Graphics Libraries


## Geometry Libraries


## Database Libraries

* Fix `TPgSQLStatement::SetBinary` to actually handle binary data (previous limited to ascii).

## Networking Libraries


## GUI Libraries


## Montecarlo Libraries


## PROOF Libraries


## Language Bindings

### Notebook integration

  * Refactoring of the Jupyter integration layer into the new package JupyROOT.
  * Added ROOT [Jupyter Kernel for ROOT](https://root.cern.ch/root-has-its-jupyter-kernel)
    * Magics are now invoked with standard syntax "%%", for example "%%cpp".
    * The methods "toCpp" and "toPython" have been removed.
  * Factorise output capturing and execution in an accelerator library and use ctypes to invoke functions.
  * When the ROOT kernel is used, the output is consumed progressively

## JavaScript ROOT


## Tutorials


## Class Reference Guide

## Build, Configuration and Testing Infrastructure


