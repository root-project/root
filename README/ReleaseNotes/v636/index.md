% ROOT Version 6.36 Release Notes
% 2025-05
<a name="TopOfPage"></a>

## Introduction

ROOT version 6.36.00 is scheduled for release at the end of May 2025.

For more information, see:

[http://root.cern](http://root.cern)

The following people have contributed to this new version:

 Bertrand Bellenot, CERN/EP-SFT,\
 Jakob Blomer, CERN/EP-SFT,\
 Philippe Canal, FNAL,\
 Mattias Ellert, Uppsala University,\
 Florine de Geus, CERN/University of Twente,\
 Fernando Hueso Gonzalez, University of Valencia,\
 Enrico Lusiani, INFN Padova,\
 Alberto Mecca, University of Turin,\
 Lorenzo Moneta, CERN/EP-SFT,\
 Mark Owen, University of Glasgow,\
 Vincenzo Eduardo Padulano, CERN/EP-SFT,\
 Giacomo Parolini, CERN/EP-SFT,\
 Danilo Piparo, CERN/EP-SFT,\
 Jonas Rembser, CERN/EP-SFT,\
 Manuel Tobias Schiller, University of Glasgow,\
 Surya Somayyajula, UMass Amherst,\
 Petr Stepanov, @petrstepanov,\
 Dongliang Zhang, University of Science and Technology of China

## Deprecation and Removal

* The RooFit legacy interfaces that were deprecated in ROOT 6.34 and scheduled for removal in ROOT 6.36 are removed. See the RooFit section in the 6.34 release notes for a full list.
* The `TPython::Eval()` function that was deprecated in ROOT 6.34 and scheduled for removal in ROOT 6.36 is removed.
* The `RooDataSet` constructors to construct a dataset from a part of an existing dataset are deprecated and will be removed in ROOT 6.38. This is to avoid interface duplication. Please use `RooAbsData::reduce()` instead, or if you need to change the weight column, use the universal constructor with the `Import()`, `Cut()`, and `WeightVar()` arguments.
* The ROOT splash screen was removed for Linux and macOS
* Proof support has been completely removed form RooFit and RooStats, after it was already not working anymore for several releases
* The build options `mysql`, `odbc`, `pgsql` and `qt5web` have been deprecated. Please complain with root-dev@cern.ch should you still need one!
* The `RooAbsPdf::generateSimGlobal()` method is deprecated and will be removed in ROOT 6.38. It only contains some workarounds that are not necessary anymore. One can just use the regular `RooAbsPdf::generate()` function.

## Python Interface

## RDataFrame

## RooFit

### Breaking function signature changes

  * The `RooRealVar::format()` function was changed to return a `std::string` instead of a `TString *`.
    The former return type was not memory safe, since the caller had to delete the `TString`.
    This pattern was not appropriate for a modern C++ library.
    If you absolutely need the old return type, wrap the call to `format()` inside `new TString{var.format(..)}`. However, this is not recommended.

## IO

* New options have been added to TFileMerger (which can be passed as whitespace-separated TStrings via `TFileMerger::SetMergeOptions`)
  * "FirstSrcCompression": when merging multiple files, instructs the class-specific merger to use the same compression as the
    first object of the destination's class as the destination's compression. Currently only recognized by the RNTuple merger;
  * "DefaultCompression": specifies that the merged output should use the class-specific default compression. Currently only
    meaningful for RNTuple, which has a default compression different from the TFile's default compression (ZSTD instead of ZLIB).
    This option is automatically set by `hadd` when no other compression option is specified;
  * "rntuple.MergingMode=(Filter|Union|Strict)": RNTuple-specific option that specifies the merging mode that should be used by
    the RNTupleMerger (see
    [RNTupleMergeOptions](https://root.cern/doc/v634/structROOT_1_1Experimental_1_1Internal_1_1RNTupleMergeOptions.html));
  * "rntuple.ErrBehavior=(Abort|Skip)": RNTuple-specific option that specifies the behavior of the RNTupleMerger on error (see link above);
  * "rntuple.ExtraVerbose": RNTuple-specific option that tells the RNTupleMerger to emit more information during the merge process.
* New attribute for I/O customization rules: `CanIgnore`.  When using this attribute the rule will be ignored if the input is missing from the schema/class-layout they apply to instead of issue a `Warning`

## RDataFrame

## Tutorials and Code Examples

## Core

## Histograms

## Math

## Graphics

## Geometry

## Montecarlo

## JavaScript ROOT

## Class Reference Guide

## Build, Configuration and Testing Infrastructure


