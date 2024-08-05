% ROOT Version 6.34 Release Notes
% 2025-05
<a name="TopOfPage"></a>

## Introduction

ROOT version 6.34.00 is scheduled for release at the end of May 2025.

For more information, see:

[http://root.cern](http://root.cern)

The following people have contributed to this new version:

 Anton Alkin, Sungkyunkwan University\
 Guilherme Amadio, CERN/IT,\
 Abhigyan Acherjee, University of Cincinnati,\
 Bertrand Bellenot, CERN/EP-SFT,\
 Jakob Blomer, CERN/EP-SFT,\
 Rene Brun,\
 Carsten Burgard, DESY\
 Will Buttinger, RAL,\
 Philippe Canal, FNAL,\
 Jolly Chen, CERN/EP-SFT,\
 Olivier Couet, CERN/EP-SFT,\
 Marta Czurylo, CERN/EP-SFT,\
 Monica Dessole, CERN/EP-SFT,\
 Mattias Ellert, Uppsala University,\
 Gerri Ganis, CERN/EP-SFT,\
 Florine de Geus, CERN/University of Twente,\
 Andrei Gheata, CERN/EP-SFT,\
 Bernhard Manfred Gruber,\
 Enrico Guiraud,
 Jonas Hahnfeld, CERN/Goethe University Frankfurt,\
 Fernando Hueso Gonzalez, University of Valencia\
 Attila Krasznahorkay, CERN/EP-ADP-OS,\
 Wim Lavrijsen, LBL,\
 Dennis Klein, GSI,\
 Christoph Langenbruch, Heidelberg University/LHCb,\
 Sergey Linev, GSI,\
 Javier Lopez-Gomez,\
 Pere Mato, CERN/EP-SFT,\
 Alaettin Serhan Mete, Argonne,\
 Thomas Madlener, DESY,\
 Lorenzo Moneta, CERN/EP-SFT,\
 Alja Mrak Tadel, UCSD/CMS,\
 Axel Naumann, CERN/EP-SFT,\
 Dante Niewenhuis, VU Amsterdam\
 Luis Antonio Obis Aparicio, University of Zaragoza,
 Ianna Osborne, Princeton University,\
 Vincenzo Eduardo Padulano, CERN/EP-SFT,\
 Danilo Piparo, CERN/EP-SFT,\
 Fons Rademakers, CERN/IT,\
 Jonas Rembser, CERN/EP-SFT,\
 Andrea Rizzi, University of Pisa,\
 Andre Sailer, CERN/EP-SFT,\
 Garima Singh, ETH,\
 Juraj Smiesko, CERN/RCS-PRJ-FC,
 Pavlo Svirin, National Technical University of Ukraine,\
 Maciej Szymanski, Argonne,\
 Christian Tacke, Darmstadt University,\
 Matevz Tadel, UCSD/CMS,\
 Alvaro Tolosa Delgado, CERN/RCS-PRJ-FC,\
 Devajith Valaparambil Sreeramaswamy, CERN/EP-SFT,\
 Peter Van Gemmeren, Argonne,\
 Vassil Vassilev, Princeton/CMS,\
 Wouter Verkerke, NIKHEF/ATLAS,
 Stefan Wunsch

## Deprecation and Removal

* The `RooAbsReal::plotSliceOn()` function that was deprecated since at least ROOT 6 was removed. Use `plotOn(frame,Slice(...))` instead.
* The `RooTemplateProxy` constructors that take a `proxyOwnsArg` parameter to manually pass ownership are deprecated and replaced by a new constructor that takes ownership via `std::unique_ptr<T>`. They will be removed in ROOT 6.36.

## Core Libraries

* The `rootcling` invocation corresponding to a `genreflex` invocation can be obtained with the new `genreflex`
  command line argument `--print-rootcling-invocation`. This can be useful when migrating from genreflex to
  rootcling.
* The `rootcling` utility now fully supports selection xml files and not only LinkDef files.

## I/O Libraries


## TTree Libraries

## RDataFrame

* The `GetColumnNames` function to retrieve the number of available column names in the RDataFrame object is now also
  usable from a node of a distributed computation graph. This makes the generation of said computation graph slightly
  less lazy than before. Notably, it used to be the case that a distributed computation graph could be defined with
  code that was not yet available on the user's local application, but that would only become available in the
  distributed worker. Now a call such as `df.Define("mycol", "return run_my_fun();")` needs to be at least declarable
  to the interpreter also locally so that the column can be properly tracked.

## Histogram Libraries


## Math Libraries


## RooFit Libraries

### Miscellaneous

* Setting `useHashMapForFind(true)` is not supported for RooArgLists anymore, since hash-assisted finding by name hash can be ambiguous: a RooArgList is allowed to have different elements with the same name. If you want to do fast lookups by name, convert your RooArgList to a RooArgSet.

## Graphics Backends

## 2D Graphics Libraries


## 3D Graphics Libraries


## Geometry Libraries


## Database Libraries


## Networking Libraries


## GUI Libraries


## Montecarlo Libraries


## PROOF Libraries


## PyROOT

### Typesafe `TTree::SetBranchAddress()` for array inputs

If you call `TTree::SetBranchAddress` with NumPy array or `array.array` inputs, ROOT will now check if the array type matches with the column type.
If it doesn't, `SetBranchAddress()` will return a negative status code and print an error.
Take for example this code snippet:
```python
arr = array.array(typecode, "d")
status = t.SetBranchAddress("name", arr)
print("Status = %s" % (status, ))
```
If the branch type is also `double` (like the type of the array indicated by `"d"`), the call to `SetBranchAddress()` would succeed with status code zero.
If the type doesn't match, you now get a clear error instead of garbage values.
```txt
Error in <TTree::SetBranchAddress>: The pointer type given "Double_t" (8) does not correspond to the type needed "Float_t" (5) by the branch: a
Status = -2
```

## Language Bindings


## JavaScript ROOT


## Tutorials


## Class Reference Guide


## Build, Configuration and Testing Infrastructure


