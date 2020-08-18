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


## I/O Libraries


## TTree Libraries


## RDataFrame

- With [ROOT-10023](https://sft.its.cern.ch/jira/browse/ROOT-10023) fixed, RDataFrame can now read and write certain branches containing unsplit objects, i.e. TBranchObjects. More information is available at [ROOT-10022](https://sft.its.cern.ch/jira/browse/ROOT-10022).
- Snapshot now respects the basket size and split level of the original branch when copying branches to a new TTree.
- For some `TTrees`, RDataFrame's `GetColumnNames` method returns multiple valid spellings for a given column. For example, leaf `"l"` under branch `"b"` might now be mentioned as `"l"` as well as `"b.l"`, while only one of the two spellings might have been recognized before.

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
