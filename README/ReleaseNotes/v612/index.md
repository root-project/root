% ROOT Version 6.12 Release Notes
% 2017-05-18

<a name="TopOfPage"></a>

## Introduction

ROOT version 6.12/00 is scheduled for release in 2018.

For more information, see:

[http://root.cern.ch](http://root.cern.ch)

The following people have contributed to this new version:

 David Abdurachmanov, CERN, CMS,\
 Bertrand Bellenot, CERN/SFT,\
 Philippe Canal, FNAL,\
 Olivier Couet, CERN/SFT,\
 Kyle Cranmer, NYU, RooStats,\
 George Troska, Dortmund Univ.,\
 Gerri Ganis, CERN/SFT,\
 Andrei Gheata, CERN/SFT,\
 Christopher Jones, Fermilab, CMS,\
 Sergey Linev, GSI, http,\
 Pere Mato, CERN/SFT,\
 Lorenzo Moneta, CERN/SFT,\
 Axel Naumann, CERN/SFT,\
 Danilo Piparo, CERN/SFT,\
 Timur Pocheptsov, CERN/SFT,\
 Enric Tejedor Saavedra, CERN/SFT,\
 Manuel Tobias Schiller,\
 David Smith, CERN/IT,\
 Matevz Tadel, UCSD/CMS, Eve,\
 Vassil Vassilev, Princeton University,\
 Wouter Verkerke, NIKHEF/Atlas, RooFit

## Removed interfaces

The following interfaces have been removed, after deprecation in v6.10.

- Remove the deprecated `TSelectorCint.h` and `TSelectorCint.cxx`.
- Remove the deprecated `Riosfwd.h` and `Rtypeinfo.h`.


### TTreeReader

`TTreeReader::SetLastEntry()` was replaced by `TTreeReader::SetEntriesRange()`.



## Core Libraries


## I/O Libraries

- Introduce TKey::ReadObject<typeName>.  This is a user friendly wrapper around ReadObjectAny.  For example
```{.cpp}
auto h1 = key->ReadObject<TH1>
```
after which h1 will either be null if the key contains something that is not a TH1 (or derived class)
or will be set to the address of the histogram read from the file.

## TTree Libraries

- Resolved O(N^2) scaling problem in ```TTree::Draw()``` observed when a branch that contains a
large TClonesArray where each element contains another small vector container.

## Histogram Libraries


## Math Libraries


## RooFit Libraries


## 2D Graphics Libraries
  - The method TColor::InvertPalette inverts the current palette. The top color becomes
    bottom and vice versa. This was [suggested by Karl Smith](https://root-forum.cern.ch/t/inverted-color-palettes/24826/2).
  - New method `TColor::SetColorThreshold(Float_t t)` to specify the color
    threshold used by GetColor to retrieve a color.
  - Improvements in candle plots:
    -  LogZ for violins
    -  scaling of candles and violins with respect to each other
    -  static functions for WhiskerRange and BoxRange
  - In some case it was not somme possible to zoom a 1D histogram using the mouse
    on the X axis. This was described
    [here](https://root-forum.cern.ch/t/axis-blocked-when-overlaying-two-histograms/25326)
  - When drawing an histogram with option "BOX" with log scale along the Z axis
    the bins were not visible in some case. This was described
    [here](https://root-forum.cern.ch/t/set-logscale-on-z-axis-in-2d-histo/25385).
  - When a TGraph2D was plotted with the option "PCOLZ" with a log scale along the
    Z axis, there was a mismatch between the markers' colors and the color palette
    displayed. It is now fixed. It was reported
    [here](https://sft.its.cern.ch/jira/browse/ROOT-8200).

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


