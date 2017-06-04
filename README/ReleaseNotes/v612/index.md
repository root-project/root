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

### TTreeReader

`TTreeReader::SetLastEntry()` was replaced by `TTreeReader::SetEntriesRange()`.



## Core Libraries

### `TObjString` to `TString`

`TObjString::GetString()` now returns a `const TString&` to the `TString` inside the `TObjString`, instead of copying it.
This is to prevent very common misunderstanding of the interface.

In several cases, the misunderstanding of the interface caused invalid memory accesses to the already destructed
temporary `TString` returned by `GetString()`, e.g. `objStr->GetString().Data()`. This will be fixed automatically by the
new return type.

In rare cases, the caller expected `GetString()` to return a (non-const) reference to the embedded `TString`, e.g.
`objString->GetString().ReplaceAll("a", "b"); // WRONG!` This will now fail to compile, instead of not doing what the author of the
code expected. Please fix that code by using the `TObjString::String()` interface, which returns a non-const `TString&`:
`objString->String().ReplaceAll("a", "b");`.

In extremely rare cases, this change breaks a valid use where the temporary `TString` was modified and then captured in a new `TString`
object before the destruction of the temporary: `TString str = objStr->GetString().ReplaceAll("a", "b");`. In these rare cases,
please use the new function `CopyString()` which clearly indicates that it involves a temporary.

## I/O Libraries

- Introduce TKey::ReadObject<typeName>.  This is a user friendly wrapper around ReadObjectAny.  For example
```{.cpp}
auto h1 = key->ReadObject<TH1>
```
after which h1 will either be null if the key contains something that is not a TH1 (or derived class)
or will be set to the address of the histogram read from the file.

## Histogram Libraries


## Math Libraries


## RooFit Libraries


## TTree Libraries


## 2D Graphics Libraries
  - The method TColor::InvertPalette inverts the current palette. The top color becomes
    bottom and vice versa. This was [suggested by Karl Smith](https://root-forum.cern.ch/t/inverted-color-palettes/24826/2).
  - New method `TColor::SetColorThreshold(Float_t t)` to specify the color
    threshold used by GetColor to retrieve a color.
  - Improvements in candle plots:
    -  LogZ for violins
    -  scaling of candles and violins with respect to each other
    -  static functions for WhiskerRange and BoxRange

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


