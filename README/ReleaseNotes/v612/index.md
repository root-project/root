% ROOT Version 6.12 Release Notes
% 2017-05-18

<a name="TopOfPage"></a>

## Introduction

ROOT version 6.12/00 is scheduled for release in 2018.

For more information, see:

[http://root.cern.ch](http://root.cern.ch)

The following people have contributed to this new version:

 Bertrand Bellenot, CERN/SFT,\
 Georgios Bitzes, CERN/IT,\
 Rene Brun, CERN/SFT,\
 Philippe Canal, FNAL,\
 Olivier Couet, CERN/SFT,\
 Gerri Ganis, CERN/SFT,\
 Andrei Gheata, CERN/SFT,\
 Sergey Linev, GSI, http,\
 Pere Mato, CERN/SFT,\
 Lorenzo Moneta, CERN/SFT,\
 Axel Naumann, CERN/SFT,\
 Danilo Piparo, CERN/SFT,\
 Fons Rademakers, CERN/SFT,\
 Enric Tejedor Saavedra, CERN/SFT,\
 Vassil Vassilev, Fermilab/CMS,\
 Wouter Verkerke, NIKHEF/Atlas, RooFit

## Removed interfaces

The following interfaces have been removed, after deprecation in v6.10.

- Remove the deprecated `TSelectorCint.h` and `TSelectorCint.cxx`.
- Remove the deprecated `Riosfwd.h` and `Rtypeinfo.h`.


### TTreeReader

`TTreeReader::SetLastEntry()` was replaced by `TTreeReader::SetEntriesRange()`.



## Core Libraries

- When invoking root with the "-t" argument, ROOT enables thread-safety and,
  if configured, implicit multithreading within ROOT.


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
  - In some case it was not possible to zoom a 1D histogram using the mouse
    on the X axis. This was described
    [here](https://root-forum.cern.ch/t/axis-blocked-when-overlaying-two-histograms/25326)
  - When drawing an histogram with option "BOX" with log scale along the Z axis
    the bins were not visible in some case. This was described
    [here](https://root-forum.cern.ch/t/set-logscale-on-z-axis-in-2d-histo/25385).
  - When a TGraph2D was plotted with the option "PCOLZ" with a log scale along the
    Z axis, there was a mismatch between the markers' colors and the color palette
    displayed. It is now fixed. It was reported
    [here](https://sft.its.cern.ch/jira/browse/ROOT-8200).
  - It is now possible to set the titles and the axis ranges of a TMultiGraph drawn as 3D lines.

## 3D Graphics Libraries


## Geometry Libraries


## Database Libraries


## Networking Libraries


## GUI Libraries


## Montecarlo Libraries


## Parallelism
  - The TTaskGroup class has been added to the ROOT::Experimental namespace. It allows to submit to the runtime
  item of work which are dealt with in parallel;
  - The Async template function has been added the ROOT::Experimental namespace. The template function is analogous
  to *std::async* but without the possibility of specifying the execution policy and without creating a thread but
  directly submitting the work to the runtime in order to use the same pool as any other item of work spawned by ROOT.
  - The TFuture template has been added to the ROOT::Experimental namespace. It represents a future and is compatible
  with the ROOT::Experimental::Async function. It has the same properties of an STL future and can be initialised by
  one of these classes. For example, *TFuture<int> = std::async(myfunc,a,b,c);*


## Language Bindings


## JavaScript ROOT


## Tutorials

## Command line tools
  - `rootls` has been extended.
    - option `-l` displays the year
    - option `-t` displays all details of 'THnSparse'


## Class Reference Guide
  - The list of libraries needed by each class is displayed as a diagram.

## Build, Configuration and Testing Infrastructure


