% ROOT Version 6.22 Release Notes
% 2020-01-10
<a name="TopOfPage"></a>

## Introduction

ROOT version 6.22/00 is scheduled for release in May, 2020.

For more information, see:

[http://root.cern](http://root.cern)

The following people have contributed to this new version:

 Guilherme Amadio, CERN/SFT,\
 Bertrand Bellenot, CERN/SFT,\
 Jakob Blomer, CERN/SFT,\
 Rene Brun, CERN/SFT,\
 Philippe Canal, FNAL,\
 Olivier Couet, CERN/SFT,\
 Andrei Gheata, CERN/SFT,\
 Enrico Guiraud, CERN/SFT,\
 Stephan Hageboeck, CERN/SFT,\
 Sergey Linev, GSI,\
 Pere Mato, CERN/SFT,\
 Lorenzo Moneta, CERN/SFT,\
 Alja Mrak-Tadel, UCSD/CMS,\
 Jan Musinsky, SAS Kosice,\
 Axel Naumann, CERN/SFT,\
 Vincenzo Eduardo Padulano, Bicocca/SFT,\
 Danilo Piparo, CERN/SFT,\
 Fons Rademakers, CERN/SFT,\
 Oksana Shadura, Nebraska,\
 Enric Tejedor Saavedra, CERN/SFT,\
 Matevz Tadel, UCSD/CMS,\
 Vassil Vassilev, Princeton/CMS,\
 Wouter Verkerke, NIKHEF/Atlas,\
 Stefan Wunsch, CERN/SFT

## Deprecation and Removal


## Core Libraries


## I/O Libraries


## TTree Libraries


## Histogram Libraries


## Math Libraries


## RooFit Libraries

### Type safe proxies to RooFit objects
RooFit's proxy classes have been modernised. The new class `RooProxy` allows for access to other RooFit objects
similarly to a smart pointer. In older versions of RooFit, the objects held by *e.g.* `RooRealProxy` had to be
accessed like this:
    RooAbsArg* absArg = realProxy.absArg();
    RooAbsPdf* pdf = dynamic_cast<RooAbsPdf*>(absArg);
    assert(pdf); // This should work, but the proxy doesn't have a way to check
    pdf->fitTo(...);
That is, a `RooRealProxy` stores a pointer to a RooAbsArg, and this pointer has to be casted. There was no type
safety, *i.e.* any object deriving from RooAbsArg could be stored in that proxy, and the user had to take care
of ensuring the correct types.
Now, if the class uses
    RooProxy<RooAbsPdf> pdfProxy;
instead of
    RooRealProxy realProxy;
the above code can be simplified to
    pdfProxy->fitTo(...);

Check the [doxygen reference guide](https://root.cern.ch/doc/master/classRooProxy.html) for `RooProxy` for
more information on how to modernise old code.

### HistFactory

#### Switch default statistical MC errors to Poisson
When defining HistFactory samples with statistical errors from C++, e.g.
    Sample background1( "background1", "background1", InputFile );
    background1.ActivateStatError();
statistical MC errors now have Poisson instead of Gaussian constraints. This better reflects the uncertainty of the MC simulations.
This can be reverted as follows:
    // C++:
    Channel chan("channel1");
    chan.SetStatErrorConfig( 0.05, "Gauss" );
    // Within <Channel ... > XML:
    <StatErrorConfig RelErrorThreshold="0.05" ConstraintType="Gauss" />

#### Less verbose HistFactory
HistFactory was very verbose, writing to the terminal with lots of `cout`. Now, many HistFactory messages are going
into RooFit's message stream number 2. The verbosity can therefore be adjusted using
    RooMsgService::instance().getStream(2).minLevel = RooFit::PROGRESS;

`hist2workspace` is also much less verbose. The verbosity can be restored with `hist2workspace -v` or `-vv`.

## 2D Graphics Libraries

 * Universal time (correct time zone and daylight saving time) in PDF file.

## 3D Graphics Libraries


## Geometry Libraries

### Geometry drawing in web browser

When ROOT compiled with -Droot7=ON flag, one can enable geometry drawing in web browser.
Just apply --web option when starting root like: `root --web tutorials/geom/rootgeom.C`
Not all features of TGeoPainter are supported - only plain drawing of selected TGeoVolume


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


