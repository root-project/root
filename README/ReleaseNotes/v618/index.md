% ROOT Version 6.18 Release Notes
% 2018-11-14
<a name="TopOfPage"></a>

## Introduction

ROOT version 6.18/00 is scheduled for release in May, 2019.

For more information, see:

[http://root.cern](http://root.cern)

The following people have contributed to this new version:

 Bertrand Bellenot, CERN/SFT,\
 Rene Brun, CERN/SFT,\
 Philippe Canal, FNAL,\
 Olivier Couet, CERN/SFT,\
 Gerri Ganis, CERN/SFT,\
 Andrei Gheata, CERN/SFT,\
 Stephan Hageboeck, CERN/SFT,\
 Sergey Linev, GSI,\
 Pere Mato, CERN/SFT,\
 Lorenzo Moneta, CERN/SFT,\
 Axel Naumann, CERN/SFT,\
 Danilo Piparo, CERN/SFT,\
 Fons Rademakers, CERN/SFT,\
 Enric Tejedor Saavedra, CERN/SFT,\
 Matevz Tadel, UCSD/CMS,\
 Vassil Vassilev, Princeton/CMS,\
 Wouter Verkerke, NIKHEF/Atlas,

## Deprecation and Removal

### THttpServer classes

The following methods were deprecated and removed:

   * Bool_t THttpServer::SubmitHttp(THttpCallArg *arg, Bool_t can_run_immediately = kFALSE, Bool_t ownership = kFALSE);
   * Bool_t THttpServer::ExecuteHttp(THttpCallArg *arg)
   * Bool_t TRootSniffer::Produce(const char *path, const char *file, const char *options, void *&ptr, Long_t &length, TString &str);
   * TString THttpCallArg::GetPostDataAsString();
   * void THttpCallArg::FillHttpHeader(TString &buf, const char *header = nullptr);
   * void THttpCallArg::SetBinData(void *data, Long_t length);

The methods could be replaced by equivalent methods with other signature:

   * Bool_t THttpServer::SubmitHttp(std::shared_ptr<THttpCallArg> arg, Bool_t can_run_immediately = kFALSE);
   * Bool_t THttpServer::ExecuteHttp(std::shared_ptr<THttpCallArg> arg);
   * Bool_t TRootSniffer::Produce(const std::string &path, const std::string &file, const std::string &options, std::string &res);
   * const void *THttpCallArg::GetPostData() const;
   * Long_t THttpCallArg::GetPostDataLength() const;
   * std::string THttpCallArg::FillHttpHeader(const char *header = nullptr);
   * void THttpCallArg::SetContent(std::string &&cont);


## Core Libraries


## I/O Libraries


## TTree Libraries

### RDataFrame
  - Use TPRegexp instead of TRegexp to interpret the regex used to select columns
    in the invocation of `Cache` and `Snapshot`. This allows usage of a larger set
    of regular expressions to specify which columns should be written out.
  - Speed up jitting of Filter and Define expressions passed as string
  - Speed up event loop, improve scaling in the presence of a large amount of Defines
  - Allow Filter expressions to return types convertible to bool, rather than only bool
  - Add `GetNSlots` method to easily retrieve the number of slots that will be used by
    `DefineSlot, `ForeachSlot`, `OnPartialResultSlot`, ...
  - Add support for TTrees/TChains with TEntryLists (currently only for single-thread event loops)
  - Add `HasColumn` method to check whether a column is available to a given RDF node
  - PyROOT: add `AsRNode` helper function to convert RDF nodes to the common RNode type
  - PyROOT: add `AsNumpy` method to export contents of a RDataFrame as a dictionary of numpy arrays


## Histogram Libraries


## Math Libraries


## RooFit Libraries
### HistFactory
hist2workspace performance optimisations. For a large, ATLAS-style Higgs-->bb workspace with > 100 systematic uncertainties and more than 10 channels, the run time decreases by a factor 11 to 12.

### Faster, STL-like Collections in RooFit
RooFit's collections `RooArgSet` and `RooArgList` have been made more STL-like. The underlying implementation used to be the `RooLinkedList`, but now both collections work with `std::vector`. The collections have an STL-like interface concerning iterators such that iterations over the two collections that looked like
```
TIterator* depIter = intDepList.createIterator() ;
RooAbsArg* arg;
while((arg=(RooAbsArg*)depIter->Next())) {
  ...
}
delete depIter;
```
now look like:
```
for (auto arg : intDepList) {
  ...
}
```
Depending on how many elements are iterated, RooFit will be between 10 and 20% faster if the new iterators are used. Heavily using old iterators might slow it down by 5 to 10%. Iterators in key classes have been updated, such that many workflows in RooFit are 10 - 20% faster.

#### Legacy iterators
The (three kinds) of legacy iterators in RooFit are still supported, such that old code will not break, but they are slower than `begin(), end()` and range-based for loops.

**Important caveat**:
The old RooFit collections could be modified while iterating. The STL-like iterators do not support this (as for a *e.g.* std::vector)! Using the legacy iterators with the new collections (*i.e.* in existing code), mutating the collection is still possible in the following cases:
- Inserting/deleting elements **after** the current iterator.
- Changing an element at a position **other than** the current iterator
- **But not** inserting/deleting before/at the current iterator position. With a debug build (with assertions), the legacy iterators will check that the collection is not mutated. In a release build, elements might be skipped or be iterated twice.

#### Moving away from the slower iterators
The legacy iterators have been flagged with a special deprecation macro that can be used help the user use the recommended ROOT interface. Defining `R__SUGGEST_NEW_INTERFACE`, (either in a single translation unit or in the build system), all uses of the legacy iterators will trigger a compiler warning like:
```
<path>/RooChebychev.cxx:66:34: warning: 'createIterator' is deprecated: There is a superior alternative: begin(), end() and range-based for loops. [-Wdeprecated-declarations]
  TIterator* coefIter = coefList.createIterator() ;
                                 ^
1 warning generated.
```




## 2D Graphics Libraries

  - In the statistics painting for 2D histograms, the central cell of
    the underflow/overflow grid was not properly rendered for very large contents.
    This problem was reported [here](https://root-forum.cern.ch/t/stat-box-for-th2/).
  - The automatic placement of legend now "sees" TMultiGraph and THStack.
  - Improve and simplify the drawing the 2D histogram's option "ARR".
  - The option ARR can be combined with the option COL or COLZ.
  - `TBox::DistancetoPrimitive` and `TBox::ExecuteEvent` now work in log scales (by Jérémie Dudouet).
  - Take the line attributes into account when drawing a histogram with option bar or hbar.
    They were ignored until now.
  - The new draw option MIN0 makes same effect as gStyle->SetHistMinimumZero(1), but can be specified
    individually for each histogram.
  - Improve the line clipping when a histogram is drawn with option "L". The following
    example shows the improvement.
~~~ {.cpp}
      {
         auto h = new TH1F("h","h",5,0.5,5.5);
         h->SetBinContent(1,100000);
         h->SetBinContent(2,10000);
         h->SetBinContent(3,1000);
         h->SetBinContent(4,100);
         h->SetBinContent(5,10);
         h->SetMinimum(50.);
         h->SetMaximum(40000);
         h->Draw("L*");
         gPad->SetLogy();
      }
~~~

## 3D Graphics Libraries

  - Make sure a TF3 is painted the same way in GL and non-GL mode.
    The mismatch was reported in [this post](https://root-forum.cern.ch/t/how-to-specify-the-level-value-in-isosurface-drawing-with-tf3-and-gl/32179)

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


