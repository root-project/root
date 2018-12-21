## Tree Libraries

### TTreeReader

ROOT offers a new class `TTreeReader` that gives simple, safe and fast access to the content of a `TTree`.
Using it is trivial:

``` {.cpp}
#include "TFile.h"
#include "TH1F.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"

void hsimpleReader() {
   TH1F *myHist = new TH1F("h1","ntuple",100,-4,4);
   TFile *myFile = TFile::Open("hsimple.root");

   // Create a TTreeReader for the tree, for instance by passing the
   // TTree's name and the TDirectory / TFile it is in.
   TTreeReader myReader("ntuple", myFile);

   // The branch "px" contains floats; access them as myPx.
   TTreeReaderValue<Float_t> myPx(myReader, "px");
   // The branch "py" contains floats, too; access those as myPy.
   TTreeReaderValue<Float_t> myPy(myReader, "py");

   // Loop over all entries of the TTree or TChain.
   while (myReader.Next()) {
      // Just access the data as if myPx and myPy were iterators (note the '*'
      // in front of them):
      myHist->Fill(*myPx + *myPy);
   }

   myHist->Draw();
}
```

TTreeReader checks whether the type that you expect can be extracted from the tree's branch and will clearly complain if not.
It reads on demand: only data that are actually needed are read, there is no need for `SetBranchStatus()`, `SetBranchAddress()`, `LoadTree()` or anything alike.
It uses the memory management of TTree, removing possible double deletions or memory leaks and relieveing you from the need to manage the memory yourself.
It turns on the tree cache, accelerating the reading of data.
It has been extensively tested on all known types of TTree branches and is thus a generic, fits-all access method for data stored in TTrees.


### TTreePlayer

-   The TEntryList for ||-Coord plot was not defined correctly.

