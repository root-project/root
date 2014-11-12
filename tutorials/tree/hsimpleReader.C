// A simple TTreeReader use: read data from hsimple.root (written by hsimple.C)
// Author: Anders Eie, 2013.

#include "TFile.h"
#include "TH1F.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"

class TVirtualPad;

void hsimpleReader() {
   // Create a histogram for the values we read.
   TH1F *myHist = new TH1F("h1","ntuple",100,-4,4);

   // Open the file containing the tree.
   TFile *myFile = TFile::Open("hsimple.root");
   if (!myFile || myFile->IsZombie()) {
      return;
   }
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
