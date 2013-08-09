#include "TFile.h"
#include "TH1F.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"

void TreeReaderSimple() {
   TH1F *myHist = new TH1F("h1","ntuple",100,-4,4);

   TFile *myFile = TFile::Open("hsimple.root");
   TTreeReader myReader("ntuple", myFile);

   TTreeReaderValue<Float_t> myPx(myReader, "px");
   TTreeReaderValue<Float_t> myPy(myReader, "py");

   while (myReader.Next()) {
      myHist->Fill(*myPx + *myPy);
   }

   myHist->Draw();
}
