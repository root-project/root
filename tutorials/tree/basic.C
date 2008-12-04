#include "Riostream.h"
void basic() {
//  Read data from an ascii file and create a root file with an histogram and an ntuple.
//   see a variant of this macro in basic2.C
//Author: Rene Brun
      

// read file $ROOTSYS/tutorials/tree/basic.dat
// this file has 3 columns of float data
   TString dir = gSystem->UnixPathName(gInterpreter->GetCurrentMacroName());
   dir.ReplaceAll("basic.C","");
   dir.ReplaceAll("/./","/");
   ifstream in;
   in.open(Form("%sbasic.dat",dir.Data()));

   Float_t x,y,z;
   Int_t nlines = 0;
   TFile *f = new TFile("basic.root","RECREATE");
   TH1F *h1 = new TH1F("h1","x distribution",100,-4,4);
   TNtuple *ntuple = new TNtuple("ntuple","data from ascii file","x:y:z");

   while (1) {
      in >> x >> y >> z;
      if (!in.good()) break;
      if (nlines < 5) printf("x=%8f, y=%8f, z=%8f\n",x,y,z);
      h1->Fill(x);
      ntuple->Fill(x,y,z);
      nlines++;
   }
   printf(" found %d points\n",nlines);

   in.close();

   f->Write();
}
