#include "TFile.h"
#include "TTree.h"
#include "TBrowser.h"
#include "TH2.h"
#include "TRandom.h"
#include "TClassTable.h"
#include "TSystem.h"
#include "TTreePerfStats.h"
#include "TROOT.h"
#include <iostream>

#include "Event.h"

bool write()
{
   // check to see if the event class is in the dictionary
   // if it is not load the definition in libEvent.so
   if (!TClassTable::GetDict("Event")) {
      std::cerr << "Dictionary for Event class is not provided!" << std::endl;
      return false;
   }

  //create a Tree file tree4.root
  TFile f("perftest.root","RECREATE");

  // Create 2 ROOT Trees
  TTree tree1("tree1","One of two simultaneous trees");
  TTree tree2("tree2","A tree made simultaneously with tree 1");

  // Create a pointer to an Event object
  Event *event1 = new Event();
  Event *event2 = new Event();

  // Create single branch of different split levels for each tree.
  tree1.Branch("event", &event1, 8000,2);
  tree2.Branch("event2", &event2, 8000,2);

  // a local variable for the event type
  char etype[20];

  // Fill the tree
  for (Int_t i = 0; i <20; i++) {
    Double_t rand1 = gRandom->Landau(0, 1);
    Double_t rand2 = gRandom->Gaus(0, 1);
    Int_t ntrack   = Int_t(600 + 600 *rand1/120.);
    Float_t random = gRandom->Rndm(1);
    snprintf(etype,20,"type%d",i/5);
    event1->SetType(etype);
    event1->SetHeader(i, 200, 960312, random);
    event1->SetNseg(Int_t(10*ntrack+20*rand2));
    event1->SetNvertex(Int_t(1+20*gRandom->Rndm()));
    event1->SetFlag(UInt_t(random+0.5));
    event1->SetTemperature(random+20.);

    event2->SetType(etype);
    event2->SetHeader(i, 200, 960312, random);
    event2->SetNseg(Int_t(10*ntrack+20*rand2));
    event2->SetNvertex(Int_t(1+20*gRandom->Rndm()));
    event2->SetFlag(UInt_t(random+0.5));
    event2->SetTemperature(random+20.);

    for(UChar_t m = 0; m < 10; m++) {
      event1->SetMeasure(m, Int_t(gRandom->Gaus(m,m+1)));
      event2->SetMeasure(m, Int_t(gRandom->Gaus(m,m+1)));
    }

    // fill the matrix
    for(UChar_t j = 0; j < 4; j++) {
      for(UChar_t k = 0; k < 4; k++) {
        event1->SetMatrix(j,k,gRandom->Gaus(k*j,1));
        event2->SetMatrix(j,k,gRandom->Gaus(k*j,1));
      }
    }

    //  Create and fill the Track objects
    for (Int_t t = 0; t < ntrack; t++){
      event1->AddTrack(random);
      event2->AddTrack(random);}

    // Fill the tree
    tree1.Fill();
    tree2.Fill();

    // Clear the event before reloading it
    event1->Clear();
    event2->Clear();
  }

  // Write the file header
  f.Write();
  return true;
}

void simultaneous()
{

   TFile *f = TFile::Open("perftest.root");

   auto S = (TTree*)f->Get("tree1");
   auto T = (TTree*)f->Get("tree2");
   Long64_t nentries = T->GetEntries();

   S->SetCacheSize(10000000);
   S->SetCacheEntryRange(0, nentries);
   S->AddBranchToCache("*");

   T->SetCacheSize(10000000);
   T->SetCacheEntryRange(0, nentries);
   T->AddBranchToCache("*");


   TTreePerfStats *ps1 = new TTreePerfStats("io1", S);
   TTreePerfStats *ps2 = new TTreePerfStats("io2", T);


   for (Int_t i = 0; i < nentries; i ++){
      S->GetEntry(i);
      T->GetEntry(i);
   }

   ps1->SaveAs("tree1ps.root");
   ps2->SaveAs("tree2ps.root");
   cout<<ps1->GetReadCalls()<<endl;
   cout<<ps2->GetReadCalls()<<endl;
  // ps1->Print();
  // ps2->Print();
}


int execperfstattest()
{
   // Event::Reset(); // Allow for re-run this script by cleaning static variables.
   if (!write())
      return 1;
   // Event::Reset(); // Allow for re-run this script by cleaning static variables.
   simultaneous();
   return 0;
}
