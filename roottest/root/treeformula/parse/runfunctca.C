#include <iostream>
#include <TROOT.h>
#include <TFile.h>
#include <TTree.h>
#include "EventTcaMember.h"

void runfunctca()
{
#ifndef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine(".L EventTcaMember.cc+");
#endif
   TFile *f = new TFile("test.root", "RECREATE");
   TTree *tree = new TTree("tree", "TTree with TClonesArray member");
   EventTcaMember *e = new EventTcaMember(10);
   tree->Branch("MyEvent", &e);
   
   new(e->tca[0]) Track(100);
   tree->Fill();
   f->Write();
   delete f;
   f = new TFile("test.root");
   tree = (TTree *)f->Get("tree");
   tree->Draw("p");
   std::cerr << "tree->Draw(\"cos(p)\")\n";
   tree->Draw("cos(p)");
}

