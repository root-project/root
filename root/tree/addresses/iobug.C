#include "TGraphErrors.h"
#include "TFile.h"
#include "TTree.h"
void iobug(int split=0, int classtype = 0, int clonesmode = 0, int show = 0, int dumpmode = 0) {
   // root -b -q iobug.C(0,0)  OK
   // root -b -q iobug.C(1,0)  crash
   // root -b -q iobug.C(2,0)  OK
   // root -b -q iobug.C(0,1)  OK
   // root -b -q iobug.C(1,1)  Bad numerical expressions
   // root -b -q iobug.C(2,1)  wrong result
   TGraph *g;
   TGraph *g2;
   TGraph *g3;
   if (clonesmode==0) clonesmode = 1;
   TClonesArray *clones = 0;
   if (classtype == 0 ) {
      g = new TGraph(2);
      clones = new TClonesArray("TGraph");
      new ((*clones)[0]) TGraph(2);
      g2 = (TGraph*)((*clones)[0]);
      new ((*clones)[1]) TGraph(2);
      g3 = (TGraph*)((*clones)[1]);
   } else {
      g = new TGraphErrors(2);
      clones = new TClonesArray("TGraphErrors");
      new ((*clones)[0]) TGraphErrors(2);
      g2 = (TGraphErrors*)((*clones)[0]);
      new ((*clones)[1]) TGraphErrors(2);
      g3 = (TGraphErrors*)((*clones)[1]);
   }
   g->SetPoint(0,1,2);
   g->SetPoint(1,3,4);
   g->SetMarkerColor(2);
   g->SetMarkerSize(1.2);
   g->SetMarkerStyle(21);

   g2->SetPoint(0,1,2);
   g2->SetPoint(1,3,4);
   g2->SetMarkerColor(2);
   g2->SetMarkerSize(1.4);
   g2->SetMarkerStyle(27);

   g3->SetPoint(7,8,9);
   g3->SetPoint(10,13,14);
   g3->SetMarkerColor(3);
   g3->SetMarkerSize(1.5);
   g3->SetMarkerStyle(30);
   delete gFile;
   TFile *f = new TFile("problem.root", "RECREATE");                      
   TTree *t = new TTree("graphs", "problematic graphs");                
   if (clonesmode&0x1) t->Branch("graph",g->ClassName(), &g, 32000, split); 
   if (clonesmode&0x2) t->Branch("graphCl",&clones, 32000, split); 

   t->Fill();
   g->SetMarkerColor(3);
   g->SetMarkerSize(1.3);
   g->SetMarkerStyle(24);

   g2->SetMarkerColor(4);
   g2->SetMarkerSize(1.6);
   g2->SetMarkerStyle(33);

   g3->SetMarkerColor(5);
   g3->SetMarkerSize(1.7);
   g3->SetMarkerStyle(36);

   t->Fill();
   if (show) t->Show(0);
   t->Write();
   if (dumpmode&0x1) g->Dump();
   if (clonesmode&0x1) {
      delete g; g = 0;
      t->SetBranchAddress("graph",&g);
   }
   if (clonesmode&0x2) {
      delete clones; clones = 0;
      t->SetBranchAddress("graphCl",&clones);
   }
   t->GetEntry(0);
   if (dumpmode&0x2) g->Dump();
   //f.Write();
   if (dumpmode&0x4) t->Print();
   if (clonesmode&0x1) t->Scan("fMarkerColor:fMarkerSize:graph.fMarkerStyle","","colsize=20");
   if (clonesmode&0x2) t->Scan("fMarkerColor:graphCl.fMarkerSize:graphCl.fMarkerStyle","","colsize=20");
}

