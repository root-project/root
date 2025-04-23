#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TROOT.h"

void writefile(const char *filename = "forchain.root") {
   TFile *f = TFile::Open(filename,"RECREATE");
   TTree *t = new TTree("t","t");
   int i;
   t->Branch("i",&i);
   t->Fill();
   f->Write();
   delete f;
}

int readchain(const char *filename = "forchain.root" ) {
   TChain *c = new TChain("t");
   c->Add(filename);
   c->GetEntry(0);
   
   // Emulate shutdown
   gROOT->GetListOfFiles()->Delete();
   // delete c->GetFile();

   if (c->GetCurrentFile() != 0) {
      fprintf(stdout,"Error the chain still point to the TFile\n");
      return 1;
   }
   
   delete c;
   return 0;
}

int execCleanup() {
   writefile();
   return readchain();
}
