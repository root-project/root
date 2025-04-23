#include "colClass1.h"
#include "TFile.h"
#include "TTree.h"

void writeMissingCollection() 
{
   TFile f("missingCollection.root","RECREATE");
   colClass obj;
   TTree *tree = new TTree("T","T");
   tree->Branch("obj.",&obj);
   obj.Fill(10);
   tree->Fill();
   tree->Fill();
   f.Write();
};
