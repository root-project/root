{
#ifndef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine(".L Embedded_load.C+");
#endif
   TFile *f    = new TFile("Embedded.root", "RECREATE", "Root Embedded test");
   TTree *tree = new TTree("T","An example of a ROOT tree");
   Normal_objects* obj = new Normal_objects();
   TBranch *b = tree->Branch("B", "Normal_objects", &obj);
   
   for(int i=0; i < 10; ++i) {
      obj->initData(i);
      printf("%d\n",tree->Fill());
   }
   
   b->Print();
   b->Write();
   tree->Print();
   tree->Write();
   f->Print();
   f->Close();
   delete f;
}
