{
#ifndef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine(".L Embedded_load.C+");
#endif
TFile *f    = new TFile("Embedded.root");
TTree *tree = (TTree*)f->Get("T");
Normal_objects* obj = new Normal_objects();
TBranch *branch  = tree->GetBranch("B");
branch->SetAddress(&obj);
Int_t nevent = (Int_t)(tree->GetEntries());
Int_t nselected = 0;
Int_t nb = 0;
for (Int_t i=0;i<nevent;i++) { printf("%d %d \n", i, tree->GetEvent(i)); obj->dump(); }
}
