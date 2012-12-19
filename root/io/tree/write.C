{
#ifndef SECOND_RUN
gROOT->ProcessLine(".L classes.C+");

#ifdef ClingWorkAroundMissingAutoLoading
gSystem->Load("libTree");
#endif
#endif

#if defined(ClingWorkAroundMissingDynamicScope) && !defined(SECOND_RUN)
#define SECOND_RUN
      gROOT->ProcessLine(".x write.C");
#else
      
TEmcl *e = new TEmcl;
e->e = 2;
TNonEmcl *ne = new TNonEmcl;
ne->e = 3;

TFile *file = new TFile("test.root","RECREATE");
TTree *tree = new TTree("T","T");
tree->Branch("emcl","TEmcl",&e);
tree->Branch("nonemcl","TNonEmcl",&ne);
tree->Fill();
file->Write();
file->Close();
      
#endif
      
}
