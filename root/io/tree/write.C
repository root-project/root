{
gROOT->ProcessLine(".L classes.C+");
TEmcl *e = new TEmcl;
e->e = 2;
TNonEmcl *ne = new TNonEmcl;
ne->e = 3;

TFile *file = new TFile("test.root","RECREATE");
TTree *tree = new TTree("T","T");
tree->Branch("emcl","TEmcl",&e);
tree->Branch("nonemcl","TNonEmcl",&e);
tree->Fill();
file->Write();
file->Close();
}
