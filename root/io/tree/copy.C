{
gROOT->ProcessLine(".L classes.C+");
TEmcl *e = 0;
TNonEmcl *ne = 0;


TFile *oldfile = new TFile("test.root");
TTree *oldtree = (TTree*)oldfile->Get("T");

oldtree->SetBranchAddress("emcl",&e);
oldtree->SetBranchAddress("nonemcl",&ne);

TFile *file = new TFile("copy.root","RECREATE");
TTree *tree = oldtree->CloneTree(0);

oldtree->GetEntry(0);
tree->Fill();

file->Write();
}

