{
TChain* c = new TChain("NtpSt","NtpSt");
c->Add("ver_40200.root");
c->Add("ver_40200_copy.root");
TFile* newfile = new TFile("CloneTree.root","recreate");
TTree* tc = c->CloneTree(-1,"fast");
tc->Write();
return 0;
}

