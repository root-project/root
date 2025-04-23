{
TChain* c = new TChain("NtpSt","NtpSt");
c->Add("ver_40200.root");
c->Add("ver_40200_copy.root");
TFile* newfile = new TFile("CloneTree.root","recreate");
TTree* tc = c->CloneTree(-1,"fast");
tc->Write();
delete tc->GetCurrentFile();
delete c;
#ifdef ClingWorkAroundBrokenUnnamedReturn
gApplication->Terminate(0);
#else
return 0;
#endif
}
