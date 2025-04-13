{
auto f = new TFile("sub.root","RECREATE");
f->mkdir("sub");
f->cd("sub");
auto t = new TTree("tree","tree");
int i = 3;
t->Fill();
t->Branch("i",i);
t->Fill();
t->Fill();
f->Write();
delete f;

TChain *data = new TChain("sub/tree");
data->Add("sub*.root");
data->Draw(">>myListData","1","entrylistdata");
TEntryList *listData=(TEntryList*)gDirectory->Get("myListData");
data->SetEntryList(listData);

#ifdef ClingWorkAroundBrokenUnnamedReturn
int res = 0;
#else
return 0;
#endif
}
