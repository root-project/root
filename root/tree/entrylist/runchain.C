{
#ifdef ClingWorkAroundMissingImplicitAuto
TFile *f;
TTree *t;
#endif
f = new TFile("sub.root","RECREATE");
f->mkdir("sub");
f->cd("sub");
t = new TTree("tree","tree");
int i = 3;
t->Fill();
t->Branch("i",i);
t->Fill();
t->Fill();
f->Write();
delete f;

#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
TChain *data; data = new TChain("sub/tree");
data->Add("sub*.root");
data->Draw(">>myListData","1","entrylistdata");
TEntryList *listData ; listData =(TEntryList*)gDirectory->Get("myListData");
data->SetEntryList(listData);   
#else
TChain *data = new TChain("sub/tree");
data->Add("sub*.root");
data->Draw(">>myListData","1","entrylistdata");
TEntryList *listData=(TEntryList*)gDirectory->Get("myListData");
data->SetEntryList(listData);
#endif

#ifdef ClingWorkAroundBrokenUnnamedReturn
int res = 0;
#else
return 0;
#endif
}
