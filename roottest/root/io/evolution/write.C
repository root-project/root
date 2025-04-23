{
#ifndef ClingWorkAroundMissingDynamicScope
gROOT->ProcessLine(".L MyClassOld.cxx+");
#endif
TFile * file = new TFile("data.root","RECREATE");
MyClass *m;
TClonesArray *tca = new TClonesArray("MyClass");
m = new ((*tca)[0]) MyClass(5);
m = new ((*tca)[1]) MyClass(6);
tca->Write("collection",TObject::kSingleKey);
m = new MyClass(5);
m->Write("myobj");

TClonesArray *tca2 = new TClonesArray("Cont");
Cont* c;
c = new ((*tca2)[0]) Cont(7);
c = new ((*tca2)[1]) Cont(8);

TTree * tree = new TTree("T","T");
tree->Branch("conts","TClonesArray",&tca2);
tree->Branch("objs","TClonesArray",&tca);
tree->Branch("obj","MyClass",&m);
tree->Fill();
file->Write();
file->Close();
}
